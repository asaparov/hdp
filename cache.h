/**
 * \file cache.h
 *
 * This file contains the ::cache structure, which optimizes MCMC inference for
 * the HDP. This structure is specialized for various combinations of
 * `BaseDistribution` and `DataDistribution`, and the specializations are also
 * implemented in this file.
 *
 *  <!-- Created on: Jul 11, 2016
 *           Author: asaparov -->
 */

#ifndef CACHE_H_
#define CACHE_H_

#include <math/distributions.h>

/* forward declarations */
#if !defined(DOXYGEN_IGNORE)
template<typename K, typename V> struct node_sampler;
template<typename BaseDistribution, typename DataDistribution, typename K, typename V> struct hdp_sampler;
#endif


template<typename K>
constexpr unsigned int count(const K& observation) {
	return 1;
}

template<typename K>
inline unsigned int count(const array_histogram<K>& observations) {
	return observations.sum;
}

template<typename V>
inline void cleanup_root_probabilities(V** root_probabilities, unsigned int row_count)
{
	for (unsigned int i = 0; i < row_count; i++)
		free(root_probabilities[i]);
	free(root_probabilities);
}

inline void cleanup_root_probabilities(array<unsigned int>* root_probabilities, unsigned int row_count)
{
	for (unsigned int i = 0; i < row_count; i++)
		free(root_probabilities[i]);
	free(root_probabilities);
}

/* a default implementation that various cache implementations can use */
template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
V** compute_root_probabilities(const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h, const K& observation)
{
	/* TODO: there's a potential speedup with a single contiguous array */
	/* initialize the root_probabilities array of arrays */
	V** root_probabilities = (V**) malloc(sizeof(V*) * h.posterior.length);
	if (root_probabilities == NULL) {
		fprintf(stderr, "predict ERROR: Insufficient memory for root_probabilities.\n");
		return NULL;
	}

	unsigned int sample_count = 0;
	for (unsigned int i = 0; i < h.posterior.length; i++) {
		root_probabilities[i] = (V*) malloc(sizeof(V) * h.posterior[i].table_count);
		if (root_probabilities[i] == NULL) {
			fprintf(stderr, "predict ERROR: Insufficient memory for root_probabilities[%u].\n", i);
			cleanup_root_probabilities(root_probabilities, sample_count);
			return NULL;
		}
		sample_count++;
	}

	/* store the appropriate conditional probabilities in root_probabilities */
	for (unsigned int i = 0; i < h.posterior.length; i++) {
		for (unsigned int j = 0; j < h.posterior[i].table_count; j++)
			/* TODO: test numerical stability */
			if (h.posterior[i].descendant_observations[j].counts.size == 0) {
				/* theoretically, this should be the prior probability of 'observation',
					but since the table is empty, its likelihood will be zero anyway */
				root_probabilities[i][j] = 0.0;
			} else {
				root_probabilities[i][j] = DataDistribution::conditional(
						h.pi(), observation, h.posterior[i].descendant_observations[j]);
			}
	}

	return root_probabilities;
}

/* a default implementation that various cache implementations can use */
template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
V** compute_root_probabilities(
	const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	const K* observations, unsigned int observation_count)
{
	V** probabilities = (V**) malloc(sizeof(V*) * observation_count);
	if (probabilities == NULL) {
		fprintf(stderr, "cache.compute_root_probabilities ERROR: Insufficient memory for matrix.\n");
		return NULL;
	}
	for (unsigned int i = 0; i < observation_count; i++) {
		probabilities[i] = (V*) malloc(sizeof(V) * h.table_count);
		if (probabilities[i] == NULL) {
			fprintf(stderr, "cache.compute_root_probabilities ERROR: Insufficient memory for matrix row.\n");
			return NULL;
		}
		for (unsigned int j = 0; j < h.table_count; j++) {
			if (h.descendant_observations[j].counts.size == 0)
				/* theoretically, this should be the prior probability, but
					since the table is empty, the likelihood will be zero anyway */
				probabilities[i][j] = 0.0;
			else probabilities[i][j] = DataDistribution::conditional(h.pi(), observations[i], h.descendant_observations[j]);
		}
	}
	return probabilities;
}

/**
 * This structure is used to provide optimizations for the Gibbs sampler in
 * mcmc.h for specific combinations of `BaseDistribution` and
 * `DataDistribution`. This struct is specialized for those particular
 * combinations of base distribution and likelihood. These specializations are
 * implemented further down in this file. Currently, a default implementation
 * is unavailable.
 */
template<typename BaseDistribution, typename DataDistribution, typename K, typename V, class Enable = void>
struct cache {
	/* TODO: create a default implementation */
	typedef hdp_sampler<BaseDistribution, dense_categorical<V>, unsigned int, V> sampler_root;

	/**
	 * Constructs a cache structure for the HDP sampler hierarchy rooted at `h`.
	 */
	cache(const sampler_root& h) {
		fprintf(stderr, "cache ERROR: Unimplemented. This combination of "
			"BaseDistribution and DataDistribution is currently unsupported.\n");
		exit(EXIT_FAILURE);
	}

	/**
	 * Computes a matrix, where every row corresponds to a
	 * sample from the posterior, and every column is a table
	 * at the root node. Each element contains the probability
	 * of assigning the given observation to each table at the
	 * root (without taking into account table sizes and the
	 * alpha parameter).
	 */
	inline V** compute_root_probabilities(const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h, const K& observation) {
		return ::compute_root_probabilities(h, observation);
	}

	/**
	 * Constructs a map from observations to a vector, of which
	 * each element is the probability of assigning that
	 * observation to each table at the root (without taking
	 * into account table sizes and the alpha parameter).
	 */
	inline V** compute_root_probabilities(
		const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
		const K* observations, unsigned int observation_count)
	{
		return ::compute_root_probabilities(h, observations, observation_count);
	}
};

/* this structure caches values of the log rising factorial
   log (pi + i)^(n) for various values of i and n */
template<typename V>
struct rising_factorial_cache {
	array_map<unsigned int, array<V>> values;

	V pi;

	rising_factorial_cache(const V& pi) :
		values(64), pi(pi)
	{ }

	~rising_factorial_cache() { free(); }

	inline const V* get_rising_factorials(unsigned int exponent, unsigned int largest_table_size) {
		if (!values.ensure_capacity((unsigned int) values.size + 1))
			return NULL;

		bool contains;
		array<V>& log_factorials = values.get(exponent, contains);
		if (!contains) {
			if (!array_init(log_factorials, largest_table_size + 1)) {
				fprintf(stderr, "rising_factorial_cache.get_rising_factorials ERROR: Out of memory.\n");
				return NULL;
			}

			values.keys[values.size] = exponent;
			values.size++;
		} else if (!log_factorials.ensure_capacity(largest_table_size + 1)) {
			fprintf(stderr, "rising_factorial_cache.get_rising_factorials"
					" ERROR: Unable to expand array of log factorials.\n");
			return NULL;
		}

		for (unsigned int i = (unsigned int) log_factorials.length; i < largest_table_size + 1; i++)
			log_factorials[i] = log_rising_factorial(pi + i, exponent);
		if (log_factorials.length < largest_table_size + 1)
			log_factorials.length = largest_table_size + 1;

		const V* internal_array = log_factorials.data;
		if (!contains)
			sort(values.keys, values.values, (unsigned int) values.size, default_sorter());
		return internal_array;
	}

	bool is_valid() const {
		for (unsigned int i = 0; i < values.size; i++)
			if (!is_valid(pi, values.keys[i], values.values[i]))
				return false;
		return true;
	}

	static bool is_valid(const V& pi, unsigned int exponent, const array<V>& log_factorials) {
		for (unsigned int i = 0; i < log_factorials.length; i++) {
			if (log_factorials[i] != log_rising_factorial(pi + i, exponent)) {
				fprintf(stderr, "rising_factorial_cache.is_valid ERROR: Incorrect "
						"log rising factorial for base %lf + %u and exponent %u.\n", pi, i, exponent);
				return false;
			}
		}
		return true;
	}

	template<typename Metric>
	static inline long unsigned int size_of(const rising_factorial_cache<V>& cache, const Metric& metric) {
		return core::size_of(cache.values, make_key_value_metric(default_metric(), metric)) + core::size_of(cache.pi);
	}

	static inline void free(rising_factorial_cache<V>& cache) {
		cache.free();
		core::free(cache.values);
	}

private:
	inline void free() {
		for (unsigned int i = 0; i < values.size; i++)
			core::free(values.values[i]);
	}
};

template<typename V>
inline bool init(rising_factorial_cache<V>& cache, const V& pi) {
	cache.pi = pi;
	return array_map_init(cache.values, 64);
}

template<typename V>
struct table_distribution {
	/* *denominators* of root-level table assignment likelihoods
	   (for a fixed size of the observation set) */
	array_categorical<V> likelihoods;

	/* *denominators* of root-level table assignment probabilities
	   (the above likelihoods multiplied by the table sizes/alpha) */
	V* probabilities;
	V sum;

	template<typename BaseDistribution, typename DataDistribution, typename K>
	inline void set_higher(
			const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
			unsigned int table, const V& log_probability)
	{
#if !defined(NDEBUG)
		if (log_probability < likelihoods.log_probabilities[table])
			fprintf(stderr, "table_distribution.set_higher WARNING: Given probability is smaller than the current value.\n");
#endif
		likelihoods.log_probabilities[table] = log_probability;
		if (likelihoods.log_probabilities[table] - likelihoods.maximum > CATEGORICAL_MAX_THRESHOLD) {
			renormalize(h);
		} else {
			likelihoods.probabilities[table] = exp(likelihoods.log_probabilities[table] - likelihoods.maximum);
			sum -= probabilities[table];
			probabilities[table] = h.table_sizes[table] * likelihoods.probabilities[table];
			sum += probabilities[table];
		}
	}

	template<typename BaseDistribution, typename DataDistribution, typename K>
	inline void set_lower(const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
			unsigned int table, const V& log_probability)
	{
#if !defined(NDEBUG)
		if (log_probability > likelihoods.log_probabilities[table])
			fprintf(stderr, "table_distribution.set_lower WARNING: Given probability is larger than the current value.\n");
#endif
		likelihoods.log_probabilities[table] = log_probability;
		likelihoods.probabilities[table] = exp(likelihoods.log_probabilities[table] - likelihoods.maximum);
		sum -= probabilities[table];
		probabilities[table] = h.table_sizes[table] * likelihoods.probabilities[table];
		sum += probabilities[table];

		if (sum < CATEGORICAL_MIN_THRESHOLD)
			renormalize(h);
	}

	/* NOTE: this function assumes the new table has already been created in the hdp */
	template<typename BaseDistribution, typename DataDistribution, typename K>
	inline bool add_new_table(
		const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
		unsigned int count, unsigned int observation_count)
	{
		if (!likelihoods.resize(h.table_count))
			return false;
		likelihoods.log_probabilities[h.table_count - 1] =
			-log_rising_factorial(h.pi().sum() + count, observation_count);
		likelihoods.probabilities[h.table_count - 1] =
			exp(likelihoods.log_probabilities[h.table_count - 1] - likelihoods.maximum);

		V* new_probabilities = (V*) realloc(probabilities, sizeof(V) * h.table_count);
		if (new_probabilities == NULL) {
			fprintf(stderr, "table_distribution.add_new_table ERROR: Out of memory.\n");
			return false;
		}
		probabilities = new_probabilities;
		probabilities[h.table_count - 1] = h.table_sizes[h.table_count - 1] * likelihoods.probabilities[h.table_count - 1];
		sum += probabilities[h.table_count - 1];
		return true;
	}

	template<bool Check, typename BaseDistribution, typename DataDistribution, typename K>
	inline void change_descendant_count(
		const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
		unsigned int table, unsigned int new_count, unsigned int observation_count)
	{
		V new_likelihood = -log_rising_factorial(h.pi().sum() + new_count, observation_count);
		if (Check) {
			set_lower(h, table, new_likelihood);
		} else {
			likelihoods.log_probabilities[table] = new_likelihood;
			likelihoods.probabilities[table] = exp(likelihoods.log_probabilities[table] - likelihoods.maximum);
			sum -= probabilities[table];
			probabilities[table] = h.table_sizes[table] * likelihoods.probabilities[table];
			sum += probabilities[table];
		}
	}

	inline void change_table_size(unsigned int table, unsigned int new_table_size) {
		sum -= probabilities[table];
		probabilities[table] = new_table_size * likelihoods.probabilities[table];
		sum += probabilities[table];
	}

	template<typename BaseDistribution, typename DataDistribution, typename K>
	bool is_valid(
		const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
		unsigned int observation_count) const
	{
		V actual_sum = 0.0;
		for (unsigned int i = 0; i < h.table_count; i++) {
			V expected = -log_rising_factorial(
					h.pi().sum() + h.descendant_observations[i].sum, observation_count);
			if (h.table_sizes[i] > 0 && fabs(likelihoods.log_probabilities[i] - expected) > fabs(expected) * 1.0e-12) {
				fprintf(stderr, "table_distribution.is_valid ERROR: Incorrect log probability for table %u.\n", i);
				return false;
			}
			V diff = fabs(likelihoods.probabilities[i] * h.table_sizes[i] - probabilities[i]);
			if (diff > probabilities[i] * 1.0e-12 && diff > 1.0e-12) {
				fprintf(stderr, "table_distribution.is_valid ERROR: Incorrect probability for table %u.\n", i);
				return false;
			}
			actual_sum += probabilities[i];
		}
		if (fabs(actual_sum - sum) > sum * 1.0e-6 && fabs(actual_sum - sum) > 1.0e-6) {
			fprintf(stderr, "table_distribution.is_valid ERROR: Incorrect probability sum.\n");
			return false;
		}
		return true;
	}

	void move_table(unsigned int src, unsigned int dst) {
		probabilities[dst] = probabilities[src];
		likelihoods.log_probabilities[dst] = likelihoods.log_probabilities[src];
		likelihoods.probabilities[dst] = likelihoods.probabilities[src];
	}

	static inline void move(const table_distribution<V>& src, table_distribution<V>& dst) {
		core::move(src.likelihoods, dst.likelihoods);
		dst.probabilities = src.probabilities;
		dst.sum = src.sum;
	}

	static inline void swap(table_distribution<V>& first, table_distribution<V>& second) {
		core::swap(first.likelihoods, second.likelihoods);
		core::swap(first.probabilities, second.probabilities);
		core::swap(first.sum, second.sum);
	}

	static inline long unsigned int size_of(const table_distribution<V>& distribution, unsigned int table_count) {
		return core::size_of(distribution.likelihoods, table_count)
				+ core::size_of(distribution.sum) + sizeof(V) * table_count;
	}

	static inline void free(table_distribution<V>& distribution) {
		core::free(distribution.probabilities);
		core::free(distribution.likelihoods);
	}

private:
	template<typename BaseDistribution, typename DataDistribution, typename K>
	inline void renormalize(
		const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h)
	{
		likelihoods.renormalize(h.table_count);

		sum = 0.0;
		for (unsigned int i = 0; i < h.table_count; i++) {
			probabilities[i] = h.table_sizes[i] * likelihoods.probabilities[i];
			sum += probabilities[i];
		}
	}
};

template<typename V>
inline bool init(table_distribution<V>& distribution, unsigned int table_count) {
	distribution.probabilities = (V*) malloc(sizeof(V) * table_count);
	if (distribution.probabilities == NULL) {
		fprintf(stderr, "init ERROR: Unable to initialize probability array in table_distribution.\n");
		return false;
	} else if (!init(distribution.likelihoods, table_count)) {
		fprintf(stderr, "init ERROR: Unable to initialize array_categorical structure in table_distribution.\n");
		core::free(distribution.probabilities);
		return false;
	}
	return true;
}

/**
 * A specialization of ::cache where the `BaseDistribution` satisfies
 * is_dirichlet and the `DataDistribution` is dense_categorical.
 */
template<typename BaseDistribution, typename V>
struct cache<BaseDistribution, dense_categorical<V>, unsigned int, V,
	typename std::enable_if<is_dirichlet<BaseDistribution>::value>::type>
{
	typedef hdp_sampler<BaseDistribution, dense_categorical<V>, unsigned int, V> sampler_root;

	struct root_distribution {
		table_distribution<V>& distribution;
		const V* log_rising_factorials;

		root_distribution(table_distribution<V>& distribution, const V* log_rising_factorials)
			: distribution(distribution), log_rising_factorials(log_rising_factorials) { }

		inline V max() const {
			return distribution.likelihoods.maximum;
		}

		inline V sum() const {
			return distribution.sum;
		}

		inline V likelihood(unsigned int index) const {
			return distribution.likelihoods.probabilities[index];
		}

		inline const V* probabilities() const {
			return distribution.probabilities;
		}

		inline void free() { }
	};

	/* a map from sizes of observation sets to table_distributions */
	array_map<unsigned int, table_distribution<V>> distributions;

	/* histogram of table sizes over *all tables* in the CRF */
	array_histogram<unsigned int> table_histogram;

	/* a map from sizes of observation sets to histogram where the
	   keys are root table identifiers and the values are the number
	   of those observations in the descendants of that table */
	hash_map<unsigned int, array_histogram<unsigned int>> table_counts;

	rising_factorial_cache<V> factorial_cache;

	/**
	 * Constructs a cache structure for the HDP sampler hierarchy rooted at `h`.
	 */
	cache(const sampler_root& h) :
			distributions(16), table_histogram(16), table_counts(128), factorial_cache(h.pi().sum()) { }
	~cache() { free(); }

	bool get_root_probabilities(
		const sampler_root& h, table_distribution<V>& distribution,
		unsigned int observation_count, unsigned int largest_table_size)
	{
		if (!init(distribution, h.table_count)) {
			fprintf(stderr, "cache.get_root_probabilities"
					" ERROR: Unable to initialize new table_distribution.\n");
			return false;
		}

		distribution.sum = 0.0;
		const V* log_rising_factorials = factorial_cache.get_rising_factorials(observation_count, largest_table_size);
		for (unsigned int i = 0; i < h.table_count; i++)
			distribution.likelihoods.place(i, -log_rising_factorials[h.descendant_observations[i].sum]);
		distribution.likelihoods.renormalize(h.table_count);
		for (unsigned int i = 0; i < h.table_count; i++) {
			distribution.probabilities[i] = distribution.likelihoods.probabilities[i] * h.table_sizes[i];
			distribution.sum += distribution.probabilities[i];
		}
		return true;
	}

	table_distribution<V>& get_root_probabilities(
		const sampler_root& h,
		unsigned int observation_count)
	{
		if (!distributions.ensure_capacity((unsigned int) distributions.size + 1)) {
			fprintf(stderr, "cache.get_root_probabilities ERROR: Unable to expand array map.\n");
			exit(EXIT_FAILURE);
		}

		bool contains;
		table_distribution<V>& distribution = distributions.get(observation_count, contains);
		if (contains)
			return distribution;
		get_root_probabilities(h, distribution, observation_count, largest_table());
		distributions.keys[distributions.size] = observation_count;
		distributions.size++;
		return distribution;
	}

	inline void increase_descendant_count(unsigned int old_count, unsigned int new_count) {
		unsigned int index = 0;
		if (old_count > 0)
			index = table_histogram.subtract(old_count) + 1;
		table_histogram.add(new_count, 1, index);
	}

	inline void decrease_descendant_count(unsigned int old_count, unsigned int new_count)
	{
		unsigned int index;
		if (new_count > 0)
			index = table_histogram.add(new_count);
		else index = 0;
		table_histogram.subtract(old_count, index);
	}

	template<bool Check>
	inline void increase_descendant_count(
		const sampler_root& h, unsigned int table,
		unsigned int old_count, unsigned int new_count)
	{
		for (unsigned int i = 0; i < distributions.size; i++)
			distributions.values[i].template change_descendant_count<Check>(h, table, new_count, distributions.keys[i]);
		log_cache<V>::instance().ensure_size(new_count);
		increase_descendant_count(old_count, new_count);
	}

	template<bool Check>
	inline void decrease_descendant_count(
		const sampler_root& h, unsigned int table,
		unsigned int old_count, unsigned int new_count)
	{
		for (unsigned int i = 0; i < distributions.size; i++)
			distributions.values[i].template change_descendant_count<Check>(h, table, new_count, distributions.keys[i]);
		decrease_descendant_count(old_count, new_count);
	}

	inline bool add_new_table(unsigned int count) {
		if (count > 0)
			return table_histogram.add(count);
		else return true;
	}

	inline bool add_new_table(const sampler_root& h, unsigned int count)
	{
		for (unsigned int i = 0; i < distributions.size; i++)
			if (!distributions.values[i].add_new_table(h, count, distributions.keys[i]))
				return false;
		log_cache<V>::instance().ensure_size(count);
		return add_new_table(count);
	}

	bool is_valid(const sampler_root& h) const {
		for (unsigned int i = 0; i < distributions.size; i++) {
			if (!distributions.values[i].is_valid(h, distributions.keys[i])) {
				fprintf(stderr, "cache.is_valid ERROR: Cache check"
						" failed for observation count %u.\n", distributions.keys[i]);
				return false;
			}
		}
		if (table_histogram.counts.contains(0)) {
			fprintf(stderr, "cache.is_valid ERROR: Table histogram contains the key '1'.");
			return false;
		}

		array_histogram<unsigned int> actual_table_counts =
			array_histogram<unsigned int>((unsigned int) table_histogram.counts.size);
		compute_table_histogram(actual_table_counts, h);
		if (actual_table_counts != table_histogram) {
			fprintf(stderr, "cache.is_valid ERROR: Table histogram is incorrect.\n");
			return false;
		}

		if (!factorial_cache.is_valid()) {
			fprintf(stderr, "cache.is_valid ERROR: Rising factorial cache is incorrect.\n");
			return false;
		}
		return true;
	}

	static bool compute_table_histogram(
		array_histogram<unsigned int>& histogram,
		const node_sampler<unsigned int, V>& n)
	{
		for (unsigned int i = 0; i < n.child_count(); i++)
			if (!compute_table_histogram(histogram, n.children[i]))
				return false;
		for (unsigned int i = 0; i < n.table_count; i++)
			if (n.descendant_observations[i].sum > 0)
				histogram.add(n.descendant_observations[i].sum);
		return histogram.add(1, n.observation_count());
	}

	static bool compute_table_histogram(
		array_histogram<unsigned int>& histogram,
		const sampler_root& h)
	{
		for (unsigned int i = 0; i < h.child_count(); i++)
			if (!compute_table_histogram(histogram, h.children[i]))
				return false;
		for (unsigned int i = 0; i < h.table_count; i++)
			if (h.descendant_observations[i].sum > 0)
				histogram.add(h.descendant_observations[i].sum);
		histogram.add(1, h.observation_count());
		return true;
	}

	inline bool compute_table_histogram(const sampler_root& h) {
		table_histogram.clear();
		if (!compute_table_histogram(table_histogram, h))
			return false;
		log_cache<V>::instance().ensure_size(largest_table());
		return true;
	}

	inline unsigned int largest_table() const {
		return table_histogram.counts.keys[table_histogram.counts.size - 1];
	}

	void clear_cache() {
		for (unsigned int i = 0; i < table_histogram.counts.size; i++) {
			if (table_histogram.counts.values[i] == 0) {
				unsigned int index = distributions.index_of(table_histogram.counts.keys[i]);
				if (index < distributions.size) {
					core::free(distributions.values[index]);
					distributions.remove_at(index);
				}
			}
		}
		table_histogram.remove_zeros();
	}

	bool add_to_cache(const sampler_root& h)
	{
		for (unsigned int i = 0; i < table_counts.table.capacity; i++)
			if (!is_empty(table_counts.table.keys[i]))
				core::free(table_counts.values[i]);
		table_counts.clear();

		for (unsigned int i = 0; i < h.table_count; i++) {
			const array_histogram<unsigned int>& observations = h.descendant_observations[i];
			for (unsigned int j = 0; j < observations.counts.size; j++) {
				unsigned int atom = observations.counts.keys[j];

				bool contains;
				unsigned int bucket;
				if (!table_counts.check_size())
					return false;
				array_histogram<unsigned int>& histogram = table_counts.get(atom, contains, bucket);
				if (!contains) {
					if (!init(histogram, 4)) return false;
					table_counts.table.keys[bucket] = atom;
					table_counts.table.size++;
				}
				if (!histogram.add_unsorted(i, observations.counts.values[j]))
					return false;
			}
		}

		for (unsigned int i = 0; i < table_counts.table.capacity; i++) {
			if (!is_empty(table_counts.table.keys[i])) {
				if (table_counts.values[i].counts.size > 1) {
					sort(table_counts.values[i].counts.keys,
						table_counts.values[i].counts.values,
						(unsigned int) table_counts.values[i].counts.size);
				}
			}
		}
		return true;
	}

	inline array_histogram<unsigned int>& get_table_counts(unsigned int atom, bool& contains) {
		return table_counts.get(atom, contains);
	}

	inline bool move_to_table(unsigned int atom,
			unsigned int src_table, unsigned int dst_table)
	{
		if (!move_to_table_helper(atom, 1, src_table, dst_table))
			return false;
		return true;
	}

	inline bool move_to_table(
			const array_histogram<unsigned int>& observations,
			unsigned int src_table, unsigned int dst_table)
	{
		for (unsigned int i = 0; i < observations.counts.size; i++)
			if (!move_to_table_helper(observations.counts.keys[i],
					observations.counts.values[i], src_table, dst_table))
				return false;
		return true;
	}

	inline bool add_to_table(unsigned int atom, unsigned int table)
	{
		if (!add_to_table_helper(atom, 1, table))
			return false;
		return true;
	}

	inline bool move_to_table(
			const array_histogram<unsigned int>& observations, unsigned int table)
	{
		for (unsigned int i = 0; i < observations.counts.size; i++)
			if (!add_to_table_helper(observations.counts.keys[i],
					observations.counts.values[i], table))
				return false;
		return true;
	}

	inline bool remove_from_table(unsigned int atom, unsigned int table)
	{
		if (!remove_from_table_helper(atom, 1, table))
			return false;
		return true;
	}

	inline bool remove_from_table(
		const array_histogram<unsigned int>& observations, unsigned int table)
	{
		for (unsigned int i = 0; i < observations.counts.size; i++)
			if (!remove_from_table_helper(observations.counts.keys[i], observations.counts.values[i], table))
				return false;
		return true;
	}

	inline void remove_zeros() {
		for (unsigned int i = 0; i < table_counts.table.capacity; i++)
			if (!is_empty(table_counts.table.keys[i]))
				table_counts.values[i].remove_zeros();
	}

	inline void relabel_tables(const unsigned int* table_map) {
		remove_zeros();

		for (unsigned int i = 0; i < table_counts.table.capacity; i++) {
			if (is_empty(table_counts.table.keys[i]))
				continue;
			array_map<unsigned int, unsigned int>& map = table_counts.values[i].counts;
			for (unsigned int j = 0; j < map.size; j++)
				map.keys[j] = table_map[map.keys[j]];
			if (map.size > 1)
				sort(map.keys, map.values, (unsigned int) map.size);
		}
	}

	inline void on_change_table_size(unsigned int table, unsigned int new_table_size)
	{
		for (unsigned int i = 0; i < distributions.size; i++)
			distributions.values[i].change_table_size(table, new_table_size);
	}

	inline void on_finished_sampling_hdp() {
		remove_zeros();
		clear_cache();
	}

	inline void prepare_sampler(const sampler_root& h) {
		compute_table_histogram(h);
		get_root_probabilities(h, 1); /* precompute the cache for singleton observations */
		add_to_cache(h);
	}

	inline void on_move_table(unsigned int src, unsigned int dst) {
		for (unsigned int i = 0; i < distributions.size; i++)
			distributions.values[i].move_table(src, dst);
	}

	template<typename Observations>
	inline void on_move_to_table(
		const node_sampler<unsigned int, V>& n,
		unsigned int src, unsigned int dst,
		const Observations& observations)
	{
		decrease_descendant_count(n.descendant_observations[src].sum + count(observations), n.descendant_observations[src].sum);
		increase_descendant_count(n.descendant_observations[dst].sum - count(observations), n.descendant_observations[dst].sum);
	}

	template<typename Observations>
	inline bool on_move_to_table(
		const sampler_root& h,
		unsigned int src, unsigned int dst,
		const Observations& observations)
	{
		decrease_descendant_count<false>(h, src,
				h.descendant_observations[src].sum + count(observations), h.descendant_observations[src].sum);
		increase_descendant_count<true>(h, dst,
				h.descendant_observations[dst].sum - count(observations), h.descendant_observations[dst].sum);
		return move_to_table(observations, src, dst);
	}

	template<typename Observations>
	inline void on_move_to_new_table(
		const node_sampler<unsigned int, V>& n,
		unsigned int src, const Observations& observations)
	{
		decrease_descendant_count(n.descendant_observations[src].sum
				+ count(observations), n.descendant_observations[src].sum);
		add_new_table(n.descendant_observations[n.table_count - 1].sum);
	}

	template<typename Observations>
	inline bool on_move_to_new_table(
		const sampler_root& h,
		unsigned int src, const Observations& observations)
	{
		decrease_descendant_count<false>(h, src,
				h.descendant_observations[src].sum + count(observations), h.descendant_observations[src].sum);
		add_new_table(h, h.descendant_observations[h.table_count - 1].sum);
		return move_to_table(observations, src, h.table_count - 1);
	}

	template<typename Observations>
	inline void on_add_to_table(
		const node_sampler<unsigned int, V>& n,
		unsigned int table, const Observations& observations)
	{
		increase_descendant_count(n.descendant_observations[table].sum - count(observations), n.descendant_observations[table].sum);
	}

	template<typename Observations>
	inline bool on_add_to_table(
		const sampler_root& h,
		unsigned int table,
		const Observations& observations)
	{
		increase_descendant_count<true>(h, table,
				h.descendant_observations[table].sum - count(observations), h.descendant_observations[table].sum);
		return add_to_table(observations, table);
	}

	template<typename Observations>
	inline void on_add_to_new_table(
		const node_sampler<unsigned int, V>& n,
		const Observations& observations)
	{
		add_new_table(n.descendant_observations[n.table_count - 1].sum);
	}

	template<typename Observations>
	inline bool on_add_to_new_table(
		const sampler_root& h,
		const Observations& observations)
	{
		add_new_table(h, h.descendant_observations[h.table_count - 1].sum);
		return add_to_table(observations, h.table_count - 1);
	}

	template<typename Observations>
	inline void on_remove_from_table(
		const node_sampler<unsigned int, V>& n,
		unsigned int table, const Observations& observations)
	{
		decrease_descendant_count(n.descendant_observations[table].sum + count(observations), n.descendant_observations[table].sum);
	}

	template<typename Observations>
	inline bool on_remove_from_table(
		const sampler_root& h,
		unsigned int table,
		const Observations& observations)
	{
		decrease_descendant_count<false>(h, table,
				h.descendant_observations[table].sum + count(observations), h.descendant_observations[table].sum);
		return remove_from_table(observations, table);
	}

	template<typename Observations>
	root_distribution compute_root_distribution(
		const sampler_root& root,
		const Observations& observations)
	{
		table_distribution<V>& distribution = get_root_probabilities(root, count(observations));
		const V* log_rising_factorials =
				factorial_cache.get_rising_factorials(count(observations), largest_table());

		compute_likelihoods(root, distribution, root.table_count, observations);
		return root_distribution(distribution, log_rising_factorials);
	}

	template<typename Observations>
	root_distribution compute_root_distribution(
		const sampler_root& root,
		const Observations& observations,
		unsigned int old_root_assignment)
	{
		table_distribution<V>& distribution = get_root_probabilities(root, count(observations));
		const V* log_rising_factorials =
				factorial_cache.get_rising_factorials(count(observations), largest_table());
		V new_likelihood = -log_rising_factorials[root.descendant_observations[old_root_assignment].sum - count(observations)];
		distribution.set_higher(root, old_root_assignment, new_likelihood);

		compute_likelihoods(root, distribution, old_root_assignment, observations);
		return root_distribution(distribution, log_rising_factorials);
	}

	void compute_likelihoods(const sampler_root& h,
		table_distribution<V>& root_distribution,
		unsigned int old_root_assignment,
		unsigned int observation, unsigned int count = 1)
	{
		bool contains;
		array_histogram<unsigned int>& table_counts = get_table_counts(observation, contains);
		if (!contains) return;
		for (unsigned int i = 0; i < table_counts.counts.size; i++) {
			unsigned int table = table_counts.counts.keys[i];

			V base = h.pi().get_for_atom(observation) + table_counts.counts.values[i];
			if (table == old_root_assignment)
				base -= count;

			V contribution = log_rising_factorial(base, count);
			if (contribution > 0) {
				root_distribution.set_higher(
					h, table, root_distribution.likelihoods.log_probabilities[table] + contribution);
			} else {
				root_distribution.set_lower(
					h, table, root_distribution.likelihoods.log_probabilities[table] + contribution);
			}
		}
	}

	inline void compute_likelihoods(const sampler_root& h,
		table_distribution<V>& root_distribution,
		unsigned int old_root_assignment,
		const array_histogram<unsigned int>& observations)
	{
		for (unsigned int i = 0; i < observations.counts.size; i++)
			compute_likelihoods(h, root_distribution, old_root_assignment,
					observations.counts.keys[i], observations.counts.values[i]);
	}

	void on_sample_table(sampler_root& h,
		root_distribution& root_probabilities,
		unsigned int old_root_assignment,
		unsigned int observation, unsigned int count = 1)
	{
		bool contains;
		array_histogram<unsigned int>& table_counts = get_table_counts(observation, contains);
		if (!contains) return;
		for (unsigned int i = 0; i < table_counts.counts.size; i++) {
			unsigned int table = table_counts.counts.keys[i];

			V base = h.pi().get_for_atom(observation) + table_counts.counts.values[i];
			if (table == old_root_assignment)
				base -= count;

			V contribution = log_rising_factorial(base, count);
			if (contribution > 0) {
				root_probabilities.distribution.set_lower(
					h, table, root_probabilities.distribution.likelihoods.log_probabilities[table] - contribution);
			} else {
				root_probabilities.distribution.set_higher(
					h, table, root_probabilities.distribution.likelihoods.log_probabilities[table] - contribution);
			}
		}
	}

	inline void on_sample_table(sampler_root& h,
		root_distribution& root_probabilities,
		unsigned int old_root_assignment,
		const array_histogram<unsigned int>& observations)
	{
		for (unsigned int i = 0; i < observations.counts.size; i++)
			on_sample_table(h, root_probabilities, old_root_assignment,
					observations.counts.keys[i], observations.counts.values[i]);
	}

	template<typename Observations>
	inline void on_add(const Observations& observations)
	{
		table_histogram.add(count(observations));
	}

	template<typename Observations>
	inline void on_remove(const Observations& observations)
	{
		table_histogram.subtract(count(observations));
	}

	inline void on_finish_sampling(sampler_root& root,
		root_distribution& root_probabilities,
		unsigned int old_root_assignment)
	{
		V new_likelihood = -root_probabilities.log_rising_factorials[root.descendant_observations[old_root_assignment].sum];
		root_probabilities.distribution.set_lower(root, old_root_assignment, new_likelihood);
	}

	static inline long unsigned int size_of(
		const cache<BaseDistribution, dense_categorical<V>, unsigned int, V>& cache,
		unsigned int table_count)
	{
		return core::size_of(cache.distributions, make_key_value_metric(default_metric(), table_count))
			 + core::size_of(cache.table_counts)
			 + core::size_of(cache.factorial_cache)
			 + core::size_of(cache.table_histogram);
	}

	/**
	 * Computes a matrix, where every row corresponds to a
	 * sample from the posterior, and every column is a table
	 * at the root node. Each element contains the probability
	 * of assigning the given observation to each table at the
	 * root (without taking into account table sizes and the
	 * alpha parameter).
	 */
	inline V** compute_root_probabilities(const sampler_root& h, const unsigned int& observation) const {
		return ::compute_root_probabilities(h, observation);
	}

	/**
	 * Constructs a map from observations to a vector, of which
	 * each element is the probability of assigning that
	 * observation to each table at the root (without taking
	 * into account table sizes and the alpha parameter).
	 */
	inline V** compute_root_probabilities(sampler_root& h,
		const unsigned int* observations, unsigned int observation_count) const
	{
		return ::compute_root_probabilities(h, observations, observation_count);
	}

private:
	inline bool add_to_new_table(array_map<unsigned int, unsigned int>& map,
			unsigned int src_index, unsigned int dst_table, unsigned int count)
	{
		if (map.values[src_index] == 0) {
			map.keys[src_index] = dst_table;
			map.values[src_index] = count;
		} else {
			if (!map.ensure_capacity((unsigned int) map.size + 1))
				return false;
			map.keys[map.size] = dst_table;
			map.values[map.size] = count;
			map.size++;
		}
		sort(map.keys, map.values, (unsigned int) map.size);
		return true;
	}

	inline bool move_to_table_helper(unsigned int atom, unsigned int count,
			unsigned int src_table, unsigned int dst_table)
	{
#if !defined(NDEBUG)
		bool contains;
		array_histogram<unsigned int>& histogram = table_counts.get(atom, contains);
		if (!contains) {
			fprintf(stderr, "cache WARNING: There are no observations of the given atom.\n");
			return false;
		}
		array_map<unsigned int, unsigned int>& map = histogram.counts;
#else
		array_map<unsigned int, unsigned int>& map = table_counts.get(atom).counts;
#endif

		unsigned int src_index, dst_index;
		if (src_table < dst_table) {
			src_index = map.index_of(src_table);
			dst_index = map.index_of(dst_table, src_index + 1);

			/* the destination table may not exist in the array_map */
			if (dst_index == map.size) {
				map.values[src_index] -= count;
				return add_to_new_table(map, src_index, dst_table, count);
			}
		} else {
			dst_index = map.index_of(dst_table);

			/* the destination table may not exist in the array_map */
			if (dst_index == map.size) {
				src_index = map.index_of(src_table);
				map.values[src_index] -= count;
				return add_to_new_table(map, src_index, dst_table, count);
			}

			src_index = map.index_of(src_table, dst_index + 1);
		}

		map.values[src_index] -= count;
		map.values[dst_index] += count;
		return true;
	}

	inline bool add_to_table_helper(unsigned int atom, unsigned int count, unsigned int table)
	{
#if !defined(NDEBUG)
		bool contains;
		array_histogram<unsigned int>& histogram = table_counts.get(atom, contains);
		if (!contains) {
			fprintf(stderr, "cache WARNING: There are no observations of the given atom.\n");
			return false;
		}
		array_map<unsigned int, unsigned int>& map = histogram.counts;
#else
		array_map<unsigned int, unsigned int>& map = table_counts.get(atom).counts;
#endif

		map.values[map.index_of(table)] += count;
		return true;
	}

	inline bool remove_from_table_helper(unsigned int atom, unsigned int count, unsigned int table)
	{
#if !defined(NDEBUG)
		bool contains;
		array_histogram<unsigned int>& histogram = table_counts.get(atom, contains);
		if (!contains) {
			fprintf(stderr, "cache WARNING: There are no observations of the given atom.\n");
			return false;
		}
		array_map<unsigned int, unsigned int>& map = histogram.counts;
#else
		array_map<unsigned int, unsigned int>& map = table_counts.get(atom).counts;
#endif

		map.values[map.index_of(table)] -= count;
		return true;
	}

	static inline void free(cache<BaseDistribution, dense_categorical<V>, unsigned int, V>& c) {
		c.~cache();
	}

	inline void free() {
		for (unsigned int i = 0; i < distributions.size; i++)
			core::free(distributions.values[i]);
		for (unsigned int i = 0; i < table_counts.table.capacity; i++)
			if (!is_empty(table_counts.table.keys[i]))
				core::free(table_counts.values[i]);
	}
};


/**
 * A specialization of ::cache where the `DataDistribution` is a constant
 * (degenerate) distribution. There is no restriction on `BaseDistribution`.
 */
template<typename BaseDistribution, typename K, typename V>
struct cache<BaseDistribution, constant<K>, K, V>
{
	typedef hdp_sampler<BaseDistribution, constant<K>, K, V> sampler_root;

	struct root_distribution {
		unsigned int* distribution;
		unsigned int maximum;
		unsigned int total;

		unsigned int max() const { return maximum; }
		unsigned int sum() const { return total; }
		unsigned int* probabilities() const { return distribution; }

		bool likelihood(unsigned int index) const {
			return distribution[index] != 0;
		}

		inline void free() {
			core::free(distribution);
		}
	};

	/**
	 * Constructs a cache structure for the HDP sampler hierarchy rooted at `h`.
	 */
	cache(const sampler_root& h) { }

	const K& first(const K& item) const {
		return item;
	}

	const K& first(const array_histogram<K>& items) const {
		return items.counts.keys[0];
	}

	template<typename Observations>
	root_distribution compute_root_distribution(
		const sampler_root& h,
		const Observations& observations)
	{
		unsigned int* distribution = (unsigned int*) calloc(h.table_count, sizeof(unsigned int));
		if (distribution == NULL) {
			fprintf(stderr, "cache.compute_root_distribution ERROR: Out of memory.\n");
			exit(EXIT_FAILURE);
		}

		for (unsigned int i = 0; i < h.table_count; i++) {
			if (h.descendant_observations[i].counts.size > 0
			 && constant<K>::conditional(h.pi(), first(observations), h.descendant_observations[i]))
				distribution[i] = h.table_sizes[i];
		}

		unsigned int sum = 0;
		unsigned int maximum = 0;
		for (unsigned int i = 0; i < h.table_count; i++) {
			sum += distribution[i];
			if (distribution[i] > maximum)
				maximum = distribution[i];
		}

		return {distribution, maximum, sum};
	}

	template<typename Observations>
	inline root_distribution compute_root_distribution(
		const sampler_root& h,
		const Observations& observations,
		unsigned int old_root_assignment)
	{
		return compute_root_distribution(h, observations);
	}

	inline void on_change_table_size(unsigned int table, unsigned int new_table_size) { }
	inline void on_finished_sampling_hdp() { }
	inline void prepare_sampler(const sampler_root& h) { }
	inline void on_move_table(unsigned int src, unsigned int dst) { }

	template<typename NodeType, typename Observations>
	inline bool on_move_to_table(
		const NodeType& n,
		unsigned int src, unsigned int dst,
		const Observations& observations)
	{ return true; }

	template<typename NodeType, typename Observations>
	inline bool on_move_to_new_table(
		const NodeType& n,
		unsigned int src, const Observations& observations)
	{ return true; }

	template<typename NodeType, typename Observations>
	inline bool on_add_to_table(
		const NodeType& n,
		unsigned int table,
		const Observations& observations)
	{ return true; }

	template<typename NodeType, typename Observations>
	inline bool on_add_to_new_table(
		const NodeType& n,
		const Observations& observations)
	{ return true; }

	template<typename NodeType, typename Observations>
	inline bool on_remove_from_table(
		const NodeType& n,
		unsigned int table,
		const Observations& observations)
	{ return true; }

	template<typename Observations>
	inline void on_add(const Observations& observations) { }

	template<typename Observations>
	inline void on_remove(const Observations& observations) { }

	void on_sample_table(sampler_root& h,
		root_distribution& root_probabilities,
		unsigned int old_root_assignment,
		const K& observation, unsigned int count = 1) { }

	inline void on_sample_table(sampler_root& h,
		root_distribution& root_probabilities,
		unsigned int old_root_assignment,
		const array_histogram<K>& observations) { }

	inline void on_finish_sampling(sampler_root& root,
		root_distribution& root_probabilities,
		unsigned int old_root_assignment) { }

	inline void relabel_tables(const unsigned int* table_map) { }

	inline bool is_valid(const sampler_root& root) const { return true; }

	static inline void free(cache<BaseDistribution, constant<K>, K, V>& c) { c.~cache(); }

	/**
	 * Computes a matrix, where every row corresponds to a
	 * sample from the posterior, and every column is a table
	 * at the root node. Each element contains the probability
	 * of assigning the given observation to each table at the
	 * root (without taking into account table sizes and the
	 * alpha parameter).
	 */
	array<unsigned int>* compute_root_probabilities(const sampler_root& h, const K& observation)
	{
		/* initialize the root_probabilities array of arrays */
		array<unsigned int>* root_probabilities = (array<unsigned int>*)
				malloc(h.posterior.length * sizeof(array<unsigned int>));
		if (root_probabilities == NULL) {
			fprintf(stderr, "cache.compute_root_probabilities ERROR: Insufficient memory for matrix.\n");
			return NULL;
		} for (unsigned int i = 0; i < h.posterior.length; i++) {
			if (!array_init(root_probabilities[i], 8)) {
				cleanup_root_probabilities(root_probabilities, i);
				fprintf(stderr, "cache.compute_root_probabilities ERROR: Insufficient memory for matrix row.\n");
				return NULL;
			}
		}

		/* store the appropriate conditional probabilities in root_probabilities */
		for (unsigned int i = 0; i < h.posterior.length; i++) {
			for (unsigned int j = 0; j < h.posterior[i].table_count; j++) {
				if (h.posterior[i].descendant_observations[j].counts.size == 0
				 || h.posterior[i].descendant_observations[j].counts.keys[0] != observation)
					continue;
				if (!root_probabilities[i].add(j)) {
					cleanup_root_probabilities(root_probabilities, h.posterior.length);
					return NULL;
				}
			}
		}

		return root_probabilities;
	}

	/**
	 * Constructs a map from observations to a vector, of which
	 * each element is the probability of assigning that
	 * observation to each table at the root (without taking
	 * into account table sizes and the alpha parameter).
	 *
	 * This optimization exploits the fact that every table must
	 * have at most one distinct observation. In addition,
	 * probabilities in each vector are either 0 or 1. As a
	 * result, the matrix is zero except for the diagonal, and
	 * so we only return the diagonal.
	 */
	array<unsigned int>* compute_root_probabilities(
			const sampler_root& h, const K* observations, unsigned int observation_count)
	{
		array<unsigned int>* probabilities = (array<unsigned int>*)
				malloc(observation_count * sizeof(array<unsigned int>));
		if (probabilities == NULL) {
			fprintf(stderr, "cache.compute_root_probabilities ERROR: Insufficient memory for matrix.\n");
			return NULL;
		} for (unsigned int i = 0; i < observation_count; i++) {
			if (!array_init(probabilities[i], 8)) {
				cleanup_root_probabilities(probabilities, i);
				fprintf(stderr, "cache.compute_root_probabilities ERROR: Insufficient memory for matrix row.\n");
				return NULL;
			}
		}

		K* table_observations = (K*) malloc(sizeof(K) * h.table_count);
		if (table_observations == NULL) {
			fprintf(stderr, "cache.compute_root_probabilities ERROR: Insufficient memory for table observation list.\n");
			cleanup_root_probabilities(probabilities, observation_count); return NULL;
		}

		unsigned int* table_indices = (unsigned int*) malloc(sizeof(unsigned int) * h.table_count);
		if (table_indices == NULL) {
			fprintf(stderr, "cache.compute_root_probabilities ERROR: Insufficient memory for table indices.\n");
			cleanup_root_probabilities(probabilities, observation_count);
			core::free(table_observations); return NULL;
		}

		for (unsigned int j = 0; j < h.table_count; j++) {
			table_indices[j] = j;
			if (h.descendant_observations[j].counts.size == 0)
				set_empty(table_observations[j]);
			else table_observations[j] = h.descendant_observations[j].counts.keys[0];
		}
		if (h.table_count > 1)
			sort(table_observations, table_indices, h.table_count, default_sorter());

		auto intersect = [&](unsigned int first_index, unsigned int second_index) {
				if (!probabilities[first_index].add(table_indices[second_index])) return false;
				return true;
			};

		unsigned int i = 0, j = 0;
		while (i < observation_count && j < h.table_count)
		{
			if (observations[i] == table_observations[j]) {
				intersect(i, j);
				j++;
			} else if (observations[i] < table_observations[j]) {
				i++;
			} else {
				j++;
			}
		}

		for (unsigned int j = 0; j < h.table_count; j++)
			if (!is_empty(table_observations[j])) core::free(table_observations[j]);
		core::free(table_observations);
		core::free(table_indices);
		return probabilities;
	}
};

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
inline bool init(
	cache<BaseDistribution, DataDistribution, K, V>& hdp_cache,
	const hdp_sampler<BaseDistribution, DataDistribution, K, V>& sampler)
{
	return new (&hdp_cache) cache<BaseDistribution, DataDistribution, K, V>(sampler) != NULL;
}

#endif /* CACHE_H_ */
