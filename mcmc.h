/**
 * mcmc.h - Markov chain Monte Carlo inference
 *		for hierarchical Dirichlet processes.
 *
 *  Created on: Jul 26, 2015
 *      Author: asaparov
 */

#ifndef MCMC_H_
#define MCMC_H_

#include <core/io.h>
#include <math/sparse_vector.h>
#include <math/distributions.h>
#include <math/log.h>

#include <algorithm>

#include "hdp.h"
#include "cache.h"

/* forward declarations */

template<typename DataDistribution, typename NodeType, typename BaseDistribution, typename Observations>
inline bool sample_initial_assignment(
        NodeType& n,
        const BaseDistribution& pi,
        const Observations& observations,
        unsigned int& assignment);

template<bool Collapsed, typename NodeType, typename Observations, typename Cache>
bool select_table(NodeType& n,
        const Observations& observations,
        unsigned int old_table,
        unsigned int selected_table,
        unsigned int& new_table,
        Cache& cache);


inline double log_rising_factorial(double base, unsigned int exponent) {
	return lgamma(base + exponent) - lgamma(base);
}

static inline bool compare_histograms(
		const unsigned int* first, const unsigned int* second,
		unsigned int first_length, unsigned int second_length)
{
	if (first_length < second_length) {
		for (unsigned int i = 0; i < first_length; i++)
			if (first[i] != second[i])
				return false;
		for (unsigned int i = first_length; i < second_length; i++)
			if (second[i] != 0)
				return false;
	} else {
		for (unsigned int i = 0; i < second_length; i++)
			if (first[i] != second[i])
				return false;
		for (unsigned int i = second_length; i < first_length; i++)
			if (first[i] != 0)
				return false;
	}
	return true;
}

static inline void add_to_histogram(unsigned int* histogram,
		unsigned int histogram_length, unsigned int item)
{
#if !defined(NDEBUG)
	if (item >= histogram_length)
		fprintf(stderr, "add_to_histogram WARNING: Index out of bounds.\n");
#endif
	histogram[item]++;
}

template<typename K>
constexpr unsigned int count(const K& observation) {
	return 1;
}

template<typename K>
inline unsigned int count(const array_histogram<K>& observations) {
	return observations.sum;
}


template<typename V>
struct node_sample {
	unsigned int* table_sizes;
	unsigned int* root_assignments;
	unsigned int customer_count;
	unsigned int table_count;

	unsigned int* table_assignments;

	node_sample(unsigned int table_count, unsigned int observation_count) {
		if (!initialize(table_count, observation_count)) {
			fprintf(stderr, "node_sample ERROR: Error during initialization.\n");
			exit(EXIT_FAILURE);
		}
	}

	inline unsigned int root_assignment(unsigned int i) const {
		return root_assignments[i];
	}

	template<typename Metric>
	static inline long unsigned int size_of(const node_sample<V>& sample, const Metric& metric) {
		return 2 * sample.table_count * sizeof(unsigned int)
				+ sample.customer_count * sizeof(unsigned int)
				+ core::size_of(sample.table_count) + core::size_of(sample.customer_count);
	}

	static inline void free(node_sample<V>& sample) {
		sample.free();
	}

private:
	inline bool initialize(unsigned int num_tables, unsigned int observation_count) {
		table_sizes = (unsigned int*) malloc(sizeof(unsigned int) * num_tables);
		if (table_sizes == NULL) {
			fprintf(stderr, "node_sample.initialize ERROR: Unable to initialize table_sizes.\n");
			return false;
		}

		if (observation_count == 0)
			observation_count = 1;
		table_assignments = (unsigned int*) malloc(sizeof(unsigned int) * observation_count);
		if (table_assignments == NULL) {
			fprintf(stderr, "node_sample.initialize ERROR: Unable to initialize table_assignments.\n");
			core::free(table_sizes);
			return false;
		}

		table_count = num_tables;
		root_assignments = (unsigned int*) malloc(sizeof(unsigned int) * num_tables);
		if (root_assignments == NULL) {
			fprintf(stderr, "node_sample.initialize ERROR: Unable to initialize root_assignments.\n");
			core::free(table_sizes);
			core::free(table_assignments);
			return false;
		}
		return true;
	}

	inline void free() {
		core::free(table_sizes);
		core::free(table_assignments);
		core::free(root_assignments);
	}

	template<typename A>
	friend bool init(node_sample<A>&, unsigned int, unsigned int);
};

template<typename V>
bool init(node_sample<V>& sample, unsigned int table_count, unsigned int observation_count) {
	return sample.initialize(table_count, observation_count);
}

template<typename AtomScribe>
struct node_sample_scribe {
	AtomScribe& atom_scribe;
	unsigned int observation_count;
	unsigned int root_cluster_count;

	node_sample_scribe(AtomScribe& atom_scribe, unsigned int observation_count, unsigned int root_cluster_count) :
		atom_scribe(atom_scribe), observation_count(observation_count), root_cluster_count(root_cluster_count) { }
};

template<typename V, typename AtomReader>
bool read(node_sample<V>& sample, FILE* in, node_sample_scribe<AtomReader>& reader) {
	if (!read(sample.table_count, in)) return false;

	sample.table_sizes = (unsigned int*) malloc(sizeof(unsigned int) * sample.table_count);
	if (sample.table_sizes == NULL)
		return false;
	sample.root_assignments = (unsigned int*) malloc(sizeof(unsigned int) * sample.table_count);
	if (sample.root_assignments == NULL) {
		free(sample.table_sizes);
		return false;
	}
	sample.table_assignments = (unsigned int*) malloc(sizeof(unsigned int) * reader.observation_count);
	if (sample.table_assignments == NULL) {
		free(sample.root_assignments);
		free(sample.table_sizes);
		return false;
	}

	if (!read(sample.table_sizes, in, sample.table_count)) return false;
	if (!read(sample.root_assignments, in, sample.table_count)) return false;
	if (!read(sample.table_assignments, in, reader.observation_count)) return false;

	sample.customer_count = 0;
	for (unsigned int i = 0; i < sample.table_count; i++)
		sample.customer_count += sample.table_sizes[i];

	return true;
}

template<typename V, typename AtomWriter>
bool write(const node_sample<V>& sample, FILE* out, node_sample_scribe<AtomWriter>& writer) {
	if (!write(sample.table_count, out)) return false;
	if (!write(sample.table_sizes, out, sample.table_count)) return false;
	if (!write(sample.root_assignments, out, sample.table_count)) return false;
	if (!write(sample.table_assignments, out, writer.observation_count)) return false;
	return true;
}

template<typename K, typename V>
struct node_sampler
{
	typedef K atom_type;
	typedef V value_type;
	typedef node_sampler<K, V> child_type;
	typedef node<K, V> node_type;

	node_sampler<K, V>* children;
	node_sampler<K, V>* parent;
	node<K, V>* n;

	/**
	 * state variables for Gibbs sampling
	 */
	unsigned int* observation_assignments;

	/**
	 * In the Chinese restaurant representation, this is a
	 * histogram of table assignments of the samples from
	 * the CRP at this node.
	 *
	 * In the direct assignment representation, this is a
	 * histogram of root assignments of the samples in the
	 * CRP at this node.
	 */
	unsigned int* table_sizes;

	/* the tables in this CRP */
	unsigned int* table_assignments;
	unsigned int* root_assignments;
	array_histogram<K>* descendant_observations;
	unsigned int table_count;
	unsigned int table_capacity;
	unsigned int customer_count;

	/* array of posterior samples */
	array<node_sample<V>> posterior;

	template<typename BaseDistribution, typename DataDistribution>
	node_sampler(
		const hdp_sampler<BaseDistribution, DataDistribution, K, V>& root,
		const node<K, V>& n, const child_type* parent, unsigned int table_capacity) :
			n(&n), parent(parent), posterior(4)
	{
		if (!initialize(root, table_capacity))
			exit(EXIT_FAILURE);
	}

	~node_sampler() { free(); }

	inline V alpha() const {
		return n->get_alpha();
	}

	inline V log_alpha() const {
		return n->get_log_alpha();
	}

	inline unsigned int child_count() const {
		return (unsigned int) n->children.size;
	}

	inline const child_type& get_child(unsigned int key, bool& contains) const {
		unsigned int index = n->children.index_of(key);
		contains = (index < child_count());
		return children[index];
	}

	inline unsigned int child_key(unsigned int index) const {
		return n->children.keys[index];
	}

	inline unsigned int observation_count() const {
		return (unsigned int) n->observations.length;
	}

	inline const K& get_observation(unsigned int index) const {
		return n->observations[index];
	}

	inline unsigned int root_assignment(unsigned int i) const {
		return root_assignments[i];
	}

	inline unsigned int table_assignment(unsigned int i) const {
		return table_assignments[i];
	}

	template<typename Cache>
	inline void relabel_tables(const unsigned int* table_map, const Cache& cache) {
		/* apply the table map to the tables in the child nodes */
		for (unsigned int i = 0; i < child_count(); i++)
			children[i].relabel_parent_tables(table_map);
		for (unsigned int i = 0; i < observation_count(); i++)
			observation_assignments[i] = table_map[observation_assignments[i]];
	}

	void relabel_parent_tables(const unsigned int* parent_table_map) {
		for (unsigned int i = 0; i < table_count; i++)
			table_assignments[i] = parent_table_map[table_assignments[i]];
	}

	bool get_observations(array_histogram<K>& dst, unsigned int assignment) const {
		for (unsigned int i = 0; i < table_count; i++) {
			if (table_assignments[i] == assignment && !dst.add(descendant_observations[i])) {
				fprintf(stderr, "node_sampler.get_observations ERROR: Unable "
						"to add descendant observations to histogram.\n");
				return false;
			}
		}
		return true;
	}

	template<typename Cache>
	inline void move_table(unsigned int src, unsigned int dst, const Cache& cache) {
		table_assignments[dst] = table_assignments[src];
		table_sizes[dst] = table_sizes[src];
		root_assignments[dst] = root_assignments[src];
		array_histogram<K>::move(descendant_observations[src], descendant_observations[dst]);
	}

	template<typename Observations, typename Cache>
	inline bool move_to_table(const Observations& observations,
		unsigned int src, unsigned int dst, Cache& cache)
	{
		if (!descendant_observations[dst].add(observations)) {
			fprintf(stderr, "node_sampler.move_to_table ERROR: Unable to add observations to new table.\n");
			return false;
		}
		descendant_observations[src].subtract(observations);
		descendant_observations[src].remove_zeros();
		cache.on_move_to_table(*this, src, dst, observations);
		return true;
	}

	template<typename Observations, typename Cache>
	inline bool move_to_new_table(const Observations& observations, unsigned int src, Cache& cache)
	{
		if (!descendant_observations[table_count - 1].add(observations)) {
			fprintf(stderr, "node_sampler.move_to_new_table ERROR: Unable to add observations to new table.\n");
			return false;
		}
		descendant_observations[src].subtract(observations);
		descendant_observations[src].remove_zeros();
		cache.on_move_to_new_table(*this, src, observations);
		return true;
	}

	template<typename Observations, typename Cache>
	inline bool add_to_table(
		const Observations& observations,
		unsigned int table, Cache& cache)
	{
		if (!descendant_observations[table].add(observations)) {
			fprintf(stderr, "node_sampler.add_to_table ERROR: Unable to add observations to new table.\n");
			return false;
		}
		cache.on_add_to_table(*this, table, observations);
		return true;
	}

	template<typename Observations, typename Cache>
	inline bool add_to_new_table(const Observations& observations, Cache& cache)
	{
		if (!descendant_observations[table_count - 1].add(observations)) {
			fprintf(stderr, "node_sampler.add_to_new_table ERROR: Unable to add observations to new table.\n");
			return false;
		}
		cache.on_add_to_new_table(*this, observations);
		return true;
	}

	template<typename Observations, typename Cache>
	inline bool remove_from_table(
		const Observations& observations,
		unsigned int table, Cache& cache)
	{
		descendant_observations[table].subtract(observations);
		descendant_observations[table].remove_zeros();
		cache.on_remove_from_table(*this, table, observations);
		return true;
	}

	inline bool new_table() {
		unsigned int new_capacity = table_capacity;
		while (new_capacity < table_count + 2)
			new_capacity *= 2;
		if (table_capacity != new_capacity && !resize(new_capacity)) {
			fprintf(stderr, "node_sampler.new_table ERROR: Unable to expand table array.\n");
			return false;
		}
		table_capacity = new_capacity;

		if (!init(descendant_observations[table_count], 4)) {
			fprintf(stderr, "node_sampler.new_table ERROR: Unable to initialize new table.\n");
			return false;
		}
		table_count++;
		return true;
	}

	/* NOTE: this does not change the total number of tables */
	inline void free_table(unsigned int i) {
		core::free(descendant_observations[i]);
	}

	inline void set_table_count(unsigned int new_count) {
		table_count = new_count;
	}

	inline bool add_sample() {
		return add_sample(table_count);
	}

	bool add_sample(unsigned int root_cluster_count) {
		for (unsigned int i = 0; i < child_count(); i++)
			children[i].add_sample(root_cluster_count);

		if (!posterior.ensure_capacity(posterior.length + 1)) {
			fprintf(stderr, "node_sampler.add_sample ERROR: Unable to expand posterior sample array.\n");
			return false;
		} else if (!init(posterior[(unsigned int) posterior.length], table_count, observation_count())) {
			fprintf(stderr, "node_sampler.add_sample ERROR: Unable to initialize new sample.\n");
			return false;
		}

		node_sample<V>& sample = posterior[(unsigned int) posterior.length];
		for (unsigned int i = 0; i < observation_count(); i++)
			sample.table_assignments[i] = observation_assignments[i];
		for (unsigned int i = 0; i < table_count; i++) {
			sample.root_assignments[i] = root_assignments[i];
			sample.table_sizes[i] = table_sizes[i];
		}
		sample.customer_count = customer_count;
		posterior.length++;
		return true;
	}

	void clear_samples() {
		for (unsigned int i = 0; i < child_count(); i++)
			children[i].clear_samples();

		for (unsigned int i = 0; i < posterior.length; i++)
			core::free(posterior[i]);
		posterior.clear();
	}

	static inline void move(
		const node_sampler<K, V>& src,
		node_sampler<K, V>& dst)
	{
		dst.table_sizes = src.table_sizes;
		core::move(src.observation_assignments, dst.observation_assignments);
		core::move(src.posterior, dst.posterior);
		dst.table_assignments = src.table_assignments;
		dst.root_assignments = src.root_assignments;
		dst.descendant_observations = src.descendant_observations;
		dst.table_count = src.table_count;
		dst.table_capacity = src.table_capacity;
		dst.customer_count = src.customer_count;
		dst.children = src.children;
		dst.parent = src.parent;
		dst.n = src.n;
	}

	template<typename Metric>
	static inline long unsigned int size_of(
		const node_sampler<K, V>& n, const Metric& metric)
	{
		long unsigned int sum = core::size_of(n.observation_assignments)
				+ core::size_of(n.children, make_key_value_metric(dummy_metric(), metric))
				+ core::size_of(n.customer_count) + core::size_of(n.parent)
				+ core::size_of(n.observations, metric) + core::size_of(n.table_count)
				+ core::size_of(n.table_capacity) + core::size_of(n.posterior);

		for (unsigned int i = 0; i < n.table_count; i++)
			sum += core::size_of(n.descendant_observations[i], metric);
		sum += sizeof(array_histogram<K>) * (n.table_capacity - n.table_count);

		sum += 3 * sizeof(unsigned int) * n.table_capacity;
		return sum;
	}

	static inline void free(node_sampler<K, V>& n) {
		n.free();
		core::free(n.posterior);
	}

private:
	template<typename BaseDistribution, typename DataDistribution>
	inline bool initialize(
		const hdp_sampler<BaseDistribution, DataDistribution, K, V>& root,
		unsigned int initial_table_capacity)
	{
		table_sizes = NULL;
		table_count = 0;
		table_capacity = initial_table_capacity;
		table_assignments = NULL;
		root_assignments = NULL;
		descendant_observations = NULL;
		observation_assignments = NULL;
		children = NULL;
		customer_count = 0;

		children = (node_sampler<K, V>*) malloc(sizeof(node_sampler<K, V>) * n->children.capacity);
		observation_assignments = (unsigned int*) calloc(n->observations.capacity, sizeof(unsigned int));
		if (children == NULL || observation_assignments == NULL || !resize(initial_table_capacity)) {
			fprintf(stderr, "node_sampler.initialize ERROR: Unable to initialize tables.\n");
			free(0); return false;
		}

		/* add the observations from the hdp node */
		for (unsigned int i = 0; i < n->children.size; i++) {
			if (!init(children[i], root, n->children.values[i], this, 8)) {
				fprintf(stderr, "node_sampler.initialize ERROR: Unable to initialize child node.\n");
				free(i); return false;
			}

			for (unsigned int j = 0; j < children[i].table_count; j++) {
				/* assign the tables in the child nodes to tables in this node */
				if (!sample_initial_assignment<DataDistribution>(
					*this, root.pi(), children[i].descendant_observations[j], children[i].table_assignments[j]))
				{
					fprintf(stderr, "node_sampler.initialize ERROR: Unable to add to descendant_observations.\n");
					free(i + 1); return false;
				}
			}
		}
		for (unsigned int i = 0; i < observation_count(); i++) {
			if (!sample_initial_assignment<DataDistribution>(
				*this, root.pi(), n->observations[i], observation_assignments[i]))
			{
				fprintf(stderr, "node_sampler.initialize ERROR: Unable to add to descendant_observations.\n");
				free(0); return false;
			}
		}
		return true;
	}

	inline void free(unsigned int child_count) {
		if (children != NULL) {
			for (unsigned int i = 0; i < child_count; i++)
				core::free(children[i]);
			core::free(children);
		}
		for (unsigned int i = 0; i < posterior.length; i++)
			core::free(posterior[i]);
		for (unsigned int i = 0; i < table_count; i++)
			core::free(descendant_observations[i]);
		if (table_assignments != NULL) core::free(table_assignments);
		if (root_assignments != NULL) core::free(root_assignments);
		if (descendant_observations != NULL) core::free(descendant_observations);
		if (observation_assignments != NULL) core::free(observation_assignments);
		if (table_sizes != NULL) core::free(table_sizes);
	}

	inline void free() { free(child_count()); }

	inline bool resize(unsigned int new_capacity)
	{
		unsigned int* new_sizes = (unsigned int*) realloc(
				table_sizes, sizeof(unsigned int) * new_capacity);
		if (new_sizes == NULL) {
			fprintf(stderr, "node_sampler.resize ERROR: Unable to expand table_sizes.\n");
			return false;
		}
		table_sizes = new_sizes;

		unsigned int* new_table_assignments = (unsigned int*) realloc(
				table_assignments, sizeof(unsigned int) * new_capacity);
		if (new_table_assignments == NULL) {
			fprintf(stderr, "node_sampler.resize ERROR: Unable to expand table_assignments.\n");
			return false;
		}
		table_assignments = new_table_assignments;

		unsigned int* new_root_assignments = (unsigned int*) realloc(
				root_assignments, sizeof(unsigned int) * new_capacity);
		if (new_root_assignments == NULL) {
			fprintf(stderr, "node_sampler.resize ERROR: Unable to expand root_assignments.\n");
			return false;
		}
		root_assignments = new_root_assignments;

		array_histogram<K>* new_descendant_observations = (array_histogram<K>*) realloc(
				descendant_observations, sizeof(array_histogram<K>) * new_capacity);
		if (new_descendant_observations == NULL) {
			fprintf(stderr, "node_sampler.resize ERROR: Unable to expand descendant_observations.\n");
			return false;
		}
		descendant_observations = new_descendant_observations;
		return true;
	}

	template<typename A, typename B, typename C, typename D>
	friend bool init(node_sampler<C, D>&, const hdp_sampler<A, B, C, D>&,
		node<C, D>&, node_sampler<C, D>*, unsigned int);

	template<typename A, typename B>
	friend bool read_node(A& n, B& stream, typename A::node_type&);

	template<typename A, typename B, typename C, typename D>
	friend bool read(node_sampler<A, B>& n, C& stream, node<A, B>& hdp_node, D& parent);
};

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
inline bool init(node_sampler<K, V>& sampler,
	const hdp_sampler<BaseDistribution, DataDistribution, K, V>& root,
	node<K, V>& n, node_sampler<K, V>* parent, unsigned int table_capacity)
{
	sampler.n = &n;
	sampler.parent = parent;
	if (!array_init(sampler.posterior, 32)) {
		fprintf(stderr, "init ERROR: Unable to initialize posterior sample array in node.\n");
		return false;
	} else if (!sampler.initialize(root, table_capacity)) {
		fprintf(stderr, "init ERROR: Error during initialization of node.\n");
		free(sampler.posterior);
		return false;
	}
	return true;
}

template<typename K, typename V>
bool copy(const node_sampler<K, V>& src, node_sampler<K, V>& dst,
	const hash_map<const node<K, V>*, node<K, V>*>& node_map,
	node_sampler<K, V>* parent = NULL)
{
	dst.n = node_map.get(src.n);
	dst.parent = parent;
	dst.table_count = src.table_count;
	dst.table_capacity = src.table_count;
	dst.customer_count = src.customer_count;
	if (!array_init(dst.posterior, 1))
		return false;
	dst.children = (node_sampler<K, V>*) malloc(sizeof(node_sampler<K, V>) * src.child_count());
	if (dst.children == NULL) {
		free(dst.posterior);
		return false;
	}
	dst.observation_assignments = (unsigned int*) malloc(sizeof(unsigned int) * src.observation_count());
	if (dst.observation_assignments == NULL) {
		free(dst.posterior);
		free(dst.children);
		return false;
	}
	memcpy(dst.observation_assignments, src.observation_assignments, sizeof(unsigned int) * src.observation_count());
	dst.descendant_observations = (array_histogram<K>*) malloc(sizeof(array_histogram<K>) * src.table_count);
	if (dst.descendant_observations == NULL) {
		free(dst.posterior);
		free(dst.observation_assignments);
		free(dst.children);
		return false;
	}
	dst.table_sizes = (unsigned int*) malloc(sizeof(unsigned int) * src.table_count);
	if (dst.table_sizes == NULL) {
		free(dst.posterior);
		free(dst.observation_assignments);
		free(dst.children);
		free(dst.descendant_observations);
		return false;
	}
	memcpy(dst.table_sizes, src.table_sizes, sizeof(unsigned int) * src.table_count);
	dst.root_assignments = (unsigned int*) malloc(sizeof(unsigned int) * src.table_count);
	if (dst.root_assignments == NULL) {
		free(dst.posterior);
		free(dst.observation_assignments);
		free(dst.children);
		free(dst.descendant_observations);
		free(dst.table_sizes);
		return false;
	}
	memcpy(dst.root_assignments, src.root_assignments, sizeof(unsigned int) * src.table_count);
	dst.table_assignments = (unsigned int*) malloc(sizeof(unsigned int) * src.table_count);
	if (dst.table_assignments == NULL) {
		free(dst.posterior);
		free(dst.observation_assignments);
		free(dst.children);
		free(dst.descendant_observations);
		free(dst.table_sizes);
		free(dst.root_assignments);
		return false;
	}
	memcpy(dst.table_assignments, src.table_assignments, sizeof(unsigned int) * src.table_count);

	for (unsigned int i = 0; i < src.table_count; i++)
		if (!init(dst.descendant_observations[i], src.descendant_observations[i]))
			return false; /* TODO: free memory (also add appropriate error messages) */
	for (unsigned int i = 0; i < src.child_count(); i++)
		if (!copy(src.children[i], dst.children[i], node_map, &dst))
			return false; /* TODO: free memory (also add appropriate error messages) */
	return true;
}

template<typename K, typename V>
struct collapsed_root_sample {
	array_histogram<K>* descendant_observations;
	unsigned int* table_sizes;
	unsigned int table_count;
	unsigned int customer_count;

	unsigned int* table_assignments;

	collapsed_root_sample(unsigned int table_count, unsigned int observation_count) : table_count(table_count) {
		if (!initialize(table_count, observation_count)) {
			fprintf(stderr, "collapsed_root_sample ERROR: Error during initialization.\n");
			exit(EXIT_FAILURE);
		}
	}

	~collapsed_root_sample() { free(); }

	inline unsigned int root_assignment(unsigned int i) const {
		return i;
	}

	template<typename Metric>
	static inline long unsigned int size_of(const collapsed_root_sample<K, V>& sample, const Metric& metric) {
		long unsigned int sum = core::size_of(sample.table_count) + core::size_of(sample.customer_count);
		for (unsigned int i = 0; i < sample.table_count; i++)
			sum += core::size_of(sample.descendant_observations[i], metric);
		return sum + (sample.table_count + sample.customer_count) * sizeof(unsigned int);
	}

	static void free(collapsed_root_sample<K, V>& sample) {
		sample.free();
	}

private:
	inline bool initialize(unsigned int table_count, unsigned int observation_count) {
		descendant_observations = (array_histogram<K>*) malloc(sizeof(array_histogram<K>) * table_count);
		if (descendant_observations == NULL) {
			fprintf(stderr, "collapsed_root_sample.initialize ERROR: Unable to"
					" initialize descendant_observations.\n");
			return false;
		}

		table_sizes = (unsigned int*) malloc(sizeof(unsigned int) * table_count);
		if (table_sizes == NULL) {
			fprintf(stderr, "collapsed_root_sample.initialize ERROR:"
					" Unable to initialize table_sizes.\n");
			core::free(descendant_observations);
			return false;
		}

		if (observation_count == 0)
			observation_count = 1;
		table_assignments = (unsigned int*) malloc(sizeof(unsigned int) * observation_count);
		if (table_assignments == NULL) {
			fprintf(stderr, "collapsed_root_sample.initialize ERROR:"
					" Unable to initialize table_assignments.\n");
			core::free(descendant_observations);
			core::free(table_sizes);
			return false;
		}

		return true;
	}

	inline void free() {
		for (unsigned int i = 0; i < table_count; i++)
			core::free(descendant_observations[i]);
		core::free(descendant_observations);
		core::free(table_sizes);
		core::free(table_assignments);
	}

	template<typename A, typename B>
	friend bool init(collapsed_root_sample<A, B>&, unsigned int, unsigned int);
};

template<typename K, typename V>
bool init(collapsed_root_sample<K, V>& sample, unsigned int table_count, unsigned int observation_count) {
	sample.table_count = table_count;
	return sample.initialize(table_count, observation_count);
}

template<typename K, typename V, typename AtomReader>
bool read(collapsed_root_sample<K, V>& sample, FILE* in, node_sample_scribe<AtomReader>& reader)
{
	if (!read(sample.table_count, in)) return false;

	sample.descendant_observations = (array_histogram<K>*)
			malloc(sizeof(array_histogram<K>) * sample.table_count);
	if (sample.descendant_observations == NULL)
		return false;
	sample.table_sizes = (unsigned int*) malloc(sizeof(unsigned int) * sample.table_count);
	if (sample.table_sizes == NULL) {
		free(sample.descendant_observations);
		return false;
	}
	sample.table_assignments = (unsigned int*) malloc(sizeof(unsigned int) * reader.observation_count);
	if (sample.table_assignments == NULL) {
		free(sample.descendant_observations);
		free(sample.table_sizes);
		return false;
	}

	for (unsigned int i = 0; i < sample.table_count; i++) {
		if (!read(sample.descendant_observations[i], in, reader.atom_scribe))
			return false;
	}
	if (!read(sample.table_sizes, in, sample.table_count)) return false;
	if (!read(sample.table_assignments, in, reader.observation_count)) return false;

	sample.customer_count = 0;
	for (unsigned int i = 0; i < sample.table_count; i++)
		sample.customer_count += sample.table_sizes[i];

	return true;
}

template<typename K, typename V, typename AtomWriter>
bool write(const collapsed_root_sample<K, V>& sample, FILE* out, node_sample_scribe<AtomWriter>& writer)
{
	if (!write(sample.table_count, out)) return false;
	for (unsigned int i = 0; i < sample.table_count; i++) {
		if (!write(sample.descendant_observations[i], out, writer.atom_scribe))
			return false;
	}
	if (!write(sample.table_sizes, out, sample.table_count)) return false;
	if (!write(sample.table_assignments, out, writer.observation_count)) return false;
	return true;
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
struct hdp_sampler
{
	typedef K atom_type;
	typedef V value_type;
	typedef BaseDistribution base_distribution_type;
	typedef DataDistribution data_distribution_type;
	typedef node_sampler<K, V> child_type;
	typedef hdp<BaseDistribution, DataDistribution, K, V> node_type;

	node_sampler<K, V>* children;
	node_type* n;

	/**
	 * state variables for Gibbs sampling
	 */
	unsigned int* observation_assignments;

	/**
	 * In the Chinese restaurant representation, this is a
	 * histogram of table assignments of the samples from
	 * the CRP at this node.
	 *
	 * In the direct assignment representation, this is a
	 * histogram of root assignments of the samples in the
	 * CRP at this node.
	 */
	unsigned int* table_sizes;

	array_histogram<K>* descendant_observations;
	unsigned int table_count;
	unsigned int table_capacity;
	unsigned int customer_count;

	/* storage for posterior samples */
	array<collapsed_root_sample<K, V>> posterior;

	hdp_sampler(hdp<BaseDistribution, DataDistribution, K, V>& root) :
		n(&root), table_count(0), table_capacity(8), posterior(32)
	{
		if (!initialize())
			exit(EXIT_FAILURE);
	}

	~hdp_sampler() { free(); }

	inline V alpha() const {
		return n->get_alpha();
	}

	inline V log_alpha() const {
		return n->get_log_alpha();
	}

	inline const BaseDistribution& pi() const {
		return n->pi;
	}

	inline unsigned int child_count() const {
		return (unsigned int) n->children.size;
	}

	inline const node_sampler<K, V>& get_child(unsigned int key, bool& contains) const {
		unsigned int index = n->children.index_of(key);
		contains = (index < child_count());
		return children[index];
	}

	inline unsigned int child_key(unsigned int index) const {
		return n->children.keys[index];
	}

	inline unsigned int observation_count() const {
		return (unsigned int) n->observations.length;
	}

	inline const K& get_observation(unsigned int index) const {
		return n->observations[index];
	}

	inline unsigned int root_assignment(unsigned int i) const {
		return i;
	}

	inline unsigned int table_assignment(unsigned int i) const {
		return i;
	}

	inline void move_table(unsigned int src, unsigned int dst,
		cache<BaseDistribution, DataDistribution, K, V>& cache)
	{
		array_histogram<K>::move(descendant_observations[src], descendant_observations[dst]);
		table_sizes[dst] = table_sizes[src];
		cache.on_move_table(src, dst);
	}

	template<typename Observations>
	inline bool move_to_table(const Observations& observations,
		unsigned int src, unsigned int dst,
		cache<BaseDistribution, DataDistribution, K, V>& cache)
	{
		if (!descendant_observations[dst].add(observations)) {
			fprintf(stderr, "hdp_sampler.move_to_table ERROR: Unable to initialize new histogram.\n");
			return false;
		}
		descendant_observations[src].subtract(observations);
		descendant_observations[src].remove_zeros();
		return cache.on_move_to_table(*this, src, dst, observations);
	}

	template<typename Observations>
	inline bool move_to_new_table(
		const Observations& observations, unsigned int src,
		cache<BaseDistribution, DataDistribution, K, V>& cache)
	{
		if (!descendant_observations[table_count - 1].add(observations)) {
			fprintf(stderr, "hdp_sampler.move_to_new_table ERROR: Unable to initialize new histogram.\n");
			return false;
		}
		descendant_observations[src].subtract(observations);
		descendant_observations[src].remove_zeros();
		return cache.on_move_to_new_table(*this, src, observations);
	}

	template<typename Observations>
	inline bool add_to_table(
		const Observations& observations, unsigned int table,
		cache<BaseDistribution, DataDistribution, K, V>& cache)
	{
		if (!descendant_observations[table].add(observations)) {
			fprintf(stderr, "hdp_sampler.add_to_table ERROR: Unable to initialize new histogram.\n");
			return false;
		}
		return cache.on_add_to_table(*this, table, observations);
	}

	template<typename Observations>
	inline bool add_to_new_table(const Observations& observations,
		cache<BaseDistribution, DataDistribution, K, V>& cache)
	{
		if (!descendant_observations[table_count - 1].add(observations)) {
			fprintf(stderr, "hdp_sampler.add_to_new_table ERROR: Unable to initialize new histogram.\n");
			return false;
		}
		return cache.on_add_to_new_table(*this, observations);
	}

	template<typename Observations>
	inline bool remove_from_table(
		const Observations& observations, unsigned int table,
		cache<BaseDistribution, DataDistribution, K, V>& cache)
	{
		descendant_observations[table].subtract(observations);
		descendant_observations[table].remove_zeros();
		return cache.on_remove_from_table(*this, table, observations);
	}

	inline bool new_table() {
		unsigned int new_capacity = table_capacity;
		while (new_capacity < table_count + 2)
			new_capacity *= 2;
		if (table_capacity != new_capacity && !resize(new_capacity)) {
			fprintf(stderr, "hdp_sampler.new_table ERROR: Unable to expand table array.\n");
			return false;
		}
		table_capacity = new_capacity;

		if (!init(descendant_observations[table_count], 8)) {
			fprintf(stderr, "hdp_sampler.new_table ERROR: Unable to initialize new histogram.\n");
			return false;
		}
		table_count++;
		return true;
	}

	inline bool new_initial_table() {
		if (!init(descendant_observations[table_count], 8)) {
			fprintf(stderr, "hdp_sampler.new_table ERROR: Unable to initialize new histogram.\n");
			return false;
		}
		table_sizes[table_count] = 0;
		table_count++;
		return true;
	}

	/* NOTE: this does not change the total number of tables */
	inline void free_table(unsigned int i) {
		core::free(descendant_observations[i]);
	}

	inline void set_table_count(unsigned int new_count) {
		table_count = new_count;
	}

	bool add_sample() {
		for (unsigned int i = 0; i < child_count(); i++)
			children[i].add_sample(table_count);

		if (!posterior.ensure_capacity(posterior.length + 1)) {
			fprintf(stderr, "hdp_sampler.add_sample ERROR: Unable to expand posterior sample array.\n");
			return false;
		} else if (!init(posterior[(unsigned int) posterior.length], table_count, observation_count())) {
			fprintf(stderr, "hdp_sampler.add_sample ERROR: Unable to initialize new sample.\n");
			return false;
		}

		collapsed_root_sample<K, V>& sample = posterior[(unsigned int) posterior.length];
		for (unsigned int i = 0; i < observation_count(); i++)
			sample.table_assignments[i] = observation_assignments[i];

		sample.table_count = 0;
		for (unsigned int i = 0; i < table_count; i++) {
			if (!array_histogram<K>::copy(descendant_observations[i], sample.descendant_observations[i])) {
				fprintf(stderr, "hdp_sampler.add_sample ERROR: Unable to copy descendant observations.\n");
				core::free(sample);
				return false;
			}
			sample.table_sizes[i] = table_sizes[i];
			sample.table_count++;
		}
		sample.customer_count = customer_count;
		posterior.length++;
		return true;
	}

	inline void clear_samples() {
		for (unsigned int i = 0; i < child_count(); i++)
			children[i].clear_samples();

		for (unsigned int i = 0; i < posterior.length; i++)
			core::free(posterior[i]);
		posterior.clear();
	}

	inline void relabel_tables(const unsigned int* table_map,
		cache<BaseDistribution, DataDistribution, K, V>& cache)
	{
		/* apply the table map to the tables in the child nodes */
		for (unsigned int i = 0; i < child_count(); i++)
			children[i].relabel_parent_tables(table_map);
		for (unsigned int i = 0; i < observation_count(); i++)
			observation_assignments[i] = table_map[observation_assignments[i]];
		cache.relabel_tables(table_map);
	}

	template<typename Metric>
	static inline long unsigned int size_of(
			const hdp<BaseDistribution, DataDistribution, K, V>& h, const Metric& metric) {
		long unsigned int sum = core::size_of(h.children) + core::size_of(h.observations)
				+ core::size_of(h.table_count) + core::size_of(h.table_capacity)
				+ core::size_of(h.customer_count) + core::size_of(h.posterior);
		sum += sizeof(unsigned int) * h.depth; /* for alpha */

		for (unsigned int i = 0; i < h.table_count; i++)
			sum += core::size_of(h.descendant_observations[i]);
		sum += (sizeof(DataDistribution) + sizeof(array_histogram<K>)) * (h.table_capacity - h.table_count);

		sum += sizeof(unsigned int) * h.table_capacity; /* for table_sizes */
		return sum;
	}

	static inline void free(hdp_sampler<BaseDistribution, DataDistribution, K, V>& h) {
		h.free();
		core::free(h.posterior);
	}

private:
	bool initialize() {
		table_count = 0;
		children = NULL;
		observation_assignments = NULL;
		descendant_observations = NULL;
		table_sizes = NULL;
		customer_count = 0;

		children = (node_sampler<K, V>*) malloc(sizeof(node_sampler<K, V>) * n->children.capacity);
		observation_assignments = (unsigned int*) calloc(sizeof(unsigned int), n->observations.capacity);
		if (children == NULL || observation_assignments == NULL || !resize(table_capacity)) {
			fprintf(stderr, "hdp_sampler.initialize ERROR: Unable to initialize tables.\n");
			free(0); return false;
		}

		/* add the observations from the hdp */
		for (unsigned int i = 0; i < n->children.size; i++) {
			if (!init(children[i], *this, n->children.values[i], (node_sampler<K, V>*) NULL, 8)) {
				fprintf(stderr, "hdp_sampler.initialize ERROR: Unable to initialize child node.\n");
				free(i); return false;
			}

			for (unsigned int j = 0; j < children[i].table_count; j++) {
				/* assign the tables in the child nodes to tables in this node */
				if (!sample_initial_assignment<DataDistribution>(
					*this, pi(), children[i].descendant_observations[j], children[i].table_assignments[j]))
				{
					fprintf(stderr, "hdp_sampler.initialize ERROR: Unable to add to descendant_observations.\n");
					free(i + 1); return false;
				}
			}
		}
		for (unsigned int i = 0; i < observation_count(); i++) {
			if (!sample_initial_assignment<DataDistribution>(
				*this, pi(), n->observations[i], observation_assignments[i]))
			{
				fprintf(stderr, "hdp_sampler.initialize ERROR: Unable to add to descendant_observations.\n");
				free(0); return false;
			}
		}

		recompute_root_assignments(*this);
		return true;
	}

	inline void free(unsigned int child_count) {
		if (children != NULL) {
			for (unsigned int i = 0; i < child_count; i++)
				core::free(children[i]);
			core::free(children);
		}
		if (descendant_observations != NULL) {
			for (unsigned int i = 0; i < table_count; i++)
				core::free(descendant_observations[i]);
		}
		for (unsigned int i = 0; i < posterior.length; i++)
			core::free(posterior[i]);
		if (observation_assignments != NULL) core::free(observation_assignments);
		if (descendant_observations != NULL) core::free(descendant_observations);
		if (table_sizes != NULL) core::free(table_sizes);
	}

	inline void free() { free(child_count()); }

	inline bool resize(unsigned int new_capacity) {
		array_histogram<K>* new_descendant_observations = (array_histogram<K>*)
				realloc(descendant_observations, sizeof(array_histogram<K>) * new_capacity);
		if (new_descendant_observations == NULL) {
			fprintf(stderr, "hdp_sampler.resize ERROR: Unable to expand descendant_observations.\n");
			return false;
		}
		descendant_observations = new_descendant_observations;

		unsigned int* new_table_sizes = (unsigned int*)
				realloc(table_sizes, sizeof(unsigned int) * new_capacity);
		if (new_table_sizes == NULL) {
			fprintf(stderr, "hdp_sampler.resize ERROR: Unable to expand table_sizes.\n");
			return false;
		}
		table_sizes = new_table_sizes;
		return true;
	}

	template<typename A, typename B, typename C, typename D>
	friend bool init(hdp_sampler<A, B, C, D>&, hdp<A, B, C, D>&, int);

	template<typename A, typename B>
	friend bool read_node(A& n, B& stream, typename A::node_type&);
};

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
bool init(hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	hdp<BaseDistribution, DataDistribution, K, V>& root, int initial_table_count = 1)
{
	h.n = &root;
	h.table_count = 0;
	h.table_capacity = 1 << (core::log2(initial_table_count) + 2);
	if (!array_init(h.posterior, 32)) {
		fprintf(stderr, "init ERROR: Unable to initialize posterior sample array in hdp.\n");
		return false;
	}
	return h.initialize();
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
bool copy(
	const hdp_sampler<BaseDistribution, DataDistribution, K, V>& src,
	hdp_sampler<BaseDistribution, DataDistribution, K, V>& dst,
	hdp<BaseDistribution, DataDistribution, K, V>& new_root,
	const hash_map<const node<K, V>*, node<K, V>*>& node_map)
{
	dst.n = &new_root;
	dst.table_count = src.table_count;
	dst.table_capacity = src.table_count;
	dst.customer_count = src.customer_count;
	if (!array_init(dst.posterior, 1))
		return false;
	dst.children = (node_sampler<K, V>*) malloc(sizeof(node_sampler<K, V>) * src.child_count());
	if (dst.children == NULL) {
		free(dst.posterior);
		return false;
	}
	dst.observation_assignments = (unsigned int*) malloc(sizeof(unsigned int) * src.observation_count());
	if (dst.observation_assignments == NULL) {
		free(dst.posterior);
		free(dst.children);
		return false;
	}
	memcpy(dst.observation_assignments, src.observation_assignments, sizeof(unsigned int) * src.observation_count());
	dst.descendant_observations = (array_histogram<K>*) malloc(sizeof(array_histogram<K>) * src.table_count);
	if (dst.descendant_observations == NULL) {
		free(dst.posterior);
		free(dst.observation_assignments);
		free(dst.children);
		return false;
	}
	dst.table_sizes = (unsigned int*) malloc(sizeof(unsigned int) * src.table_count);
	if (dst.table_sizes == NULL) {
		free(dst.posterior);
		free(dst.observation_assignments);
		free(dst.children);
		free(dst.descendant_observations);
		return false;
	}
	memcpy(dst.table_sizes, src.table_sizes, sizeof(unsigned int) * src.table_count);

	for (unsigned int i = 0; i < src.table_count; i++)
		if (!init(dst.descendant_observations[i], src.descendant_observations[i]))
			return false; /* TODO: free memory (also add appropriate error messages) */
	for (unsigned int i = 0; i < src.child_count(); i++)
		if (!copy(src.children[i], dst.children[i], node_map))
			return false; /* TODO: free memory (also add appropriate error messages) */
	return true;
}

template<typename NodeType, typename Stream, typename KeyPrinter,
	typename AtomPrinter, typename K = typename NodeType::atom_type>
bool print(const NodeType& node, Stream& out,
	KeyPrinter& key_printer, AtomPrinter& atom_printer,
	unsigned int level = 0)
{
	bool success = true;
	if (node.observation_count() > 0) {
		array_histogram<K> histogram = array_histogram<K>(node.observation_count());
		for (const K& observation : node.n->observations)
			histogram.add_unsorted(observation);
		insertion_sort(histogram.counts.keys, histogram.counts.values, (unsigned int) histogram.counts.size, dummy_sorter());

		for (unsigned int i = 0; i < histogram.counts.size; i++) {
			if (i > 0) success &= print(", ", out);
			success &= print(histogram.counts.keys[i], out, atom_printer)
					&& print(':', out) && print(histogram.counts.values[i], out);
		}
	}

	bool first = true;
	for (unsigned int i = 0; i < node.child_count(); i++) {
		if (node.children[i].customer_count == 0) continue;
		if (first) first = false;
		else success &= print(' ', out);
		success &= print('{', out) && print(node.child_key(i), out, key_printer, level) && print(": ", out)
				&& print(node.children[i], out, key_printer, atom_printer, level + 1) && print('}', out);
	}
	return success;
}

template<typename K, typename V>
inline void fix_parent_pointers(node_sampler<K, V>& node) {
	for (unsigned int i = 0; i < node.child_count(); i++)
		node.children[i].parent = &node;
}

template<typename K, typename V>
inline node_sampler<K, V>* get_parent(node_sampler<K, V>& parent) {
	return &parent;
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
constexpr node_sampler<K, V>* get_parent(hdp_sampler<BaseDistribution, DataDistribution, K, V>& parent) {
	return NULL;
}

template<typename K, typename V, typename Stream, typename ParentType>
bool read(node_sampler<K, V>& n, Stream& stream, node<K, V>& hdp_node, ParentType& parent)
{
	n.parent = get_parent(parent);
	n.table_assignments = NULL;
	n.root_assignments = NULL;
	if (!read_node(n, stream, hdp_node))
		return false;

	if (!read(n.table_assignments, stream, n.table_count)) {
		fprintf(stderr, "read ERROR: Insufficient memory for table assignment array.\n");
		n.free(); free(n.posterior); return false;
	}
	return true;
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V, typename Stream>
bool read(hdp_sampler<BaseDistribution, DataDistribution, K, V>& n,
	Stream& stream, hdp<BaseDistribution, DataDistribution, K, V>& h)
{
	if (!read_node(n, stream, h)) return false;
	recompute_root_assignments(n);
	return true;
}

template<typename NodeType, typename Stream>
bool write_node(const NodeType& n, Stream& stream)
{
	if (!write(n.table_count, stream)
	 || !write(n.observation_assignments, stream, n.observation_count()))
		 return false;
	for (unsigned int i = 0; i < n.child_count(); i++)
		if (!write(n.children[i], stream)) return false;
	return true;
}

template<typename K, typename V, typename Stream>
bool write(const node_sampler<K, V>& n, Stream& stream)
{
	return write_node(n, stream)
		&& write(n.table_assignments, stream, n.table_count);
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V, typename Stream>
bool write(const hdp_sampler<BaseDistribution, DataDistribution, K, V>& n, Stream& stream)
{
	return write_node(n, stream);
}

template<typename NodeType, typename Stream>
bool read_node(NodeType& n, Stream& stream, typename NodeType::node_type& hdp_node)
{
	unsigned int table_count;
	if (!array_init(n.posterior, 32)) {
		return false;
	} else if (!read(table_count, stream)) {
		free(n.posterior);
		return false;
	}
	n.table_count = 0;
	n.customer_count = 0;
	n.table_capacity = max(1u, table_count);
	n.n = &hdp_node;

	n.descendant_observations = NULL; n.table_sizes = NULL;
	n.children = (typename NodeType::child_type*) malloc(sizeof(typename NodeType::child_type) * hdp_node.children.capacity);
	n.observation_assignments = (unsigned int*) calloc(sizeof(unsigned int), hdp_node.observations.capacity);
	if (n.children == NULL || n.observation_assignments == NULL || !n.resize(n.table_capacity)
	 || !read(n.observation_assignments, stream, (unsigned int) hdp_node.observations.length))
	{
		fprintf(stderr, "read_node ERROR: Unable to initialize tables.\n");
		n.free(0); free(n.posterior); return false;
	}
	memset(n.table_sizes, 0, sizeof(unsigned int) * table_count);

	for (unsigned int i = 0; i < table_count; i++) {
		if (!init(n.descendant_observations[i], 8)) {
			fprintf(stderr, "read_node ERROR: Unable to initialize descendant observation histograms.\n");
			n.free(0); free(n.posterior); return false;
		}
		n.table_count++;
	}

	for (unsigned int i = 0; i < hdp_node.children.size; i++) {
		if (!read(n.children[i], stream, hdp_node.children.values[i], n)) {
			fprintf(stderr, "read_node ERROR: Unable to read child node.\n");
			n.free(i); free(n.posterior); return false;
		}
	}

	/* compute descendant observations, table sizes, and customer count */
	for (unsigned int i = 0; i < n.observation_count(); i++) {
		unsigned int table = n.observation_assignments[i];
		if (!n.descendant_observations[table].add(n.get_observation(i))) {
			fprintf(stderr, "read_node ERROR: Unable to add observation.\n");
			n.free(); free(n.posterior); return false;
		}
		n.table_sizes[table]++;
		n.customer_count++;
	}
	for (unsigned int i = 0; i < n.child_count(); i++) {
		for (unsigned int j = 0; j < n.children[i].table_count; j++) {
			unsigned int table = n.children[i].table_assignments[j];
			if (!n.descendant_observations[table].add(n.children[i].descendant_observations[j])) {
				fprintf(stderr, "read_node ERROR: Unable to add child node's observations.\n");
				n.free(); free(n.posterior); return false;
			}
			n.table_sizes[table]++;
			n.customer_count++;
		}
	}
	return true;
}

template<typename DataDistribution, typename NodeType, typename BaseDistribution, typename Observations>
inline bool sample_initial_assignment(
	NodeType& n,
	const BaseDistribution& pi,
	const Observations& observations,
	unsigned int& assignment)
{
	typedef typename NodeType::value_type V;

	/* compute likelihood of the observations for each root cluster */
	array<V> distribution = array<V>(n.table_count + 1);
	for (unsigned int i = 0; i < n.table_count; i++)
		distribution[i] = DataDistribution::log_conditional(pi, observations, n.descendant_observations[i]);
	distribution[n.table_count] = DataDistribution::log_probability(pi, observations);
	distribution.length = n.table_count + 1;

	normalize_exp(distribution.data, (unsigned int) distribution.length);
	assignment = sample_categorical(distribution.data, (unsigned int) distribution.length);
	if (assignment == n.table_count) {
		n.table_sizes[n.table_count] = 1;
		if (!n.new_table()) return false;
	} else {
		n.table_sizes[assignment]++;
	}
	n.customer_count++;
	return n.descendant_observations[assignment].add(observations);
}

template<typename RootType>
struct root_position {
	typedef RootType root_type;
	typedef typename RootType::atom_type atom_type;
	typedef typename RootType::value_type value_type;

	RootType& node;

	root_position(RootType& root) : node(root) { }

	inline unsigned int depth() const {
		return node.depth;
	}

	inline const RootType& get_root() const {
		return node;
	}

	inline RootType& get_root() {
		return node;
	}
};

template<typename RootType>
inline root_position<RootType> make_root_node(RootType& root) {
	return root_position<RootType>(root);
}

template<typename RootType>
struct child_position {
	typedef RootType root_type;
	typedef typename RootType::child_type node_type;
	typedef typename RootType::atom_type atom_type;
	typedef typename RootType::value_type value_type;

	RootType& root;
	node_type& node;

	child_position(RootType& root, node_type& node) : root(root), node(node) { }

	inline unsigned int depth() const {
		return root.depth;
	}

	inline RootType& get_root() {
		return root;
	}

	inline const RootType& get_root() const {
		return root;
	}
};

template<typename RootType>
inline child_position<RootType> make_child_node(
		const root_position<RootType>& root, typename RootType::child_type& node) {
	return child_position<RootType>(root.node, node);
}

template<typename RootType>
inline child_position<RootType> make_child_node(
		const child_position<RootType>& root, typename RootType::child_type& node) {
	return child_position<RootType>(root.root, node);
}

template<bool UpdateCustomerCount, typename K, typename V, typename Cache>
inline void decrement_table(node_sampler<K, V>& n, unsigned int table, Cache& cache)
{
#if !defined(NDEBUG)
	if (n.table_sizes[table] == 0) {
		fprintf(stderr, "decrement_table ERROR: table_sizes is invalid.\n");
		return;
	}
#endif
	n.table_sizes[table]--;
	if (UpdateCustomerCount)
		n.customer_count--;
}

template<bool UpdateCustomerCount, typename BaseDistribution,
	typename DataDistribution, typename K, typename V>
inline void decrement_table(
	hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	unsigned int table, cache<BaseDistribution, DataDistribution, K, V>& cache)
{
#if !defined(NDEBUG)
	if (h.table_sizes[table] == 0) {
		fprintf(stderr, "decrement_table ERROR: table_sizes is invalid.\n");
		return;
	}
#endif
	cache.on_change_table_size(table, h.table_sizes[table] - 1);
	h.table_sizes[table]--;
	if (UpdateCustomerCount)
		h.customer_count--;
}

template<bool UpdateCustomerCount, typename K, typename V, typename Cache>
inline void increment_table(node_sampler<K, V>& n, unsigned int table, Cache& cache)
{
	n.table_sizes[table]++;
	if (UpdateCustomerCount)
		n.customer_count++;
}

template<bool UpdateCustomerCount, typename BaseDistribution,
	typename DataDistribution, typename K, typename V>
inline void increment_table(
	hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	unsigned int table, cache<BaseDistribution, DataDistribution, K, V>& cache)
{
	cache.on_change_table_size(table, h.table_sizes[table] + 1);
	h.table_sizes[table]++;
	if (UpdateCustomerCount)
		h.customer_count++;
}

template<typename NodeType>
void compute_table_sizes(const NodeType& n, unsigned int* dst, unsigned int dst_length) {
	for (unsigned int i = 0; i < n.observation_count(); i++)
		add_to_histogram(dst, dst_length, n.observation_assignments[i]);
	for (unsigned int i = 0; i < n.child_count(); i++) {
		typename NodeType::child_type& child = n.children[i];
		for (unsigned int j = 0; j < child.table_count; j++)
			if (child.descendant_observations[j].counts.size != 0)
				add_to_histogram(dst, dst_length, child.table_assignments[j]);
	}
}

template<typename NodeType>
bool check_table(const NodeType& n, unsigned int table,
		const array_histogram<typename NodeType::atom_type>& expected_observations)
{
	bool success = true;

	auto computed_observations =
		array_histogram<typename NodeType::atom_type>((unsigned int) expected_observations.counts.size);
	for (unsigned int i = 0; i < n.child_count(); i++) {
		if (!n.children[i].get_observations(computed_observations, table)) {
			fprintf(stderr, "check_table ERROR: Unable to add observations from child node.\n");
			return false;
		}
	}

	for (unsigned int i = 0; i < n.observation_count(); i++) {
		if (n.observation_assignments[i] == table)
			computed_observations.add(n.n->observations[i]);
	}

	if (computed_observations == expected_observations)
		return success;
	fprintf(stderr, "check_table ERROR: descendant_observations of table %u is invalid.\n", table);
	return false;
}

template<typename K, typename V>
bool check_tables(const node_sampler<K, V>& n)
{
	unsigned int* expected_sizes = (unsigned int*) calloc(n.table_count, sizeof(unsigned int));
	compute_table_sizes(n, expected_sizes, n.table_count);
	if (!compare_histograms(expected_sizes, n.table_sizes, n.table_count, n.table_count)) {
		fprintf(stderr, "check_tables WARNING: table_sizes is inconsistent.\n");
		free(expected_sizes);
		return false;
	}
	free(expected_sizes);

	for (unsigned int i = 0; i < n.child_count(); i++) {
		if (n.children[i].parent != &n) {
			fprintf(stderr, "check_tables WARNING: Parent pointer is inconsistent.\n");
			return false;
		}
		if (!check_tables(n.children[i]))
			return false;
	}

	for (unsigned int i = 0; i < n.table_count; i++)
		if (!check_table(n, i, n.descendant_observations[i]))
			return false;
	return true;
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
bool check_tables(const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h)
{
	unsigned int* expected_table_sizes = (unsigned int*) calloc(h.table_count, sizeof(unsigned int));
	compute_table_sizes(h, expected_table_sizes, h.table_count);
	if (!compare_histograms(expected_table_sizes, h.table_sizes, h.table_count, h.table_count))
	{
		fprintf(stderr, "check_tables WARNING: table_sizes is inconsistent.\n");
		free(expected_table_sizes);
		return false;
	}
	free(expected_table_sizes);

	for (unsigned int i = 0; i < h.child_count(); i++) {
		if (h.children[i].parent != NULL) {
			fprintf(stderr, "check_tables WARNING: Parent pointer is inconsistent.\n");
			return false;
		}
		if (!check_tables(h.children[i]))
			return false;
	}

	for (unsigned int i = 0; i < h.table_count; i++)
		if (!check_table(h, i, h.descendant_observations[i]))
			return false;
	return true;
}

template<typename NodeType>
bool tables_sorted(const NodeType& h)
{
	for (unsigned int i = 0; i < h.child_count(); i++)
		if (!tables_sorted(h.children[i]))
			return false;

	for (unsigned int i = 0; i < h.table_count; i++)
		if (!h.descendant_observations[i].is_sorted())
			return false;
	return true;
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
bool is_valid(const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	const cache<BaseDistribution, DataDistribution, K, V>& cache)
{
	if (!tables_sorted(h)) {
		fprintf(stderr, "is_valid WARNING: Table array_histograms are not sorted.\n");
		return false;
	} else if (!check_tables(h)) {
		fprintf(stderr, "sample_assignment WARNING: The HDP structure is invalid.\n");
		return false;
	} else if (!cache.is_valid(h)) {
		fprintf(stderr, "sample_assignment WARNING: Table distribution cache is invalid.\n");
		return false;
	}
	return true;
}

template<typename K, typename V>
void prepare_sampler(node_sampler<K, V>& n, unsigned int root_table_count)
{
#if !defined(NDEBUG)
	/* for debugging, check the consistency of the table_sizes data structure */
	unsigned int* expected_sizes = (unsigned int*) calloc(n.table_count, sizeof(unsigned int));
	compute_table_sizes(n, expected_sizes, n.table_count);
	if (!compare_histograms(expected_sizes, n.table_sizes, n.table_count, n.table_count))
		fprintf(stderr, "prepare_sampler WARNING: table_sizes is inconsistent.\n");
	free(expected_sizes);
#endif

	/* sort any histograms */
	for (unsigned int j = 0; j < n.table_count; j++) {
		if (n.descendant_observations[j].counts.size <= 1)
			continue;
		quick_sort(n.descendant_observations[j].counts.keys,
				n.descendant_observations[j].counts.values,
				(unsigned int) n.descendant_observations[j].counts.size, dummy_sorter());
	}

	/* recurse to all descendants */
	for (unsigned int i = 0; i < n.child_count(); i++)
		prepare_sampler(n.children[i], root_table_count);
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
inline void prepare_sampler(
		hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
		cache<BaseDistribution, DataDistribution, K, V>& cache)
{
#if !defined(NDEBUG)
	/* for debugging, check the consistency of the table_sizes data structure */
	unsigned int* expected_sizes = (unsigned int*) calloc(h.table_count, sizeof(unsigned int));
	compute_table_sizes(h, expected_sizes, h.table_count);
	if (!compare_histograms(expected_sizes, h.table_sizes, h.table_count, h.table_count))
		fprintf(stderr, "prepare_sampler WARNING: table_sizes is inconsistent.\n");
	free(expected_sizes);
#endif

	h.clear_samples();
	cache.prepare_sampler(h);

	/* recurse to all descendants */
	for (unsigned int i = 0; i < h.child_count(); i++)
		prepare_sampler(h.children[i], h.table_count);
}

template<typename K, typename V, typename NodeType>
void recompute_root_assignments(node_sampler<K, V>& n, const NodeType& parent) {
	for (unsigned int i = 0; i < n.table_count; i++) {
		unsigned int assignment = n.table_assignments[i];
		n.root_assignments[i] = parent.root_assignment(assignment);
	}

	for (unsigned int i = 0; i < n.child_count(); i++)
		recompute_root_assignments(n.children[i], n);
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
void recompute_root_assignments(hdp_sampler<BaseDistribution, DataDistribution, K, V>& h) {
	for (unsigned int i = 0; i < h.child_count(); i++)
		recompute_root_assignments(h.children[i], h);
}

template<typename NodeType, typename Cache>
void remove_empty_tables(NodeType& n, Cache& cache)
{
	/* recursively remove empty tables in the child nodes */
	for (unsigned int i = 0; i < n.child_count(); i++)
		remove_empty_tables(n.children[i], cache);

	/* compute the table map for the children of this node */
	unsigned int new_length = 0;
	unsigned int* table_map = (unsigned int*) alloca(
			sizeof(unsigned int) * n.table_count);
	for (unsigned int i = 0; i < n.table_count; i++) {
		if (n.table_sizes[i] == 0) {
			n.free_table(i);
		} else {
			table_map[i] = new_length;
			if (new_length != i)
				n.move_table(i, new_length, cache);
			new_length++;
		}
	}
	n.set_table_count(new_length);

	n.relabel_tables(table_map, cache);
}

template<typename V>
struct sample_result {
	unsigned int selected_table;
	V new_table_probability;
};

template<bool Collapsed, typename RootType, typename Distribution, typename V>
inline sample_result<V> sample_table(
	const root_position<RootType>& n,
	const Distribution& root_distribution,
	const V& new_table_probability)
{
	/* sample either a new or old table at the root */
	V max = root_distribution.max();
	V old_table = root_distribution.sum() / (n.node.alpha() + n.node.customer_count);
	V new_table = exp(n.node.log_alpha() + new_table_probability - max) / (n.node.alpha() + n.node.customer_count);

	V uniform = sample_uniform<double>() * (old_table + new_table);
	if (old_table == 0.0 || uniform < new_table)
		return {n.node.table_count, old_table + new_table};
	else {
		unsigned int selected = sample_categorical(
				root_distribution.probabilities(),
				root_distribution.sum(), n.node.table_count);
		return {selected, old_table + new_table};
	}
}

template<bool Collapsed, typename RootType, typename Distribution, typename V>
inline sample_result<V> sample_table(
		const child_position<RootType>& n,
		const Distribution& root_distribution,
		const V& new_table_probability)
{
	V* probabilities = (V*) malloc(sizeof(V) * (n.node.table_count + 1));
	for (unsigned int j = 0; j < n.node.table_count; j++)
		probabilities[j] = n.node.table_sizes[j] / (n.node.alpha() + n.node.customer_count)
				* root_distribution.likelihood(n.node.root_assignment(j));

	sample_result<V> parent_result;
	if (n.node.parent == NULL) {
		parent_result = sample_table<Collapsed>(
  				make_root_node(n.root), root_distribution, new_table_probability);
	} else {
		parent_result = sample_table<Collapsed>(make_child_node(n, *n.node.parent),
				root_distribution, new_table_probability);
	}
	probabilities[n.node.table_count] = parent_result.new_table_probability
			* n.node.alpha() / (n.node.alpha() + n.node.customer_count);

	unsigned int selected = sample_categorical(probabilities, n.node.table_count + 1);
	if (selected == n.node.table_count)
		selected += parent_result.selected_table;
	V sum = probabilities[n.node.table_count];
	free(probabilities);
	return {selected, sum};
}


/**
 * Below are functions for adding new customers to existing or new tables.
 */

/* initializes a new table at the existing position 'index' */
template<bool Collapsed, typename RootType, typename Observations, typename Cache>
void replace_with_new_table(
	child_position<RootType>& n,
	unsigned int index,
	const Observations& observations,
	unsigned int selected_parent_table,
	Cache& cache)
{
	/* need to draw a new assignment for the parent table */
	unsigned int& new_table_assignment = n.node.table_assignments[index];
	if (n.node.parent != NULL) {
		auto parent = make_child_node(n, *n.node.parent);
		select_table<Collapsed>(parent, observations,
			selected_parent_table, new_table_assignment, cache);
		n.node.root_assignments[index] = parent.node.root_assignment(new_table_assignment);
	} else {
		auto root = make_root_node(n.root);
		select_table<Collapsed>(root, observations,
			selected_parent_table, new_table_assignment, cache);
		n.node.root_assignments[index] = new_table_assignment;
	}
}

template<bool Collapsed, typename RootType, typename Observations, typename Cache>
inline void assign_new_table(
	root_position<RootType>& n,
	const Observations& observations,
	unsigned int selected_parent_table,
	const Cache& cache)
{
	if (!Collapsed)
		fprintf(stderr, "assign_new_table ERROR: Unsupported operation.\n");
		//n.node.pi.sample(n.node.phi[n.node.table_count - 1]);
}

template<bool Collapsed, typename RootType, typename Observations, typename Cache>
inline void assign_new_table(
	child_position<RootType>& n,
	const Observations& observations,
	unsigned int selected_parent_table,
	Cache& cache)
{
	replace_with_new_table<Collapsed>(n,
		n.node.table_count - 1, observations, selected_parent_table, cache);
}

template<bool Collapsed, typename NodeType, typename Observations, typename Cache>
bool move_to_new_table(NodeType& n,
	const Observations& observations,
	unsigned int selected_parent_table,
	Cache& cache)
{
	/* create a new table, and add the observations to it */
	if (!n.node.add_to_new_table(observations, cache))
		return false;

	assign_new_table<Collapsed>(n, observations, selected_parent_table, cache);
	return true;
}

template<typename RootType, typename Observations, typename Cache>
bool add_to_table(root_position<RootType>& n,
	unsigned int dst, const Observations& observations, Cache& cache)
{
	return n.node.add_to_table(observations, dst, cache);
}

template<typename RootType, typename Observations, typename Cache>
bool add_to_table(child_position<RootType>& n,
	unsigned int dst, const Observations& observations, Cache& cache)
{
	if (!n.node.add_to_table(observations, dst, cache))
		return false;

	/* if the table becomes empty, "delete" it (remove_empty_tables will properly remove it) */
	if (n.node.parent == NULL) {
		auto root = make_root_node(n.root);
		return add_to_table(root, n.node.table_assignments[dst], observations, cache);
	} else {
		auto parent = make_child_node(n, *n.node.parent);
		return add_to_table(parent, n.node.table_assignments[dst], observations, cache);
	}
}

template<bool Collapsed, typename NodeType, typename Observations, typename Cache>
bool add_to_new_table(NodeType& n,
	const Observations& observations,
	unsigned int selected_parent_table,
	Cache& cache)
{
	/* create a new table, and add the observations to it */
	if (!n.node.add_to_new_table(observations, cache))
		return false;

	assign_new_table<Collapsed>(n, observations, selected_parent_table, cache);
	return true;
}

template<bool Collapsed, typename NodeType, typename Observations, typename Cache>
bool select_table(NodeType& n,
	const Observations& observations,
	unsigned int selected_table,
	unsigned int& new_table,
	Cache& cache)
{
	if (selected_table < n.node.table_count) {
		/* selected an existing table in this level */
		new_table = selected_table;
		n.node.table_sizes[selected_table]++;
		n.node.customer_count++;
		add_to_table(n, selected_table, observations, cache);
	} else {
		if (!n.node.new_table()) {
			fprintf(stderr, "select_table ERROR: Unable to create table.\n");
			return false;
		}
		new_table = n.node.table_count - 1;
		n.node.table_sizes[new_table] = 1;
		n.node.customer_count++;
		move_to_new_table<Collapsed>(n, observations,
				selected_table - (n.node.table_count - 1), cache);
	}
	return true;
}

template<bool Collapsed, typename NodeType, typename BaseDistribution, typename Observations, typename Cache>
inline bool sample_new_assignment(
	NodeType& n,
	const BaseDistribution& pi,
	const Observations& observations,
	unsigned int& assignment,
	Cache& cache)
{
	typedef typename NodeType::value_type V;

	/* compute likelihood of the observations for each root cluster */
	auto& root = n.get_root();
	auto root_distribution = cache.compute_root_distribution(root, observations);
	V new_table_probability = NodeType::root_type::data_distribution_type::log_probability(root.pi(), observations);

	sample_result<V> result = sample_table<Collapsed>(n, root_distribution, new_table_probability);
	cache.on_sample_table(root, root_distribution, root.table_count, observations);
	cache.on_add(observations);
	root_distribution.free();

	/* select the new table assignment */
	if (result.selected_table < n.node.table_count) {
		/* we chose a table at this level */
		increment_table<true>(n.node, result.selected_table, cache);
		assignment = result.selected_table;
		add_to_table(n, result.selected_table, observations, cache);
	} else {
		/* we chose a table at a higher level */
		unsigned int new_table = n.node.table_count;
		if (!n.node.new_table()) {
			fprintf(stderr, "sample_assignment ERROR: Unable to create table.\n");
			return false;
		}
		n.node.table_sizes[new_table] = 1;
		n.node.customer_count++;
		assignment = new_table;
		move_to_new_table<Collapsed>(n, observations,
				result.selected_table - (n.node.table_count - 1), cache);
	}
	return true;
}

template<typename RootType>
inline node_sampler<typename RootType::atom_type, typename RootType::value_type>*
get_node_pointer(child_position<RootType>& child) {
	return &child.node;
}

template<typename RootType>
inline node_sampler<typename RootType::atom_type, typename RootType::value_type>*
get_node_pointer(root_position<RootType>& child) {
	return NULL;
}

template<bool Collapsed = true, typename NodeType, typename BaseDistribution, typename Cache>
bool add(NodeType n, const BaseDistribution& pi,
	const typename NodeType::value_type* alpha,
	const unsigned int* path, unsigned int length,
	const typename NodeType::atom_type& observation, Cache& cache)
{
	typedef typename NodeType::atom_type K;
	typedef typename NodeType::value_type V;

	if (length == 0) {
		size_t old_capacity = n.node.n->observations.capacity;
		if (!n.node.n->observations.add(observation)) {
			fprintf(stderr, "add ERROR: Unable to add new observation.\n");
			return false;
		} else if (old_capacity < n.node.n->observations.capacity) {
			if (!resize(n.node.observation_assignments, n.node.n->observations.capacity)) {
				free(n.node.n->observations.last());
				n.node.n->observations.length--;
				return false;
			}
		}

		return sample_new_assignment<Collapsed>(n, pi, observation,
			n.node.observation_assignments[n.node.n->observations.length - 1], cache);
	}

	size_t old_capacity = n.node.n->children.capacity;
	if (!n.node.n->children.ensure_capacity((unsigned int) n.node.n->children.size + 1)) {
		fprintf(stderr, "add ERROR: Unable to expand children map.\n");
		return false;
	} else if (old_capacity < n.node.n->children.capacity) {
		if (!resize(n.node.children, n.node.n->children.capacity))
			return false;
		/* pointers to parent and hdp nodes could be invalidated here, so fix them */
		for (unsigned int i = 0; i < n.node.n->children.size; i++) {
			n.node.children[i].n = &n.node.n->children.values[i];
			fix_parent_pointers(n.node.children[i]);
		}
	}

	unsigned int index = strict_linear_search(n.node.n->children.keys, *path, 0, (unsigned int) n.node.n->children.size);
	if (index > 0 && n.node.n->children.keys[index - 1] == *path) {
		/* a child node with this key already exists, so recurse into it */
		return add(make_child_node(n, n.node.children[index - 1]), pi, alpha + 1, path + 1, length - 1, observation, cache);
	} else {
		/* there is no child node with the given key, so create it */
		shift_right(n.node.n->children.keys, (unsigned int) n.node.n->children.size, index);
		shift_right(n.node.n->children.values, (unsigned int) n.node.n->children.size, index);
		node<K, V>& child = n.node.n->children.values[index];
		if (!init(child, *alpha, 1, 8)) {
			fprintf(stderr, "add ERROR: Error creating new child.\n");
			return false;
		}
		shift_right(n.node.children, n.node.child_count(), index);
		n.node.n->children.keys[index] = *path;
		n.node.n->children.size++;
		/* TODO: this insertion algorithm can be made more efficient */
		if (!init(n.node.children[index], n.get_root(), child, get_node_pointer(n), 4)) {
			free(child);
			return false;
		}
		/* correct the node pointers */
		for (unsigned int i = 0; i < n.node.child_count(); i++) {
			n.node.children[i].n = &n.node.n->children.values[i];
			fix_parent_pointers(n.node.children[i]);
		}
		return add(make_child_node(n, n.node.children[index]), pi, alpha + 1, path + 1, length - 1, observation, cache);
	}
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V, typename Cache>
inline bool add(hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
		const unsigned int* path, unsigned int length, const K& observation, Cache& cache)
{
	return add(make_root_node(h), h.n->pi, h.n->alpha + 1, path, length, observation, cache);
}

template<typename RootType, typename Observations, typename Cache>
bool remove_from_table(RootType& n,
	RootType& root, unsigned int table,
	const Observations& observations, Cache& cache)
{
	return n.remove_from_table(observations, table, cache);
}

template<typename RootType, typename Observations, typename Cache>
bool remove_from_table(
	typename RootType::child_type& n,
	RootType& root, unsigned int table,
	const Observations& observations, Cache& cache)
{
	if (!n.remove_from_table(observations, table, cache))
		return false;

	/* if the table becomes empty, "delete" it (remove_empty_tables will properly remove it) */
	if (n.parent == NULL) {
		if (n.descendant_observations[table].counts.size == 0)
			decrement_table<true>(root, n.table_assignments[table], cache);
		return remove_from_table(root, root, n.table_assignments[table], observations, cache);
	} else {
		if (n.descendant_observations[table].counts.size == 0)
			decrement_table<true>(*n.parent, n.table_assignments[table], cache);
		return remove_from_table(*n.parent, root, n.table_assignments[table], observations, cache);
	}
}

template<typename NodeType, typename RootType, typename Cache>
bool remove(NodeType& n, RootType& root,
	const unsigned int* path, unsigned int length,
	const typename RootType::atom_type& observation, Cache& cache)
{
	if (length == 0) {
		unsigned int index = n.n->observations.index_of(observation);
		unsigned int last = (unsigned int) (n.observation_count() - 1);
		if (index == n.observation_count()) {
			fprintf(stderr, "remove WARNING: Observation does not exist at this node.\n");
			return true;
		}

		unsigned int table = n.observation_assignments[index];
		decrement_table<true>(n, table, cache);
		cache.on_remove(observation);
		remove_from_table(n, root, table, observation, cache);
		if (index + 1 < n.observation_count()) {
			core::swap(n.n->observations[index], n.n->observations[last]);
			core::swap(n.observation_assignments[index], n.observation_assignments[last]);
		}
		free(n.n->observations.last());
		n.n->observations.length--;
		return true;
	}

	unsigned int index = n.n->children.index_of(*path);
	if (index == n.n->children.size) {
		fprintf(stderr, "remove WARNING: Given path does not exist in HDP tree.\n");
		return true;
	}
	return remove(n.children[index], root, path + 1, length - 1, observation, cache);
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V, typename Cache>
inline bool remove(hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
		const unsigned int* path, unsigned int length, const K& observation, Cache& cache)
{
	return remove(h, h, path, length, observation, cache);
}


/**
 * Below are functions for moving existing customers to existing or new tables.
 */

template<typename RootType, typename Observations, typename Cache>
bool move_to_table(root_position<RootType>& n,
	unsigned int src, unsigned int dst,
	const Observations& observations, Cache& cache)
{
	if (src == dst) return true;
	return n.node.move_to_table(observations, src, dst, cache);
}

template<typename RootType, typename Observations, typename Cache>
bool move_to_table(child_position<RootType>& n,
	unsigned int src, unsigned int dst,
	const Observations& observations, Cache& cache)
{
	if (src == dst) return true;
	if (!n.node.move_to_table(observations, src, dst, cache))
		return false;

	/* if the table becomes empty, "delete" it (remove_empty_tables will properly remove it) */
	bool success;
	if (n.node.parent == NULL) {
		auto root = make_root_node(n.root);
		if (n.node.descendant_observations[src].counts.size == 0)
			decrement_table<true>(n.root, n.node.table_assignments[src], cache);
		success = move_to_table(root, n.node.table_assignments[src],
				n.node.table_assignments[dst], observations, cache);
	} else {
		auto parent = make_child_node(n, *n.node.parent);
		if (n.node.descendant_observations[src].counts.size == 0)
			decrement_table<true>(parent.node, n.node.table_assignments[src], cache);
		success = move_to_table(parent, n.node.table_assignments[src],
				n.node.table_assignments[dst], observations, cache);
	}

#if !defined(NDEBUG)
	/* for debugging, check the consistency of the table_sizes data structure */
	unsigned int* expected_sizes = (unsigned int*) calloc(n.node.table_count, sizeof(unsigned int));
	compute_table_sizes(n.node, expected_sizes, n.node.table_count);
	if (!compare_histograms(expected_sizes, n.node.table_sizes, n.node.table_count, n.node.table_count))
		fprintf(stderr, "move_to_table WARNING: table_sizes is inconsistent.\n");
	free(expected_sizes);
#endif

	return success;
}

/* initializes a new table at the existing position 'index' */
template<bool RemoveOldTable, bool Collapsed,
	typename RootType, typename Observations, typename Cache>
void replace_with_new_table(
	root_position<RootType>& n,
	unsigned int old_table, unsigned int index,
	const Observations& observations,
	unsigned int selected_parent_table,
	Cache& cache)
{
	if (!Collapsed)
		fprintf(stderr, "assign_new_table ERROR: Unsupported operation.\n");
		//n.node.pi.sample(n.node.phi[index]);
}

/* initializes a new table at the existing position 'index' */
template<bool RemoveOldTable, bool Collapsed,
	typename RootType, typename Observations, typename Cache>
void replace_with_new_table(
	child_position<RootType>& n,
	unsigned int old_table, unsigned int index,
	const Observations& observations,
	unsigned int selected_parent_table,
	Cache& cache)
{
	/* need to draw a new assignment for the parent table */
	unsigned int& new_table_assignment = n.node.table_assignments[index];
	unsigned int old_parent_table = n.node.table_assignments[old_table];
	if (n.node.parent != NULL) {
		auto parent = make_child_node(n, *n.node.parent);
		if (RemoveOldTable)
			decrement_table<true>(parent.node, old_parent_table, cache);
		select_table<Collapsed>(parent, observations, old_parent_table,
				selected_parent_table, new_table_assignment, cache);
		n.node.root_assignments[index] = parent.node.root_assignment(new_table_assignment);
	} else {
		auto root = make_root_node(n.root);
		if (RemoveOldTable)
			decrement_table<true>(n.root, old_parent_table, cache);
		select_table<Collapsed>(root, observations, old_parent_table,
				selected_parent_table, new_table_assignment, cache);
		n.node.root_assignments[index] = new_table_assignment;
	}
}

template<bool Collapsed, typename RootType, typename Observations, typename Cache>
inline void assign_new_table(
	root_position<RootType>& n,
	unsigned int src,
	const Observations& observations,
	unsigned int selected_parent_table,
	const Cache& cache)
{
	if (!Collapsed)
		fprintf(stderr, "assign_new_table ERROR: Unsupported operation.\n");
		//n.node.pi.sample(n.node.phi[n.node.table_count - 1]);
}

template<bool Collapsed, typename RootType, typename Observations, typename Cache>
inline void assign_new_table(
	child_position<RootType>& n,
	unsigned int src,
	const Observations& observations,
	unsigned int selected_parent_table,
	Cache& cache)
{
	if (n.node.descendant_observations[src].counts.size != 0)
		replace_with_new_table<false, Collapsed>(n, src,
				n.node.table_count - 1, observations, selected_parent_table, cache);
	else replace_with_new_table<true, Collapsed>(n, src,
			n.node.table_count - 1, observations, selected_parent_table, cache);
}

template<bool Collapsed, typename NodeType, typename Observations, typename Cache>
bool move_to_new_table(
	NodeType& n, unsigned int src,
	const Observations& observations,
	unsigned int selected_parent_table,
	Cache& cache)
{
	/* create a new table, and add the observations to it */
	if (!n.node.move_to_new_table(observations, src, cache))
		return false;

	assign_new_table<Collapsed>(n, src, observations, selected_parent_table, cache);
	return true;
}

template<bool Collapsed, typename NodeType, typename Observations, typename Cache>
bool select_table(NodeType& n,
	const Observations& observations,
	unsigned int old_table,
	unsigned int selected_table,
	unsigned int& new_table,
	Cache& cache)
{
	if (selected_table < n.node.table_count) {
		/* selected an existing table in this level */
		new_table = selected_table;
		n.node.table_sizes[selected_table]++;
		n.node.customer_count++;
		move_to_table(n, old_table, selected_table, observations, cache);
	} else {
		if (!n.node.new_table()) {
			fprintf(stderr, "select_table ERROR: Unable to create table.\n");
			return false;
		}
		new_table = n.node.table_count - 1;
		n.node.table_sizes[new_table] = 1;
		n.node.customer_count++;
		move_to_new_table<Collapsed>(n, old_table, observations,
				selected_table - (n.node.table_count - 1), cache);
	}

#if !defined(NDEBUG)
	/* for debugging, check the consistency of the table_sizes data structure */
	unsigned int* expected_sizes = (unsigned int*) calloc(n.node.table_count, sizeof(unsigned int));
	compute_table_sizes(n.node, expected_sizes, n.node.table_count);
	if (!compare_histograms(expected_sizes, n.node.table_sizes, n.node.table_count, n.node.table_count))
		fprintf(stderr, "select_table WARNING: table_sizes is inconsistent.\n");
	free(expected_sizes);
#endif

	return true;
}

template<bool Collapsed, typename NodeType, typename Observations, typename Cache>
void sample_assignment(NodeType& n,
	const Observations& observations,
	unsigned int& assignment,
	unsigned int old_root_assignment,
	Cache& cache)
{
#if !defined(NDEBUG)
	if (assignment >= n.node.table_count || n.node.table_sizes[assignment] == 0) {
		fprintf(stderr, "sample_assignment ERROR: Invalid table assignment histogram.\n");
		return;
	}
#endif

	typedef typename NodeType::root_type::value_type V;

	decrement_table<false>(n.node, assignment, cache);

	/* compute likelihood of the observations for each root cluster */
	auto& root = n.get_root();
	auto root_distribution = cache.compute_root_distribution(root, observations, old_root_assignment);
	V new_table_probability = NodeType::root_type::data_distribution_type::log_probability(root.pi(), observations);

	sample_result<V> result = sample_table<Collapsed>(n, root_distribution, new_table_probability);
	cache.on_sample_table(root, root_distribution, old_root_assignment, observations);

	/* select the new table assignment */
	if (result.selected_table < n.node.table_count) {
		/* we chose a table at this level */
		increment_table<false>(n.node, result.selected_table, cache);
		unsigned int old_table = assignment;
		assignment = result.selected_table;
		move_to_table(n, old_table, result.selected_table, observations, cache);
	} else {
		/* we chose a table at a higher level */
		if (n.node.table_sizes[assignment] == 0) {
			increment_table<false>(n.node, assignment, cache);
			replace_with_new_table<true, Collapsed>(n, assignment, assignment,
					observations, result.selected_table - n.node.table_count, cache);
		} else {
			unsigned int new_table = n.node.table_count;
			if (!n.node.new_table()) {
				fprintf(stderr, "sample_assignment ERROR: Unable to create table.\n");
				return;
			}
			n.node.table_sizes[new_table] = 1;
			unsigned int old_table = assignment;
			assignment = new_table;

			move_to_new_table<Collapsed>(n, old_table, observations,
					result.selected_table - (n.node.table_count - 1), cache);
		}
	}

	cache.on_finish_sampling(root, root_distribution, old_root_assignment);
	root_distribution.free();

#if !defined(NDEBUG)
	/* for debugging, check the consistency of this data structure */
	if (!tables_sorted(n.get_root()))
		fprintf(stderr, "sample_assignment WARNING: Table array_histograms are not sorted.\n");
	/*if (!check_tables(n.get_root()))
		fprintf(stderr, "sample_assignment WARNING: The HDP structure is invalid.\n");
	if (!cache.is_valid(root))
		fprintf(stderr, "sample_assignment WARNING: Table distribution cache is invalid.\n");*/

	unsigned int computed_sum = 0;
	for (unsigned int i = 0; i < n.node.table_count; i++)
		computed_sum += n.node.table_sizes[i];
	if (computed_sum != n.node.customer_count) {
		fprintf(stderr, "sample_assignment WARNING: customer_count is incorrect.\n");
		n.node.customer_count = computed_sum;
	}
#endif
}

template<bool Collapsed, typename NodeType, typename Cache>
void sample_nodes(NodeType& n, unsigned int depth, Cache& cache)
{
	/* visit each child node in random order */
	unsigned int* child_order = (unsigned int*) alloca(sizeof(unsigned int) * n.node.child_count());
	for (unsigned int i = 0; i < n.node.child_count(); i++)
		child_order[i] = i;
	if (n.node.child_count() > 1) shuffle(child_order, (unsigned int) n.node.child_count());

	for (unsigned int i = 0; i < n.node.child_count(); i++) {
		/* visit nodes in prefix order, so recurse first */
		auto& child = n.node.children[child_order[i]];
		auto child_node = make_child_node(n, child);
		sample_nodes<Collapsed>(child_node, depth + 1, cache);

		/* visit each table in the child node in random order */
		unsigned int* order = (unsigned int*) alloca(sizeof(unsigned int) * child.table_count);
		for (unsigned int j = 0; j < child.table_count; j++)
			order[j] = j;
		if (child.table_count > 1) shuffle(order, child.table_count);

		/* sample the assignment for every table at this node */
		for (unsigned int j = 0; j < child.table_count; j++) {
			unsigned int index = order[j];
			if (child.descendant_observations[index].counts.size == 0)
				continue;

			auto& observations = child.descendant_observations[index];
			if (observations.counts.size == 1 && observations.counts.values[0] == 1) {
				sample_assignment<Collapsed>(n, observations.counts.keys[0],
						child.table_assignments[index], child.root_assignments[index], cache);
			} else {
				sample_assignment<Collapsed>(n, observations,
						child.table_assignments[index], child.root_assignments[index], cache);
			}
		}
	}

	/* visit each observation in this node in random order */
	unsigned int* order = (unsigned int*) malloc(sizeof(unsigned int) * n.node.observation_count());
	for (unsigned int i = 0; i < n.node.observation_count(); i++)
		order[i] = i;
	if (n.node.observation_count() > 1)
		shuffle(order, (unsigned int) n.node.observation_count());

	/* sample the table assignment for each observation at this node */
	for (unsigned int i = 0; i < n.node.observation_count(); i++) {
		unsigned int& table_assignment = n.node.observation_assignments[order[i]];
		sample_assignment<Collapsed>(
			n, n.node.get_observation(order[i]), table_assignment,
			n.node.root_assignment(table_assignment), cache);
	}
	free(order);
}

template<bool Collapsed, typename BaseDistribution,
	typename DataDistribution, typename K, typename V>
inline void sample_hdp(
	hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	cache<BaseDistribution, DataDistribution, K, V>& cache)
{
	auto root = make_root_node(h);
	sample_nodes<Collapsed>(root, 1, cache);

	cache.on_finished_sampling_hdp();
	remove_empty_tables(h, cache);
	recompute_root_assignments(h);
}


/**
 * Auxiliary variable sampler for HDP hyperparameters.
 * (Teh et al. 2006 appendix A, Escobar and West 1995 section 6)
 */

template<typename K, typename V>
inline void set_alpha(node_sampler<K, V>& n, const V* alpha) {
	n.n->alpha = *alpha;
	n.n->log_alpha = log(*alpha);

	for (unsigned int i = 0; i < n.child_count(); i++)
		set_alpha(n.children[i], alpha + 1);
}

template<typename NodeType>
inline void compute_auxiliary(const NodeType& n,
	unsigned int& s_sum, typename NodeType::value_type& log_w_sum)
{
	if (n.customer_count > 0) {
		s_sum += n.table_count - sample_bernoulli(n.customer_count / (n.customer_count + n.alpha()));
		log_w_sum += log(sample_beta(n.alpha() + 1, (typename NodeType::value_type) n.customer_count));
	}
}

template<typename K, typename V>
inline void sample_alpha_each_node(
	node_sampler<K, V>& n,
	const V* a, const V* b)
{
	V s_sum = 0.0, log_w_sum = 0.0;
	compute_auxiliary(n, s_sum, log_w_sum);
	n.n->alpha = sample_gamma(*a + s_sum, *b - log_w_sum);
	n.n->log_alpha = log(n.alpha());

	for (unsigned int i = 0; i < n.child_count(); i++)
		sample_alpha(n.children[i], a + 1, b + 1);
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
inline void sample_alpha_each_node(
	hdp_sampler<BaseDistribution, DataDistribution, K, V>& n,
	const V* a, const V* b)
{
	V s_sum = 0.0, log_w_sum = 0.0;
	compute_auxiliary(n, s_sum, log_w_sum);
	n.n->alpha[0] = sample_gamma(*a + s_sum, *b - log_w_sum);
	n.n->log_alpha = log(n.alpha());

	for (unsigned int i = 0; i < n.child_count(); i++)
		sample_alpha_each_node(n.children[i], a + 1, b + 1);
}

template<typename NodeType>
inline void sample_alpha(NodeType& n,
	unsigned int* s_sum,
	typename NodeType::value_type* log_w_sum)
{
	compute_auxiliary(n, *s_sum, *log_w_sum);

	for (unsigned int i = 0; i < n.child_count(); i++)
		sample_alpha(n.children[i], s_sum + 1, log_w_sum + 1);
}

/* this sampling algorithm for alpha assumes that alpha is
   the same across all nodes at each level in the HDP */
template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
inline bool sample_alpha_each_level(
	hdp_sampler<BaseDistribution, DataDistribution, K, V>& n,
	const V* a, const V* b)
{
	unsigned int* s_sum = (unsigned int*) alloca(sizeof(unsigned int) * n.n->depth);
	V* log_w_sum = (V*) alloca(sizeof(V) * n.n->depth);
	if (s_sum == NULL || log_w_sum == NULL) {
		fprintf(stderr, "sample_alpha ERROR: Out of memory.\n");
		return false;
	}
	for (unsigned int i = 0; i < n.n->depth; i++) {
		s_sum[i] = 0;
		log_w_sum[i] = 0.0;
	}

	sample_alpha(n, s_sum, log_w_sum);

	for (unsigned int i = 0; i < n.n->depth; i++) {
		n.n->alpha[i] = sample_gamma(a[i] + s_sum[i], b[i] - log_w_sum[i]);
		if (n.n->alpha[i] == 0.0)
			n.n->alpha[i] = 1.0e-16; /* bad things happen if this is zero */
	}
	n.n->log_alpha = log(n.alpha());
	for (unsigned int i = 0; i < n.child_count(); i++)
		set_alpha(n.children[i], n.n->alpha + 1);
	return true;
}


/**
 * Logic for computing the probability of a path given an observation.
 */

template<typename V>
struct hdp_path {
	unsigned int* path;
	V probability;

	unsigned int* excluded;
	unsigned int excluded_count;

	hdp_path(unsigned int length, unsigned int excluded_capacity) : excluded_count(0) {
		if (!initialize(length, excluded_capacity))
			exit(EXIT_FAILURE);
	}

	~hdp_path() { free(); }

	inline unsigned int get_feature(unsigned int index) const {
		return path[index];
	}

	inline void set_feature(unsigned int index, unsigned int feature) {
		path[index] = feature;
	}

	inline V get_probability() const {
		return probability;
	}

	inline void set_probability(const V& item) {
		probability = item;
	}

	inline void add_excluded_unsorted(unsigned int item) {
		excluded[excluded_count] = item;
		excluded_count++;
	}

	static inline void swap(hdp_path<V>& first, hdp_path<V>& second) {
		core::swap(first.path, second.path);
		core::swap(first.probability, second.probability);
		core::swap(first.excluded, second.excluded);
		core::swap(first.excluded_count, second.excluded_count);
	}

	static inline void move(const hdp_path<V>& first, hdp_path<V>& second) {
		second.path = first.path;
		second.probability = first.probability;
		second.excluded = first.excluded;
		second.excluded_count = first.excluded_count;
	}

	static inline void free(hdp_path<V>& path) { path.free(); }

private:
	inline bool initialize(unsigned int length, unsigned int excluded_capacity) {
		path = (unsigned int*) malloc(sizeof(unsigned int) * length);
		if (path == NULL) {
			fprintf(stderr, "hdp_path.initialize ERROR: Out of memory.\n");
			return false;
		}
		excluded = (unsigned int*) malloc(std::max((size_t) 1, sizeof(unsigned int) * excluded_capacity));
		if (excluded == NULL) {
			fprintf(stderr, "hdp_path.initialize ERROR: Out of memory.\n");
			return false;
		}
		return true;
	}

	inline void free() {
		core::free(path);
		core::free(excluded);
	}

	template<typename A>
	friend bool init(hdp_path<A>&, unsigned int, unsigned int);
};

template<typename V>
inline bool init(hdp_path<V>& path, unsigned int length, unsigned int excluded_capacity) {
	path.excluded_count = 0;
	return path.initialize(length, excluded_capacity);
}

template<typename V>
inline V log_probability(const hdp_path<V>& path) {
	return path.probability;
}

template<typename V>
struct hdp_path_sorter {
	unsigned int depth;

	hdp_path_sorter(unsigned int depth) : depth(depth) { }
};

template<typename V>
inline bool less_than(
	const hdp_path<V>& first,
	const hdp_path<V>& second,
	const hdp_path_sorter<V>& sorter)
{
	for (unsigned int i = 0; i < sorter.depth - 1; i++) {
		if (first.get_feature(i) < second.get_feature(i))
			return true;
		else if (first.get_feature(i) > second.get_feature(i))
			return false;
	}
	return false;
}

template<typename K, typename V>
struct hdp_search_state {
	unsigned int* path;
	V* probabilities;
	V maximum;

	const node_sampler<K, V>* n;
	unsigned int level;

	static inline void free(hdp_search_state<K, V>& state) {
		core::free(state.path);
		core::free(state.probabilities);
	}
};

template<typename K, typename V>
inline bool operator < (const hdp_search_state<K, V>& first, const hdp_search_state<K, V>& second)
{
	return first.maximum < second.maximum;
}

template<typename V, typename NodeSample>
inline V compute_probability(const NodeSample& sample,
		const V* root_probabilities, const V& prior, const V& alpha)
{
	V probability = 0.0;
	for (unsigned int i = 0; i < sample.table_count; i++)
		probability += sample.table_sizes[i] * root_probabilities[sample.root_assignments[i]];
	probability += alpha * prior;
	return probability / (sample.customer_count + alpha);
}

template<typename V, typename RootSample>
inline V compute_root_probability(const RootSample& sample,
		const V* root_probabilities, const V& prior, const V& alpha)
{
	V probability = 0.0;
	for (unsigned int i = 0; i < sample.table_count; i++)
		probability += sample.table_sizes[i] * root_probabilities[i];
	probability += alpha * prior;
	return probability / (sample.customer_count + alpha);
}

template<typename V>
inline V sum(const V* values, unsigned int length) {
	V sum = 0.0;
	for (unsigned int i = 0; i < length; i++)
		sum += values[i];
	return sum;
}

template<typename NodeType, typename V, typename FeatureSequence>
inline void complete_path(
	const NodeType& node, const V* probabilities, unsigned int label,
	unsigned int sample_count, const unsigned int* path,
	const unsigned int* const* excluded, const unsigned int* excluded_counts,
	unsigned int level, unsigned int depth, array<FeatureSequence>& x)
{
	unsigned int max_excluded_count = 0;
	if (label == IMPLICIT_NODE)
		max_excluded_count = node.child_count() + excluded_counts[level];

	double probability = sum(probabilities, sample_count) / sample_count;
	if (probability == 0.0) {
		return;
	} else if (!x.ensure_capacity(x.length + 1)) {
		fprintf(stderr, "complete_path ERROR: Unable to expand x array.\n");
		return;
	} else if (!init(x[(unsigned int) x.length], depth - 1)) {
		fprintf(stderr, "complete_path ERROR: Unable to initialize feature sequence.\n");
		return;
	}
	FeatureSequence& completed = x[(unsigned int) x.length];
	x.length++;

	for (unsigned int i = 0; i < depth - 1; i++) {
		if (i == level) {
			completed.set_feature(i, label);
		} else {
			completed.set_feature(i, path[i]);
			if (path[i] == IMPLICIT_NODE && !completed.set_excluded(i, excluded[i], excluded_counts[i])) {
				free(completed);
				return;
			}
		}
	}

	completed.set_probability(probability);

	/* compute the union of the set of child node ids and the excluded set */
	if (max_excluded_count == 0) return;
	completed.ensure_excluded_capacity(level, max_excluded_count);
	auto do_union = [&](unsigned int child_id, unsigned int i, unsigned int j) {
		completed.exclude_unsorted(level, child_id);
	};
	set_union(do_union, do_union, do_union,
			node.n->children.keys, node.child_count(),
			excluded[level], excluded_counts[level]);
	completed.sort_excluded(level);
}

template<typename K, typename V, typename FeatureSequence>
inline void complete_path(const node_sampler<K, V>& leaf,
	const V* const* root_probabilities, const unsigned int* path,
	const V* probabilities, unsigned int new_key,
	unsigned int level, unsigned int depth, array<FeatureSequence>& x)
{
#if !defined(NDEBUG)
	if (level + 1 != depth - 1)
		fprintf(stderr, "complete_path WARNING: This function should only be called at the leaves.\n");
#endif
	
	V probability = 0.0;
	for (unsigned int i = 0; i < leaf.posterior.length; i++) {
		probability += compute_probability(leaf.posterior[i],
				root_probabilities[i], probabilities[i], leaf.alpha());
	}
	probability /= leaf.posterior.length;

	if (probability == 0.0) {
		return;
	} else if (!x.ensure_capacity(x.length + 1)) {
		fprintf(stderr, "complete_path ERROR: Unable to expand x array.\n");
		return;
	} else if (!init(x[(unsigned int) x.length], depth - 1)) {
		fprintf(stderr, "complete_path ERROR: Unable to initialize feature sequence.\n");
		return;
	}
	FeatureSequence& completed = x[(unsigned int) x.length];
	x.length++;

	for (unsigned int i = 0; i < level; i++)
		completed.set_feature(i, path[i]);
	completed.set_feature(level, new_key);

	completed.set_probability(probability);
}

template<typename NodeType, typename V>
inline V compute_maximum(const NodeType& n,
	const V* const* root_probabilities, const V* probabilities)
{
	V maximum = 0.0;
	for (unsigned int i = 0; i < n.posterior.length; i++) {
		V maximum_i = probabilities[i];
		const V* root = root_probabilities[i];
		auto& sample = n.posterior[i];
		for (unsigned int j = 0; j < sample.table_count; j++)
			if (sample.table_sizes[j] > 0 && root[sample.root_assignment(j)] > maximum_i)
				maximum_i = root[sample.root_assignment(j)];
		maximum += maximum_i;
	}
	return maximum;
}

template<typename NodeType, typename V>
inline V compute_maximum(const NodeType& n,
		const V* const* root_probabilities, const V* probabilities,
		const unsigned int* excluded, unsigned int excluded_count)
{
	if (excluded_count == 0)
		return compute_maximum(n, root_probabilities, probabilities);

	V maximum = 0.0;
	unsigned int i = 0, j = 0;
	while (i < excluded_count && j < n.children.size) {
		if (excluded[i] == n.children.keys[j]) {
			/* this child is excluded */
			i++; j++;
		} else if (excluded[i] < n.children.keys[j]) {
			i++;
		} else {
			V new_maximum = compute_maximum(n.children.values[j], root_probabilities, probabilities);
			if (new_maximum > maximum) maximum = new_maximum;
			j++;
		}
	}

	while (j < n.children.size) {
		V new_maximum = compute_maximum(n.children.values[j], root_probabilities, probabilities);
		if (new_maximum > maximum) maximum = new_maximum;
		j++;
	}
	return maximum;
}

template<typename K, typename V, typename FeatureSequence>
void push_node_state(const node_sampler<K, V>& child,
	const V* const* root_probabilities,
	const unsigned int* path, const V* probabilities,
	unsigned int new_key, unsigned int level, unsigned int depth,
	array<hdp_search_state<K, V>>& queue, array<FeatureSequence>& x)
{
	if (level + 1 == depth - 1) {
		/* this is the last level */
		complete_path(child, root_probabilities, path,
				probabilities, new_key, level, depth, x);
		return;
	}

#if !defined(NDEBUG)
	/* we expect the caller to ensure there is enough capacity */
	if (queue.length == queue.capacity) {
		fprintf(stderr, "push_node_state ERROR: No capacity in the queue.\n");
		return;
	}
#endif
	hdp_search_state<K, V>& state = queue[(unsigned int) queue.length];

	state.probabilities = (V*) malloc(sizeof(V) * child.posterior.length);
	if (state.probabilities == NULL) {
		fprintf(stderr, "push_node_state ERROR: Out of memory.\n");
		return;
	}
	state.path = (unsigned int*) malloc((depth - 1) * sizeof(unsigned int));
	if (state.path == NULL) {
		fprintf(stderr, "push_node_state ERROR: Out of memory.\n");
		free(state.probabilities);
		return;
	}
	for (unsigned int i = 0; i < level; i++)
		state.path[i] = path[i];
	state.path[level] = new_key;
	for (unsigned int i = level + 1; i + 1 < depth; i++)
		state.path[i] = path[i];
	state.level = level + 1;
	state.n = &child;

	for (unsigned int i = 0; i < child.posterior.length; i++) {
		state.probabilities[i] = compute_probability(child.posterior[i],
				root_probabilities[i], probabilities[i], child.alpha());
	}
	state.maximum = compute_maximum(child, root_probabilities, state.probabilities);

	queue.length++;
	std::push_heap(queue.data, queue.data + queue.length);
}

template<typename NodeType, typename K, typename V, typename FeatureSequence>
void process_search_state(const NodeType& n,
	const V* const* root_probabilities, const unsigned int* path,
	const unsigned int* const* excluded, const unsigned int* excluded_count,
	const V* probabilities, unsigned int level, unsigned int depth,
	array<hdp_search_state<K, V>>& queue, array<FeatureSequence>& x)
{
	bool contains;
	if (path[level] == IMPLICIT_NODE) {
		/* compute the score for a descendant that is not explicitly stored in the tree */
		complete_path(n, probabilities, IMPLICIT_NODE,
			(unsigned int) n.posterior.length, path, excluded, excluded_count, level, depth, x);

		if (!queue.ensure_capacity(queue.length + n.child_count())) {
			fprintf(stderr, "process_search_state ERROR: Unable to expand search queue.\n");
			return;
		}

		/* consider all child nodes not in the 'excluded' set */
		unsigned int i = 0, j = 0;
		while (i < n.child_count() && j < excluded_count[level]) {
			unsigned int child_id = n.child_key(i);
			if (child_id == excluded[level][j]) {
				i++; j++; continue;
			} else if (child_id < excluded[level][j]) {
				push_node_state(n.children[i], root_probabilities,
					path, probabilities, n.child_key(i), level, depth, queue, x);
				i++;
			} else {
				j++;
			}
		}

		while (i < n.child_count()) {
			push_node_state(n.children[i], root_probabilities, path,
					probabilities, n.child_key(i), level, depth, queue, x);
			i++;
		}
	} else if (path[level] == UNION_NODE) {
		fprintf(stderr, "process_search_state ERROR: Support for UNION_NODE is unimplemented.");
		exit(EXIT_FAILURE);
		/* TODO: implement this; it should look something like below */
		/*for (unsigned int i = 0; i < excluded_count; i++) {
			auto& child = n.get_child(excluded[i], contains);
			if (contains) {
				push_node_state(child, root_probabilities, path,
						probabilities, excluded[i], level, depth, queue, x);
			} else {
				complete_path(n, probabilities, excluded[i],
					(unsigned int) n.posterior.length, path, NULL, 0, level, depth, x);
			}
		}*/
	} else {
		auto& child = n.get_child(path[level], contains);
		if (contains) {
			push_node_state(child, root_probabilities, path,
					probabilities, path[level], level, depth, queue, x);
		} else {
			complete_path(n, probabilities, path[level],
				(unsigned int) n.posterior.length, path, NULL, 0, level, depth, x);
		}
	}
}

template<typename V>
inline void cleanup_root_probabilities(V** root_probabilities, unsigned int row_count)
{
	for (unsigned int i = 0; i < row_count; i++)
		free(root_probabilities[i]);
	free(root_probabilities);
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
V** copy_root_probabilities(const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h, const V* const* src)
{
	V** root_probabilities = (V**) malloc(sizeof(V*) * h.posterior.length);
	if (root_probabilities == NULL) {
		fprintf(stderr, "copy_root_probabilities ERROR: Out of memory.\n");
		return NULL;
	}

	for (unsigned int i = 0; i < h.posterior.length; i++) {
		root_probabilities[i] = (V*) malloc(sizeof(V) * h.posterior[i].table_count);
		if (root_probabilities[i] == NULL) {
			fprintf(stderr, "predict ERROR: Insufficient memory for root_probabilities[%u].\n", i);
			cleanup_root_probabilities(root_probabilities, i);
			return NULL;
		}
		memcpy(root_probabilities[i], src[i], sizeof(V) * h.posterior[i].table_count);
	}

	return root_probabilities;
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
V** copy_root_probabilities(
		const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
		const V* const* src, unsigned int observation_count)
{
	V** root_probabilities = (V**) malloc(sizeof(V*) * observation_count);
	if (root_probabilities == NULL) {
		fprintf(stderr, "copy_root_probabilities ERROR: Out of memory.\n");
		return NULL;
	}

	for (unsigned int i = 0; i < observation_count; i++) {
		root_probabilities[i] = (V*) malloc(sizeof(V) * h.table_count);
		if (root_probabilities[i] == NULL) {
			fprintf(stderr, "predict ERROR: Insufficient memory for root_probabilities[%u].\n", i);
			cleanup_root_probabilities(root_probabilities, i);
			return NULL;
		}
		memcpy(root_probabilities[i], src[i], sizeof(V) * h.table_count);
	}

	return root_probabilities;
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
unsigned int* compute_root_counts(const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h)
{
	/* initialize and store the list of root table counts */
	unsigned int* root_table_counts = (unsigned int*) malloc(sizeof(unsigned int) * h.posterior.length);
	if (root_table_counts == NULL) {
		fprintf(stderr, "predict ERROR: Insufficient memory for root_table_counts.\n");
		return NULL;
	}
	for (unsigned int i = 0; i < h.posterior.length; i++)
		root_table_counts[i] = h.posterior[i].table_count;

	return root_table_counts;
}

/**
 * Computes a matrix, where every row corresponds to a
 * sample from the posterior, and every column is a table
 * at the root node. Each element contains the probability
 * of assigning the given observation to each table at the
 * root (without taking into account table sizes and the
 * alpha parameter).
 */
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

template<typename BaseDistribution, typename DataDistribution, typename K, typename V,
	typename std::enable_if<is_dirichlet<BaseDistribution>::value>::type* = nullptr>
V max_root_probability(
		const hdp<BaseDistribution, dense_categorical<V>, K, V>& h,
		const K& observation, cache<BaseDistribution, dense_categorical<V>, K, V>* caches)
{
	V pi = h.pi.get_for_atom(observation);
	if (h.posterior.length == 0)
		return pi / h.pi.sum();

	V maximum = 0.0;
	for (unsigned int i = 0; i < h.posterior.length; i++) {
		V maximum_i = pi / h.pi.sum();

		/* TODO: test numerical stability */
		bool contains;
		array_histogram<unsigned int>& table_counts = caches[i].get_table_counts(observation, contains);
		if (contains) {
			for (unsigned int j = 0; j < table_counts.counts.size; j++) {
				unsigned int table = table_counts.counts.keys[j];
				V value = (pi + table_counts.counts.values[j])
						/ (h.pi.sum() + h.posterior[i].descendant_observations[table].total());
				if (value > maximum_i)
					maximum_i = value;
			}
		}
		maximum += maximum_i;
	}

	return maximum / h.posterior.length;
}

template<typename BaseDistribution, typename DataDistribution,
	typename K, typename V, typename FeatureSet>
void predict(
	const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	const K& observation, const unsigned int* path,
	const unsigned int* const* excluded, const unsigned int* excluded_counts,
	array<FeatureSet>& x, const V* const* root_probabilities)
{
#if !defined(NDEBUG)
	if (x.length > 0) {
		fprintf(stderr, "predict WARNING: Provided 'x' is non-empty.\n");
		for (unsigned int i = 0; i < x.length; i++)
			free(x[i]);
		x.clear();
	}
#endif

	V pi = DataDistribution::probability(h.pi(), observation);
	if (h.posterior.length == 0) {
		if (!x.ensure_capacity(1) || !init(x[0], h.n->depth - 1))
			return;
		x.length = 1;
		x[0].set_probability(pi);
		for (unsigned int i = 0; i + 1 < h.n->depth; i++) {
			x[0].set_feature(i, path[i]);
			if (path[i] == IMPLICIT_NODE)
				x[0].set_excluded(i, excluded[i], excluded_counts[i]);
		}
		return;
	}

	V* probabilities = (V*) malloc(sizeof(V) * h.posterior.length);
	if (probabilities == NULL) {
		fprintf(stderr, "predict ERROR: Insufficient memory for probabilities vector.\n");
		return;
	}
	for (unsigned int i = 0; i < h.posterior.length; i++)
		probabilities[i] = compute_root_probability(h.posterior[i], root_probabilities[i], pi, h.alpha());

	/* TODO: initial array size can be set more intelligently (depending on termination condition) */
	array<hdp_search_state<K, V>> queue = array<hdp_search_state<K, V>>(1024);
	process_search_state(h, root_probabilities, path,
		excluded, excluded_counts, probabilities, 0, h.n->depth, queue, x);
	free(probabilities);

	while (queue.length > 0) {
		std::pop_heap(queue.data, queue.data + queue.length);
		hdp_search_state<K, V> state = queue.last();
		queue.length--;

		process_search_state(*state.n, root_probabilities, state.path,
			excluded, excluded_counts, state.probabilities, state.level, h.n->depth, queue, x);
		free(state);
	}

	for (unsigned int i = 0; i < queue.length; i++)
		free(queue[i]);
}

template<typename BaseDistribution, typename DataDistribution,
	typename K, typename V, typename FeatureSequence>
inline void predict(const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	const K& observation, array<FeatureSequence>& x, const V* const* root_probabilities)
{
	unsigned int* path = (unsigned int*) malloc(sizeof(unsigned int) * (h.n->depth - 1));
	if (path == NULL) {
		fprintf(stderr, "predict ERROR: Out of memory.\n");
		return;
	}

	for (unsigned int i = 0; i + 1 < h.n->depth; i++)
		path[i] = IMPLICIT_NODE;
	predict(h, observation, path, NULL, 0, x, root_probabilities);
	free(path);
}


/**
 * The below functions compute HDP likelihoods over
 * multiple observations, but only with the current
 * sampler state (instead of the set of samples in the
 * 'posterior' array, as the above code does).
 */

template<typename T>
struct histogram_keys
{
	struct histogram_key_view {
		const T* keys;
		unsigned int length;
	};

	const array_histogram<T>* histograms;

	histogram_keys(const array_histogram<T>* histograms) :
		histograms(histograms) { }

	inline histogram_key_view operator [] (unsigned int index) {
		return { histograms[index].counts.keys };
	}
};

template<typename T>
unsigned int size(const typename histogram_keys<T>::histogram_key_view& view) {
	return view.length;
}

/**
 * Constructs a map from observations to a vector, of which
 * each element is the probability of assigning that
 * observation to each table at the root (without taking
 * into account table sizes and the alpha parameter).
 */
template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
V** compute_root_probabilities(
	const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	const K* observations, unsigned int observation_count)
{
	V** probabilities = (V**) malloc(sizeof(V*) * observation_count);
	if (probabilities == NULL) {
		fprintf(stderr, "compute_root_probabilities ERROR: Insufficient memory for matrix.\n");
		return NULL;
	}
	for (unsigned int i = 0; i < observation_count; i++) {
		probabilities[i] = (V*) malloc(sizeof(V) * h.table_count);
		if (probabilities[i] == NULL) {
			fprintf(stderr, "compute_root_probabilities ERROR: Insufficient memory for matrix row.\n");
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

template<typename NodeType, typename V, typename FeatureSet>
inline void complete_path_distribution(
	const NodeType& node,
	const V* probabilities, unsigned int label,
	unsigned int row_count, const unsigned int* path,
	const unsigned int* const* excluded,
	const unsigned int* excluded_counts,
	unsigned int level, unsigned int depth,
	hash_map<FeatureSet, V*>& x)
{
	if (!x.check_size()) {
		fprintf(stderr, "complete_path_distribution ERROR: Unable to expand x map.\n");
		return;
	}

	FeatureSet& features = *((FeatureSet*) alloca(sizeof(FeatureSet)));
	if (!init(features, depth - 1)) {
		fprintf(stderr, "complete_path_distribution ERROR: Unable to initialize feature sequence.\n");
		return;
	}
	for (unsigned int i = 0; i < depth - 1; i++) {
		if (i == level) {
			features.set_feature(i, label);
		} else {
			features.set_feature(i, path[i]);
			if (path[i] == IMPLICIT_NODE && !features.set_excluded(i, excluded[i], excluded_counts[i])) {
				free(features);
				return;
			}
		}
	}

	bool contains; unsigned int index;
	V*& new_probabilities = x.get(features, contains, index);
#if !defined(NDEBUG)
	if (contains) {
		fprintf(stderr, "complete_path_distribution ERROR: Path already exists in map.\n");
		free(features);
		return;
	}
#endif
	new_probabilities = (V*) malloc(sizeof(V) * (row_count + 1));
	if (new_probabilities == NULL) {
		fprintf(stderr, "complete_path_distribution ERROR: Insufficient memory for probability array.\n");
		return;
	}
	memcpy(new_probabilities, probabilities, sizeof(V) * (row_count + 1));
	move(features, x.table.keys[index]);
	x.table.size++;

	/* compute the union of the set of child node ids and the excluded set */
	unsigned int max_excluded_count = node.child_count() + excluded_counts[level];
	if (label != IMPLICIT_NODE || max_excluded_count == 0) return;
	FeatureSet& completed = x.table.keys[index];
	completed.ensure_excluded_capacity(level, max_excluded_count);
	auto do_union = [&](unsigned int child_id, unsigned int i, unsigned int j) {
		completed.exclude_unsorted(level, child_id);
	};
	set_union(do_union, do_union, do_union,
			node.n->children.keys, node.child_count(),
			excluded[level], excluded_counts[level]);
	completed.sort_excluded(level);
}

template<typename K, typename V, typename FeatureSet>
inline void complete_path_distribution(const node_sampler<K, V>& leaf,
	const V* const* root_probabilities, const unsigned int* path,
	const V* probabilities, unsigned int row_count, unsigned int new_key,
	unsigned int level, unsigned int depth, hash_map<FeatureSet, V*>& x)
{
#if !defined(NDEBUG)
	if (level + 1 != depth - 1)
		fprintf(stderr, "complete_path_distribution WARNING: This function should only be called at the leaves.\n");
#endif

	if (!x.check_size()) {
		fprintf(stderr, "complete_path_distribution ERROR: Unable to expand x array.\n");
		return;
	}

	/* this function is only called when there are no
	   wildcards in the path (and therefore, no exclusions) */
	FeatureSet& features = *((FeatureSet*) alloca(sizeof(FeatureSet)));
	if (!init(features, depth - 1)) {
		fprintf(stderr, "complete_path_distribution ERROR: Unable to initialize feature sequence.\n");
		return;
	}
	for (unsigned int i = 0; i < level; i++)
		features.set_feature(i, path[i]);
	features.set_feature(level, new_key);

	bool contains; unsigned int index;
	V*& new_probabilities = x.get(features, contains, index);
#if !defined(NDEBUG)
	if (contains) {
		fprintf(stderr, "complete_path_distribution ERROR: Path already exists in map.\n");
		free(features);
		return;
	}
#endif
	new_probabilities = (V*) malloc(sizeof(V) * (row_count + 1));
	if (new_probabilities == NULL) {
		fprintf(stderr, "complete_path_distribution ERROR: Insufficient memory for probability array.\n");
		return;
	}
	for (unsigned int i = 0; i < row_count; i++)
		new_probabilities[i] = compute_probability(leaf, root_probabilities[i], probabilities[i], leaf.alpha());
	new_probabilities[row_count] = probabilities[row_count] * leaf.alpha() / (leaf.alpha() + leaf.customer_count);
	move(features, x.table.keys[index]);
	x.table.size++;
}

template<typename K, typename V, typename FeatureSequence>
void push_node_state(const node_sampler<K, V>& child,
	const V* const* root_probabilities, unsigned int row_count,
	const unsigned int* path, const V* probabilities,
	unsigned int new_key, unsigned int level, unsigned int depth,
	array<hdp_search_state<K, V>>& queue, hash_map<FeatureSequence, V*>& x)
{
	if (level + 1 == depth - 1) {
		/* this is the last level */
		complete_path_distribution(child, root_probabilities,
			path, probabilities, row_count, new_key, level, depth, x);
		return;
	}

#if !defined(NDEBUG)
	/* we expect the caller to ensure there is enough capacity */
	if (queue.length == queue.capacity) {
		fprintf(stderr, "push_node_state ERROR: No capacity in the queue.\n");
		return;
	}
#endif
	hdp_search_state<K, V>& state = queue[(unsigned int) queue.length];

	state.probabilities = (V*) malloc(sizeof(V) * (row_count + 1));
	if (state.probabilities == NULL) {
		fprintf(stderr, "push_node_state ERROR: Out of memory.\n");
		return;
	}
	state.path = (unsigned int*) malloc((depth - 1) * sizeof(unsigned int));
	if (state.path == NULL) {
		fprintf(stderr, "push_node_state ERROR: Out of memory.\n");
		free(state.probabilities);
		return;
	}
	for (unsigned int i = 0; i < level; i++)
		state.path[i] = path[i];
	state.path[level] = new_key;
	for (unsigned int i = level + 1; i + 1 < depth; i++)
		state.path[i] = path[i];
	state.level = level + 1;
	state.n = &child;

	for (unsigned int i = 0; i < row_count; i++) {
		state.probabilities[i] = compute_probability(child,
				root_probabilities[i], probabilities[i], child.alpha());
	}
	state.probabilities[row_count] = probabilities[row_count] * child.alpha() / (child.alpha() + child.customer_count);
	state.maximum = compute_maximum(child, root_probabilities, state.probabilities);

	queue.length++;
	std::push_heap(queue.data, queue.data + queue.length);
}

template<typename NodeType, typename K, typename V, typename FeatureSet>
void process_search_state(
	const NodeType& n, const V* const* root_probabilities,
	unsigned int row_count, const unsigned int* path,
	const unsigned int* const* excluded, const unsigned int* excluded_counts,
	const V* probabilities, unsigned int level, unsigned int depth,
	array<hdp_search_state<K, V>>& queue, hash_map<FeatureSet, V*>& x)
{
	bool contains;
	if (path[level] == IMPLICIT_NODE) {
		/* compute the score for a descendant that is not explicitly stored in the tree */
		complete_path_distribution(n, probabilities, IMPLICIT_NODE,
			row_count, path, excluded, excluded_counts, level, depth, x);

		if (!queue.ensure_capacity(queue.length + n.child_count())) {
			fprintf(stderr, "process_search_state ERROR: Unable to expand search queue.\n");
			return;
		}

		/* consider all child nodes not in the 'excluded' set */
		auto subtract = [&](unsigned int i) {
			push_node_state(n.children[i], root_probabilities, row_count,
				path, probabilities, n.child_key(i), level, depth, queue, x);
		};
		set_subtract(subtract, n.n->children.keys,
			(unsigned int) n.n->children.size, excluded[level], excluded_counts[level]);
	} else if (path[level] == UNION_NODE) {
		for (unsigned int i = 0; i < excluded_counts[level]; i++) {
			auto& child = n.get_child(excluded[level][i], contains);
			if (contains) {
				push_node_state(child, root_probabilities, row_count,
					path, probabilities, excluded[level][i], level, depth, queue, x);
			} else {
				complete_path_distribution(n, probabilities, excluded[level][i],
					row_count, path, excluded, excluded_counts, level, depth, x);
			}
		}
	} else {
		auto& child = n.get_child(path[level], contains);
		if (contains) {
			push_node_state(child, root_probabilities, row_count,
				path, probabilities, path[level], level, depth, queue, x);
		} else {
			complete_path_distribution(n, probabilities, path[level],
				row_count, path, excluded, excluded_counts, level, depth, x);
		}
	}
}

template<typename BaseDistribution, typename DataDistribution,
	typename K, typename V, typename FeatureSequence>
void predict(
	const hdp_sampler<BaseDistribution, DataDistribution, K, V>& h,
	const unsigned int* path, const unsigned int* const* excluded,
	const unsigned int* excluded_counts, hash_map<FeatureSequence, V*>& x,
	const K* observations, unsigned int observation_count, const V* const* root_probabilities)
{
#if !defined(NDEBUG)
	if (size(x) > 0) {
		fprintf(stderr, "predict WARNING: Provided 'x' is non-empty.\n");
		for (auto entry : x) {
			free(entry.key);
			free(entry.value);
		}
		x.clear();
	}
#endif

	V* probabilities = (V*) malloc(sizeof(V) * (observation_count + 1));
	if (probabilities == NULL) {
		fprintf(stderr, "predict ERROR: Insufficient memory for probabilities vector.\n");
		return;
	}
	for (unsigned int i = 0; i < observation_count; i++) {
		V pi = DataDistribution::probability(h.pi(), observations[i]);
		probabilities[i] = compute_root_probability(h, root_probabilities[i], pi, h.alpha());
	}
	probabilities[observation_count] = h.alpha() / (h.alpha() + h.customer_count);

	/* TODO: initial array size can be set more intelligently (depending on termination condition) */
	array<hdp_search_state<K, V>> queue = array<hdp_search_state<K, V>>(1024);
	process_search_state(h, root_probabilities, observation_count, path,
		excluded, excluded_counts, probabilities, 0, h.n->depth, queue, x);
	free(probabilities);

	while (queue.length > 0) {
		std::pop_heap(queue.data, queue.data + queue.length);
		hdp_search_state<K, V> state = queue.last();
		queue.length--;

		process_search_state(*state.n, root_probabilities, observation_count, state.path,
			excluded, excluded_counts, state.probabilities, state.level, h.n->depth, queue, x);
		free(state);
	}

	for (unsigned int i = 0; i < queue.length; i++)
		free(queue[i]);
}


/**
 * These functions compute HDP likelihoods for a single
 * observation over all samples in the 'posterior' array.
 */

template<typename NodeType, typename V>
inline void process_search_state_node(
	const NodeType& n, unsigned int label,
	const unsigned int* root_table_counts,
	const V* const* root_probabilities, const unsigned int* path,
	const unsigned int* excluded, unsigned int excluded_count,
	V* probabilities, unsigned int level, unsigned int depth,
	V& output_probability)
{
	bool contains;
	auto& child = n.get_child(label, contains);
	if (!contains) {
		output_probability = log(compute_maximum(n, root_table_counts,
				root_probabilities, probabilities)) - log(n.posterior.length);
	} else {
		for (unsigned int i = 0; i < child.posterior.length; i++) {
			probabilities[i] = compute_probability(child.posterior[i],
					root_probabilities[i], probabilities[i], child.alpha());
		}

		if (level + 1 == depth - 1) {
			output_probability = sum(probabilities, child.posterior.length) / child.posterior.length;
			return;
		} else {
			process_search_state(child, root_table_counts, root_probabilities, path,
					excluded, excluded_count, probabilities, level + 1, depth, output_probability);
		}
	}
}

template<typename NodeType, typename V>
void process_search_state(
	const NodeType& n, const unsigned int* root_table_counts,
	const V* const* root_probabilities, const unsigned int* path,
	const unsigned int* excluded, unsigned int excluded_count,
	V* probabilities, unsigned int level, unsigned int depth,
	V& output_probability)
{
	if (path[level] == IMPLICIT_NODE) {
		output_probability = log(compute_maximum(n, root_table_counts,
				root_probabilities, probabilities, excluded, excluded_count)) - log(n.posterior.length);
		return;
	} else if (path[level] == UNION_NODE) {
		fprintf(stderr, "process_search_state ERROR: Support for UNION_NODE is unimplemented.");
		exit(EXIT_FAILURE);
		/* TODO: implement this; it should look something like below */
		/*process_search_state_node(n, excluded[0], root_table_counts, root_probabilities,
				path, excluded, excluded_count, probabilities, level, depth, output_probability);
		for (unsigned int i = 1; i < excluded_count; i++) {
			double subtree_output;
			process_search_state_node(n, excluded[i], root_table_counts, root_probabilities,
					path, excluded, excluded_count, probabilities, level, depth, subtree_output);
			output_probability = max(subtree_output, output_probability);
		}*/
	} else {
		process_search_state_node(n, path[level], root_table_counts, root_probabilities,
				path, excluded, excluded_count, probabilities, level, depth, output_probability);
	}
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
void predict(
	const hdp<BaseDistribution, DataDistribution, K, V>& h,
	const K& observation, const unsigned int* path,
	const unsigned int* excluded, unsigned int excluded_count,
	const V* const* root_probabilities, V& output_probability)
{
#if !defined(NDEBUG)
	if (!std::is_sorted(excluded, excluded + excluded_count)) {
		fprintf(stderr, "predict ERROR: Provided 'excluded' is not sorted.\n");
		exit(EXIT_FAILURE);
	}
#endif

	V pi = h.pi.get_for_atom(observation) / h.pi.sum();
	if (h.posterior.length == 0) {
		output_probability = log(pi);
		return;
	}

	V* probabilities = (V*) malloc(sizeof(V) * h.posterior.length);
	if (probabilities == NULL) {
		fprintf(stderr, "predict ERROR: Insufficient memory for probabilities vector.\n");
		return;
	}
	for (unsigned int i = 0; i < h.posterior.length; i++)
		probabilities[i] = compute_root_probability(h.posterior[i], root_probabilities[i], pi, h.alpha());

	process_search_state(h, root_probabilities, path,
			excluded, excluded_count, probabilities, 0, h.depth, output_probability);

	free(probabilities);
}

template<typename V>
void log_normalize(array<hdp_path<V>>& x) {
	V sum = 0.0;
	for (unsigned int i = 0; i < x.length; i++) {
		hdp_path<V>& path = x[i];
		sum += path.probability * path.count;
	}

	V log_sum = log(sum);
	for (unsigned int i = 0; i < x.length; i++)
		x[i].probability = log(x[i].probability) - log_sum;
}

template<typename V>
void log(array<hdp_path<V>>& x) {
	for (unsigned int i = 0; i < x.length; i++)
		x[i].probability = log(x[i].probability);
}


/**
 * The following are unit tests for the HDP MCMC code.
 */

#include <core/timer.h>

template<typename BaseDistribution, typename DataDistribution,
	typename BaseParameters, typename K, typename V>
bool discrete_hdp_test(const BaseParameters& base_params,
		const V* alpha, unsigned int* const* const x, const K* y,
		unsigned int train_count, unsigned int depth)
{
	hdp<BaseDistribution, DataDistribution, K, V> h(base_params, alpha, depth);

	printf("discrete_hdp_test: Adding %u training samples, with tree depth %u.\n", train_count, depth);
	for (unsigned int i = 0; i < train_count; i++) {
		if (!add(h, x[i], depth, y[i]))
			return false;
	}

	hdp_sampler<BaseDistribution, DataDistribution, K, V> sampler(h);
	cache<BaseDistribution, DataDistribution, K, V> cache(sampler);
	prepare_sampler(sampler, cache);

	timer stopwatch;
	unsigned int burn_in = 200;
	for (unsigned int i = 0; i < burn_in; i++)
		sample_hdp<true>(sampler, cache);
	printf("discrete_hdp_test: Completed %u burn-in iterations in %lf s.\n", burn_in, stopwatch.nanoseconds() / 1.0e9);

	stopwatch.start();
	unsigned int iterations = 800;
	unsigned int skip = 5;
	for (unsigned int i = 0; i < iterations; i++) {
		sample_hdp<true>(sampler, cache);
		if (i % skip == 0)
			sampler.add_sample();
	}
	printf("discrete_hdp_test: Completed %u iterations in %lf s.\n", iterations, stopwatch.nanoseconds() / 1.0e9);
	printf("discrete_hdp_test: Gathered %zu samples.\n", sampler.posterior.length);

	array<hdp_path<V>> paths = array<hdp_path<V>>(32);
	V** root_probabilities = compute_root_probabilities(sampler, (unsigned int) 1);
	predict(sampler, (unsigned int) 1, paths, root_probabilities);
	cleanup_root_probabilities(root_probabilities, (unsigned int) sampler.posterior.length);

	printf("discrete_hdp_test: The following paths were predicted\n");
	FILE* out = stdout;
	if (paths.length > 1)
		quick_sort(paths, hdp_path_sorter<V>(depth));
	for (unsigned int i = 0; i < paths.length; i++) {
		print(paths[i].path, h.depth - 1, out);
		printf(" with probability %lf.\n", paths[i].probability);

		free(paths[i]);
	}
	return true;
}

template<typename V>
bool hdp_test() {
	V pi = 0.01;
	V alpha[] = { 1000000.0, 0.01 };
	unsigned int depth = 2;
	unsigned int atom_count = 100;

	unsigned int index = 0;
	unsigned int* x[20 + 20 + 20];
	unsigned int y[20 + 20 + 20];
	for (unsigned int i = 0; i < 20; i++) {
		x[index] = (unsigned int*) alloca(sizeof(unsigned int) * (depth - 1));
		x[index][0] = 1;
		y[index] = 1;
		index++;
	} for (unsigned int i = 0; i < 20; i++) {
		x[index] = (unsigned int*) alloca(sizeof(unsigned int) * (depth - 1));
		x[index][0] = 2;
		y[index] = 2;
		index++;
	} for (unsigned int i = 0; i < 20; i++) {
		x[index] = (unsigned int*) alloca(sizeof(unsigned int) * (depth - 1));
		x[index][0] = 3;
		y[index] = 1 + (i % 2);
		index++;
	}

	dense_categorical<V> prior = dense_categorical<V>(atom_count);
	for (unsigned int i = 0; i < atom_count; i++)
		prior.phi[i] = 1.0 / atom_count;
	return discrete_hdp_test<dense_categorical<V>, constant<unsigned int>>(
			prior, alpha, x, y, (unsigned int) array_length(y), depth);
	//return discrete_hdp_test<symmetric_dirichlet<V>, dense_categorical<V>>(
	//		symmetric_dirichlet<V>(atom_count, pi), alpha, x, y, (unsigned int) array_length(y), depth);
}

template<typename V>
bool hdp_complex_test() {
	V pi = 0.001;
	V alpha[] = { 1000000.0, 0.001 };
	unsigned int depth = 2;
	unsigned int atom_count = 100;

	unsigned int* x[10 + 20 + 40 + 20];
	unsigned int y[10 + 20 + 40 + 20];
	unsigned int index = 0;
	for (unsigned int i = 0; i < 10; i++) {
		x[index] = (unsigned int*) alloca(sizeof(unsigned int) * (depth - 1));
		x[index][0] = 1;
		y[index] = 1 + (i % 2);
		index++;
	} for (unsigned int i = 0; i < 20; i++) {
		x[index] = (unsigned int*) alloca(sizeof(unsigned int) * (depth - 1));
		x[index][0] = 2;
		y[index] = 1 + (i % 2);
		index++;
	} for (unsigned int i = 0; i < 40; i++) {
		x[index] = (unsigned int*) alloca(sizeof(unsigned int) * (depth - 1));
		x[index][0] = 3;
		y[index] = 1 + (i % 2);
		index++;
	} for (unsigned int i = 0; i < 20; i++) {
		x[index] = (unsigned int*) alloca(sizeof(unsigned int) * (depth - 1));
		x[index][0] = 4;
		y[index] = 3;
		index++;
	}

	return discrete_hdp_test<symmetric_dirichlet<V>, dense_categorical<V>>(
			symmetric_dirichlet<V>(atom_count, pi), alpha, x, y, (unsigned int) array_length(y), depth);
}

template<typename V>
bool hdp_verb_test() {
	V pi = 0.001;
	V alpha[] = { 1000000.0, 1000000.0, 0.001, 1.0, 1.0, 400.0 };
	unsigned int depth = 6;
	unsigned int atom_count = 100;

	unsigned int x[][5] = {
			{4294967292, 1786902, 163, 160, 20347},
			{4294967292, 1786826, 134, 158, 120659},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786902, 163, 162, 20347},
			{4294967292, 1786826, 124, 158, 3582},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786826, 124, 158, 791541},
			{4294967292, 1786874, 163, 155, 20346},
			{4294967292, 1786826, 124, 158, 94746},
			{4294967292, 1786826, 124, 158, 66199},
			{4294967292, 1786874, 163, 163, 20347},
			{4294967292, 1786826, 124, 158, 66199},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 163, 733497},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 124, 158, 764406},
			{4294967292, 1786826, 124, 160, 94742},
			{4294967292, 1786775, 149, 159, 796815},
			{4294967292, 1786826, 134, 163, 66197},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786826, 124, 163, 34468},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 134, 158, 66197},
			{4294967292, 1786826, 124, 158, 34468},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786874, 163, 155, 772669},
			{4294967292, 1786826, 124, 163, 15246},
			{4294967292, 1786826, 134, 160, 66197},
			{4294967292, 1786826, 124, 158, 4588},
			{4294967292, 1786874, 163, 155, 20347},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786826, 134, 158, 2508},
			{4294967292, 1786826, 124, 158, 15246},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 134, 163, 66197},
			{4294967292, 1786902, 163, 160, 20347},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 158, 3582},
			{4294967292, 1786826, 124, 158, 791540},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 159, 796815},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 155, 772021},
			{4294967292, 1786775, 149, 155, 768186},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 155, 768186},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 155, 766880},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 155, 768186},
			{4294967292, 1786775, 149, 155, 766720},
			{4294967292, 1786775, 149, 155, 768327},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 150, 81996},
			{4294967292, 1786775, 149, 155, 772021},
			{4294967292, 1786775, 149, 150, 114603},
			{4294967292, 1786775, 149, 155, 920},
			{4294967292, 1786775, 149, 155, 772022},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 155, 766720},
			{4294967292, 1786775, 149, 159, 796815},
			{4294967292, 1786775, 149, 155, 81996},
			{4294967292, 1786775, 149, 150, 920},
			{4294967292, 1786775, 149, 150, 740510},
			{4294967292, 1786775, 149, 155, 28942},
			{4294967292, 1786775, 149, 150, 94748},
			{4294967292, 1786826, 147, 163, 98499},
			{4294967292, 1786826, 124, 158, 94735},
			{4294967292, 1786826, 124, 163, 15246},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786826, 124, 158, 97898},
			{4294967292, 1786826, 124, 163, 15246},
			{4294967292, 1786826, 134, 158, 66197},
			{4294967292, 1786826, 124, 163, 733497},
			{4294967292, 1786826, 124, 163, 15246},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 134, 158, 2508},
			{4294967292, 1786826, 124, 158, 66199},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 124, 163, 15246},
			{4294967292, 1786826, 124, 158, 97899},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 134, 163, 66197},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 163, 15246},
			{4294967292, 1786826, 134, 158, 2508},
			{4294967292, 1786826, 124, 163, 15246},
			{4294967292, 1786826, 134, 160, 66197},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 134, 158, 2508},
			{4294967292, 1786826, 124, 158, 15246},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 124, 158, 3582},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 158, 791540},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 163, 34468},
			{4294967292, 1786826, 124, 158, 3582},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786826, 124, 163, 15246},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786826, 124, 163, 34468},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786826, 124, 158, 66199},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786826, 124, 163, 34468},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 158, 94746},
			{4294967292, 1786826, 124, 158, 66199},
			{4294967292, 1786826, 124, 163, 34468},
			{4294967292, 1786826, 134, 158, 2508},
			{4294967292, 1786826, 124, 163, 3582},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 124, 158, 764406},
			{4294967292, 1786826, 124, 158, 66199},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786826, 124, 163, 34468},
			{4294967292, 1786826, 124, 158, 791540},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786826, 134, 158, 2508},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786826, 124, 163, 916},
			{4294967292, 1786826, 134, 158, 2508},
			{4294967292, 1786826, 124, 158, 34468},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786826, 124, 158, 916},
			{4294967292, 1786826, 124, 163, 791540},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 163, 764406},
			{4294967292, 1786826, 124, 158, 764406},
			{4294967292, 1786826, 134, 158, 66197},
			{4294967292, 1786826, 134, 163, 2508},
			{4294967292, 1786826, 124, 163, 15246},
			{4294967292, 1786961, 87, 87, 561935},
			{4294967292, 1786961, 87, 87, 563759}
	};
	unsigned int y[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 2, 3, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 3,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
			2, 5, 6, 2, 2, 7, 7, 4, 7, 5, 7, 7, 6, 7, 2, 7,
			4, 7, 7, 2, 2, 4, 4, 2, 4, 4, 2, 7, 4, 2, 4, 4,
			2, 2, 2, 5, 2, 2, 4, 4, 2, 2, 4, 2, 2, 5, 2, 7,
			2, 4, 7, 2, 4, 4, 6, 2, 7, 7, 4, 6, 2, 2, 2, 2,
			7, 2, 4, 4, 2, 4, 8, 4, 2, 7, 2, 4, 7, 2, 2, 9, 9};

	unsigned int** paths = (unsigned int**) alloca(sizeof(unsigned int*) * array_length(x));
	for (unsigned int i = 0; i < array_length(x); i++) {
		paths[i] = (unsigned int*) malloc(sizeof(unsigned int) * (depth - 1));
		for (unsigned int j = 0; j < depth - 1; j++)
			paths[i][j] = x[i][j];
	}

	bool success = discrete_hdp_test<symmetric_dirichlet<V>, dense_categorical<V>>(
			symmetric_dirichlet<V>(atom_count, pi), alpha, paths, y, (unsigned int) array_length(y), depth);

	for (unsigned int i = 0; i < array_length(x); i++)
		free(paths[i]);
	return success;
}

#endif /* MCMC_H_ */
