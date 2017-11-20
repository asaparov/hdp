/**
 * \file hdp.h
 *
 * This file contains data structures that implement
 * [Dirichlet processes](https://en.wikipedia.org/wiki/Dirichlet_process)
 * and hierarchical Dirichlet processes. This file does not implement
 * inference algorithms for these structures. See hdp/mcmc.h for examples that
 * use DPs and HDPs and perform inference.
 *
 * Dirichlet processes
 * -------------------
 *
 * A *Dirichlet process* (DP) can be understood as a distribution
 * *over distributions*. That is, samples from a Dirichlet process are
 * themselves distributions. A Dirichlet process is characterized by two
 * parameters: a real-valued concentration parameter \f$ \alpha > 0 \f$, and a
 * base distribution \f$ H \f$. So if we let \f$ G \f$ be a random variable
 * distributed according to a Dirichlet process with parameters \f$ \alpha \f$
 * and \f$ H \f$, we can express this as:
 * \f[ G \sim \text{DP}(H, \alpha). \f]
 *
 * There are a handful of equivalent representations of Dirichlet processes.
 * One such representation is the
 * [Chinese restaurant process](https://en.wikipedia.org/wiki/Dirichlet_process#The_Chinese_restaurant_process).
 * Imagine a restaurant with an infinite number of tables, numbered
 * \f$ 0, 1, \ldots \f$ On each table is an independent sample from the base
 * distribution \f$ H \f$. When the first customer walks into the restaurant,
 * they sit at table \f$ 0 \f$. When the second customer enters, they either
 * sit at table \f$ 0 \f$ with probability \f$ 1/(1 + \alpha) \f$ or they sit
 * at table \f$ 1 \f$ with probability \f$ \alpha/(1 + \alpha) \f$. More
 * generally, when the \f$ n \f$-th customer enters the restaurant, they choose
 * to sit at a *non-empty* table \f$ i \f$ with probability proportional to the
 * number of people sitting at that table, or they sit at the next empty table
 * with probability proportional to \f$ \alpha \f$.
 *
 * This process is equivalent to the process of drawing samples from $G$.
 * \f[
 * 	\begin{align*}
 * 		G &\sim \text{DP}(H, \alpha), \\
 * 		X_1, X_2, \ldots, X_n &\sim G,
 * 	\end{align*}
 * \f]
 * where \f$ X_1, \ldots, X_n \f$ are drawn independently and identically from
 * \f$ G \f$. In the restaurant metaphor, \f$ X_1 \f$ is the sample from
 * \f$ H \f$ that the first customer discovered at their table, \f$ X_2 \f$ is
 * the sample from \f$ H \f$ that the second customer discovered, and so on.
 * Thus, this representation describes how to draw samples from a Dirichlet
 * process, when \f$ G \f$ is collapsed/integrated out.
 *
 * It is impossible to write a closed-form expression for the distribution
 * \f$ G \f$, since its specification requires an infinite amount of
 * information. But for the useful applications of the Dirichlet process, this
 * is not necessary.
 *
 * Hierarchical Dirichlet processes
 * --------------------------------
 *
 * A hierarchical Dirichlet process (HDP) is a hierarchy of random variables,
 * where each random variable is distributed according to a Dirichlet process
 * with base distribution given by the parent in the hierarchy. To be more
 * precise, given a tree \f$ T \f$. Every node \f$ \textbf{n} \in T \f$ is
 * associated with a random variable \f$ G_{\textbf{n}} \f$ such that
 * 		\f[ G_{\textbf{n}} \sim \text{DP}(G_{p(\textbf{n})}, \alpha_{\textbf{n}}), \f]
 * where \f$ p(\textbf{n}) \f$ returns the parent node of \f$ \textbf{n} \f$ in
 * \f$ T \f$. The root node \f$ \textbf{0} \f$ is drawn from a single root base
 * distribution:
 * 		\f[ G_{\textbf{0}} \sim \text{DP}(H, \alpha_{\textbf{0}}). \f]
 * Note that the concentration parameter may also differ across the nodes in
 * the tree.
 *
 * The HDP allows statistical information to be shared across groups, and is
 * useful for modeling clustered data.
 *
 * DP/HDP mixture models
 * ---------------------
 *
 * DPs and HDPs are frequently used in mixture models, where the samples from
 * the DP/HDP are not themselves directly observed. Rather, they are inputs to
 * another distribution, which in turn, provides the observed samples. For
 * example, the following is a simple DP mixture model, where the base
 * distribution is Beta and the likelihood is Bernoulli:
 * \f[
 * 	\begin{align*}
 * 		H &= \text{Beta}(2, 4), \\
 * 		G &\sim \text{DP}(H, 0.1), \\
 * 		X_1, \ldots, X_n &\sim G, \\
 * 		Y_i &\sim \text{Bernoulli}(X_i) \text{ where } i = 0, \ldots, n.
 * 	\end{align*}
 * \f]
 * In the mixture model, we observe \f$ Y_i \f$ but not \f$ X_i \f$. If the
 * user wishes to use the DP/HDP samples directly, they can do so by using the
 * constant (degenerate) distribution as the likelihood.
 *
 *  <!-- Created on: Jul 10, 2016
 *           Author: asaparov -->
 */

#ifndef HDP_H_
#define HDP_H_

#include <core/map.h>
#include <math/multiset.h>
#include <limits.h>

/**
 * Each node in the HDP hierarchy has a collection of child nodes. Each child
 * node is indexed by an `unsigned int` key. This is a special key that
 * represents the set of all keys.
 */
#define IMPLICIT_NODE UINT_MAX

#define UNION_NODE (UINT_MAX - 3)

using namespace core;

/* forward declarations */
#if !defined(DOXYGEN_IGNORE)
template<typename K, typename V> struct node;
template<typename BaseDistribution, typename DataDistribution, typename K, typename V> struct hdp;
#endif

/**
 * Represents a non-root node in an HDP hierarchy.
 *
 * To use this structure or ::hdp, an inference method is required. See
 * hdp/mcmc.h for an example.
 *
 * \tparam K the generic type of the observations drawn from this distribution.
 * \tparam V the type of the probabilities.
 */
template<typename K, typename V>
struct node {
	/**
	 * The type of the observations drawn from this distribution.
	 */
	typedef K atom_type;

	/**
	 * The type of the probabilities.
	 */
	typedef V value_type;

	/**
	 * The child nodes of this node in the HDP hierarchy.
	 */
	array_map<unsigned int, node<K, V>> children;

	/**
	 * The concentration parameter \f$ \alpha \f$ at this node.
	 */
	V alpha;

	/**
	 * The natural logarithm of node::alpha.
	 */
	V log_alpha;

	/**
	 * The observations drawn from this node.
	 */
	array<K> observations;

	/**
	 * Constructs an HDP node with the given concentration parameter `alpha`
	 * and no child nodes or observations.
	 */
	node(const V& alpha) :
		children(4), alpha(alpha), log_alpha(log(alpha)), observations(4)
	{ }

	~node() { free(); }

	/**
	 * Returns the concentration parameter node::alpha.
	 */
	inline V get_alpha() const {
		return alpha;
	}

	/**
	 * Returns the logarithm of the concentration parameter node::log_alpha.
	 */
	inline V get_log_alpha() const {
		return log_alpha;
	}

	/**
	 * Moves the HDP node from `src` to `dst`.
	 */
	static inline void move(const node<K, V>& src, node<K, V>& dst) {
		dst.alpha = src.alpha;
		dst.log_alpha = src.log_alpha;
		core::move(src.children, dst.children);
		core::move(src.observations, dst.observations);
	}

	template<typename Metric>
	static inline long unsigned int size_of(const node<K, V>& n, const Metric& metric) {
		long unsigned int sum = core::size_of(n.alpha) + core::size_of(n.log_alpha)
			+ core::size_of(n.children, make_key_value_metric(default_metric(), metric))
			+ core::size_of(n.observations, metric);
		return sum;
	}

	/**
	 * Swaps the HDP node in `first` and `second`.
	 */
	static inline void swap(node<K, V>& first, node<K, V>& second) {
		core::swap(first.children, second.children);
		core::swap(first.alpha, second.alpha);
		core::swap(first.log_alpha, second.log_alpha);
		core::swap(first.observations, second.observations);
	}

	/**
	 * Releases the memory resources associated with the HDP node `n`.
	 */
	static inline void free(node<K, V>& n) {
		n.free();
		core::free(n.observations);
		core::free(n.children);
	}

private:
	inline void free() {
		for (unsigned int i = 0; i < children.size; i++)
			core::free(children.values[i]);
		for (unsigned int i = 0; i < observations.length; i++)
			core::free(observations[i]);
	}

	template<typename A, typename B, typename C, typename D>
	friend struct hdp;
};

/**
 * Initializes the given HDP node with the concentration parameter `alpha` and
 * no child nodes or observations.
 */
template<typename K, typename V>
inline bool init(node<K, V>& n, const V& alpha)
{
	n.alpha = alpha;
	n.log_alpha = ::log(alpha);
	if (!array_map_init(n.children, 4)) {
		fprintf(stderr, "init ERROR: Unable to initialize children map for node.\n");
		return false;
	} else if (!array_init(n.observations, 4)) {
		fprintf(stderr, "init ERROR: Unable to initialize observation array for node.\n");
		core::free(n.children);
		return false;
	}
	return true;
}

/**
 * Copies the HDP node from `src` to `dst` (as well as its descendants,
 * recursively), while adding a entries to `node_map` mapping the pointers of
 * every source node to the pointer of the corresponding destination node.
 */
template<typename K, typename V>
bool copy(
	const node<K, V>& src, node<K, V>& dst,
	hash_map<const node<K, V>*, node<K, V>*>& node_map)
{
	if (!node_map.put(&src, &dst))
		return false;
	dst.alpha = src.alpha;
	dst.log_alpha = src.log_alpha;
	if (!array_init(dst.observations, core::max(1u, (unsigned int) src.observations.length))) {
		return false;
	} else if (!array_map_init(dst.children, (unsigned int) src.children.size)) {
		free(dst.observations);
		return false;
	}
	for (unsigned int i = 0; i < src.observations.length; i++)
		dst.observations.add(src.observations[i]);

	for (unsigned int i = 0; i < src.children.size; i++) {
		dst.children.keys[i] = src.children.keys[i];
		if (!copy(src.children.values[i], dst.children.values[i], node_map))
			return false; /* TODO: free memory */
		dst.children.size++;
	}
	return true;
}

inline void print_prefix(FILE* out, unsigned int indent) {
	for (unsigned int i = 0; i < indent; i++)
		fputc('|', out);
	fputc(' ', out);
}

/**
 * A scribe structure useful for reading/writing/printing node and hdp objects.
 */
template<typename AtomScribe, typename KeyScribe>
struct node_scribe {
	/**
	 * The scribe for reading/writing/printing observations drawn from HDP nodes.
	 */
	AtomScribe& atom_scribe;

	/**
	 * The scribe for reading/writing/printing the keys that index the children
	 * at each HDP node.
	 */
	KeyScribe& key_scribe;

	/**
	 * Constructs the node_scribe with the given atom scribe and the key scribe.
	 */
	node_scribe(AtomScribe& atom_scribe, KeyScribe& key_scribe) :
		atom_scribe(atom_scribe), key_scribe(key_scribe) { }
};

template<typename NodeType, typename AtomReader, typename KeyReader>
bool read_node(NodeType& node, FILE* in, AtomReader& atom_reader, KeyReader& key_reader)
{
	/* we assume alpha, beta, table assignments, and
	   posterior have already been read by the caller */

	if (!read(node.observations, in, atom_reader)) return false;

	auto node_reader = node_scribe<AtomReader, KeyReader>(atom_reader, key_reader);
	if (!read(node.children, in, key_reader, node_reader)) {
		for (unsigned int i = 0; i < node.observations.length; i++)
			free(node.observations[i]);
		free(node.observations);
		return false;
	}
	if (node.children.size > 1)
		sort(node.children.keys, node.children.values, node.children.size, default_sorter());
	return true;
}

/**
 * Reads an HDP `node` (and all descendants) from `in`.
 * \tparam AtomScribe a scribe type for which the function
 * 		`bool read(K&, FILE*, AtomScribe&)` is defined.
 * \tparam KeyScribe a scribe type for which the function
 * 		`bool read(unsigned int&, FILE*, KeyScribe&)` is defined.
 */
template<typename K, typename V, typename AtomReader, typename KeyReader>
bool read(node<K, V>& node, FILE* in,
		node_scribe<AtomReader, KeyReader>& node_reader)
{
	if (!read(node.alpha, in)) return false;
	node.log_alpha = log(node.alpha);

	return read_node(node, in, node_reader.atom_scribe, node_reader.key_scribe);
}

template<typename NodeType, typename AtomWriter, typename KeyWriter>
bool write_node(
	const NodeType& node, FILE* out,
	AtomWriter& atom_writer,
	KeyWriter& key_writer)
{
	/* we assume alpha has already been written by the caller */

	if (!write(node.observations, out, atom_writer)) return false;

	auto node_writer = node_scribe<AtomWriter, KeyWriter>(atom_writer, key_writer);
	return write(node.children, out, key_writer, node_writer);
}

/**
 * Writes the given HDP `node` (and all descendants) to `out`.
 * \tparam AtomScribe a scribe type for which the function
 * 		`bool write(const K&, FILE*, AtomScribe&)` is defined.
 * \tparam KeyScribe a scribe type for which the function
 * 		`bool write(unsigned int, FILE*, KeyScribe&)` is defined.
 */
template<typename K, typename V, typename AtomWriter, typename KeyWriter>
bool write(const node<K, V>& node, FILE* out,
		node_scribe<AtomWriter, KeyWriter>& node_writer)
{
	if (!write(node.alpha, out)) return false;
	return write_node(node, out, node_writer.atom_scribe, node_writer.key_scribe);
}

template<typename KeyPrinter>
inline void increment_level(KeyPrinter&) { }

template<typename KeyPrinter>
inline void decrement_level(KeyPrinter&) { }

template<typename K, typename V>
inline void print_crf_node(const node<K, V>& n, FILE* out, unsigned int level) {
	print_prefix(out, level); fprintf(out, "table_assignments: ");
	print(n.table_assignments, n.table_count, out); fputc('\n', out);
	print_prefix(out, level); fprintf(out, "root_assignments: ");
	print(n.root_assignments, n.table_count, out); fputc('\n', out);
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
inline void print_crf_node(const hdp<BaseDistribution, DataDistribution, K, V>& h, FILE* out, unsigned int level) { }

template<typename NodeType, typename AtomPrinter, typename KeyPrinter>
void print_node(const NodeType& node, FILE* out, unsigned int level,
		unsigned int root_cluster_count, AtomPrinter& atom_printer, KeyPrinter& key_printer)
{
	print_prefix(out, level); fprintf(out, "alpha: %lf\n", node.get_alpha());
	if (node.observations.length > 0) {
		print_prefix(out, level); fprintf(out, "observations: ");
		print(node.observations, out, atom_printer); fputc('\n', out);
	} else {
		print_prefix(out, level); fprintf(out, "observations: None.\n");
	}
	print_prefix(out, level); fprintf(out, "table_sizes: ");
	print(node.table_sizes, node.table_count, out); fputc('\n', out);
	print_crf_node(node, out, level);
	print_prefix(out, level); fprintf(out, "customer_count: %u\n", node.customer_count);
	for (unsigned int i = 0; i < node.table_count; i++) {
		print_prefix(out, level); fprintf(out, "table %u: ", i);
		print(node.descendant_observations[i], out, atom_printer); fputc('\n', out);
	}

	for (unsigned int i = 0; i < node.children.size; i++) {
		print_prefix(out, level); fputc('\n', out);
		print_prefix(out, level); fprintf(out, "child ");
		print(node.children.keys[i], out, key_printer); fprintf(out, ":\n");

		increment_level(key_printer);
		print_node(node.children.values[i], out, level + 1,
				root_cluster_count, atom_printer, key_printer);
		decrement_level(key_printer);
	}
}

/**
 * Represents the root node in an HDP hierarchy (or equivalently, a single
 * Dirichlet process). The hierarchy has a maximum depth, as specified by the
 * hdp::depth variable. When new nodes are added to the HDP using the function
 * ::add, they are initialized using the concentration parameter in hdp::alpha
 * that corresponds to the *level* of the node in the tree. However, the
 * concentration parameter can easily be changed by the user after the node is
 * created. In addition, the user can add nodes manually to the hierarchy.
 *
 * To use this structure, an inference method is required. See hdp/mcmc.h for
 * an example.
 *
 * \tparam BaseDistribution the type of the base distribution.
 * \tparam DataDistribution the type of the likelihood (as in a
 * 		[DP/HDP mixture model](#dp/hdp-mixture-models)).
 * \tparam K the generic type of the observations drawn from this distribution.
 * \tparam V the type of the probabilities.
 */
template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
struct hdp {
	/**
	 * The generic type of the observations drawn from this distribution.
	 */
	typedef K atom_type;

	/**
	 * The type of the probabilities.
	 */
	typedef V value_type;

	/**
	 * The type of the base distribution.
	 */
	typedef BaseDistribution base_distribution_type;

	/**
	 * The type of the likelihood (as in a
	 * [DP/HDP mixture model](#dp/hdp-mixture-models)).
	 */
	typedef DataDistribution data_distribution_type;

	/**
	 * The base distribution.
	 */
	BaseDistribution pi;

	/**
	 * The depth of the HDP hierarchy, where a depth of 1 indicates that the
	 * root node is the only node in the hierarchy.
	 */
	unsigned int depth;

	/**
	 * An array of concentration parameters, with length hdp::depth, one for
	 * each level in the HDP.
	 */
	V* alpha;

	/**
	 * The natural logarithm of the concentration parameter of this node (i.e.
	 * `hdp::alpha[0]`).
	 */
	V log_alpha;

	/**
	 * The child nodes of this root node in the HDP hierarchy.
	 */
	array_map<unsigned int, node<K, V>> children;

	/**
	 * The observations sampled from this distribution.
	 */
	array<K> observations;

	/**
	 * Constructs this HDP root node with the given parameters to the base
	 * distribution `base_params`, the given array of concentration parameters
	 * `alpha` (one for every level in the hierarchy), and the given `depth` of
	 * the hierarchy.
	 * \tparam BaseParameters the type of the parameters passed to the
	 * 		constructor of BaseDistribution.
	 */
	template<typename BaseParameters>
	hdp(const BaseParameters& base_params, const V* alpha, unsigned int depth) :
		pi(base_params), depth(depth), log_alpha(::log(alpha[0])), children(4), observations(4)
	{
		if (!initialize(alpha)) {
			fprintf(stderr, "hdp ERROR: Error during initialization.\n");
			exit(EXIT_FAILURE);
		}
	}

	~hdp() { free(); }

	void ensure_atom_count(unsigned int atom_count) {
		pi.ensure_atom_count(atom_count);
	}

	/**
	 * Returns the concentration parameter of this root node.
	 */
	inline V get_alpha() const {
		return alpha[0];
	}

	/**
	 * Returns the natural logarithm of the concentration parameter of this
	 * root node.
	 */
	inline V get_log_alpha() const {
		return log_alpha;
	}

	inline void set_uninitialized() {
		alpha = NULL;
	}

	inline bool uninitialized() const {
		return alpha == NULL;
	}

	template<typename Metric>
	static inline long unsigned int size_of(
			const hdp<BaseDistribution, DataDistribution, K, V>& h, const Metric& metric) {
		long unsigned int sum = core::size_of(h.pi) + core::size_of(h.depth) + core::size_of(h.log_alpha)
				+ core::size_of(h.children) + core::size_of(h.observations);
		sum += sizeof(unsigned int) * h.depth; /* for alpha */

		return sum;
	}

	/**
	 * Releases the memory resources of this HDP root node.
	 */
	static inline void free(hdp<BaseDistribution, DataDistribution, K, V>& h) {
		h.free();
		core::free(h.observations);
		core::free(h.children);
		core::free(h.pi);
	}

private:
	bool initialize(const V* alpha_src)
	{
		alpha = (V*) malloc(sizeof(V) * depth);
		if (alpha == NULL) {
			fprintf(stderr, "hdp.initialize ERROR: Insufficient memory for alpha.\n");
			return false;
		}

		for (unsigned int i = 0; i < depth; i++)
			alpha[i] = alpha_src[i];
		return true;
	}

	inline void free() {
		for (unsigned int i = 0; i < children.size; i++)
			core::free(children.values[i]);
		for (unsigned int i = 0; i < observations.length; i++)
			core::free(observations[i]);
		core::free(alpha);
	}

	friend struct node<K, V>;

	template<typename A, typename B, typename C, typename D, typename E>
	friend bool init(hdp<A, B, C, D>&, E&, const D*, unsigned int);
};

/**
 * Initializes the given HDP root node `h` with the given parameters to the
 * base distribution `base_params`, the given array of concentration parameters
 * `alpha` (one for every level in the hierarchy), and the given `depth` of the
 * hierarchy.
 * \tparam BaseParameters the type of the parameters passed to the
 * 		constructor of BaseDistribution.
 */
template<typename BaseDistribution, typename DataDistribution,
	typename K, typename V, typename BaseParameters>
bool init(hdp<BaseDistribution, DataDistribution, K, V>& h,
		BaseParameters& base_params, const V* alpha, unsigned int depth)
{
	h.depth = depth;
	h.log_alpha = log(alpha[0]);
	if (!init(h.pi, base_params)) {
		fprintf(stderr, "init ERROR: Unable to initialize pi in hdp.\n");
		return false;
	} else if (!array_map_init(h.children, 4)) {
		fprintf(stderr, "init ERROR: Unable to initialize children array map in hdp.\n");
		free(h.pi); return false;
	} else if (!array_init(h.observations, 4)) {
		fprintf(stderr, "init ERROR: Unable to initialize observations array.\n");
		free(h.pi); free(h.children);
		return false;
	}
	return h.initialize(alpha);
}

/**
 * Copies the HDP root node (as well as its descendants, recursively) from
 * `src` to `dst`, while adding a entries to `node_map` mapping the pointers of
 * every source node to the pointer of the corresponding destination node.
 */
template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
bool copy(
	const hdp<BaseDistribution, DataDistribution, K, V>& src,
	hdp<BaseDistribution, DataDistribution, K, V>& dst,
	hash_map<const node<K, V>*, node<K, V>*>& node_map)
{
	dst.depth = src.depth;
	dst.log_alpha = src.log_alpha;
	dst.alpha = (V*) malloc(sizeof(V) * src.depth);
	if (dst.alpha == NULL) {
		fprintf(stderr, "copy ERROR: Insufficient memory for alpha in hdp.\n");
		return false;
	}
	memcpy(dst.alpha, src.alpha, sizeof(V) * src.depth);
	if (!copy(src.pi, dst.pi)) {
		free(dst.alpha);
		return false;
	} else if (!array_map_init(dst.children, (unsigned int) src.children.capacity)) {
		free(dst.alpha); free(dst.pi);
		return false;
	} else if (!array_init(dst.observations, core::max(1u, (unsigned int) src.observations.length))) {
		free(dst.alpha); free(dst.pi);
		free(dst.children);
		return false;
	}

	for (unsigned int i = 0; i < src.observations.length; i++)
		dst.observations.add(src.observations[i]);
	for (unsigned int i = 0; i < src.children.size; i++) {
		dst.children.keys[i] = src.children.keys[i];
		if (!copy(src.children.values[i], dst.children.values[i], node_map))
			return false; /* TODO: free memory */
		dst.children.size++;
	}
	return true;
}

/**
 * Reads the given HDP root node `h` (and all descendants) from `in`.
 * \tparam BaseDistributionScribe a scribe type for which the function
 * 		`bool read(BaseDistribution&, FILE*, BaseDistributionScribe&)` is
 * 		defined.
 * \tparam AtomScribe a scribe type for which the function
 * 		`bool read(K&, FILE*, AtomScribe&)` is defined.
 * \tparam KeyScribe a scribe type for which the function
 * 		`bool read(unsigned int&, FILE*, KeyScribe&)` is defined.
 */
template<
	typename BaseDistribution, typename DataDistribution,
	typename K, typename V, typename BaseDistributionScribe,
	typename AtomScribe, typename KeyScribe>
bool read(hdp<BaseDistribution, DataDistribution, K, V>& h, FILE* in,
	BaseDistributionScribe& base_reader, AtomScribe& atom_reader, KeyScribe& key_reader)
{
	bool success = true;
	success &= read(h.depth, in);
	success &= read(h.pi, in, base_reader);
	if (!success) return false;

	h.alpha = (V*) malloc(sizeof(V) * h.depth);
	if (h.alpha == NULL) return false;

	if (!read(h.alpha, in, h.depth)) {
		free(h.alpha);
		return false;
	}
	h.log_alpha = log(h.alpha[0]);

	if (!read_node(h, in, atom_reader, key_reader)) {
		free(h.alpha);
		return false;
	}
	return true;
}

/**
 * Writes the given HDP root node `h` (and all descendants) to `out`.
 * \tparam BaseDistributionScribe a scribe type for which the function
 * 		`bool write(const BaseDistribution&, FILE*, BaseDistributionScribe&)`
 * 		is defined.
 * \tparam AtomScribe a scribe type for which the function
 * 		`bool write(const K&, FILE*, AtomScribe&)` is defined.
 * \tparam KeyScribe a scribe type for which the function
 * 		`bool write(unsigned int, FILE*, KeyScribe&)` is defined.
 */
template<
	typename BaseDistribution, typename DataDistribution,
	typename K, typename V, typename BaseDistributionScribe,
	typename AtomScribe, typename KeyScribe>
bool write(const hdp<BaseDistribution, DataDistribution, K, V>& h, FILE* out,
	BaseDistributionScribe& base_writer, AtomScribe& atom_writer, KeyScribe& key_writer)
{
	bool success = true;
	success &= write(h.depth, out);
	success &= write(h.pi, out, base_writer);
	success &= write(h.alpha, out, h.depth);
	if (!success) return false;

	return write_node(h, out, atom_writer, key_writer);
}

template<typename BaseDistribution, typename DataDistribution,
	typename K, typename V, typename AtomPrinter, typename KeyPrinter>
void print(const hdp<BaseDistribution, DataDistribution, K, V>& h,
		FILE* out, AtomPrinter& atom_printer, KeyPrinter& key_printer)
{
	if (h.table_count == 0)
		fprintf(out, "This HDP has no clusters at the root.\n");
	print_node(h, out, 0, h.table_count, atom_printer, key_printer);
}

template<typename NodeType>
bool add(NodeType& n, const typename NodeType::value_type* alpha,
	const unsigned int* path, unsigned int length,
	const typename NodeType::atom_type& observation)
{
	typedef typename NodeType::atom_type K;
	typedef typename NodeType::value_type V;

	if (length == 0) {
		if (!n.observations.add(observation)) {
			fprintf(stderr, "add ERROR: Unable to add new observation.\n");
			return false;
		}
		return true;
	}

	if (!n.children.ensure_capacity((unsigned int) n.children.size + 1)) {
		fprintf(stderr, "add ERROR: Unable to expand children map.\n");
		return false;
	}

	unsigned int index = strict_linear_search(n.children.keys, *path, 0, n.children.size);
	if (index > 0 && n.children.keys[index - 1] == *path) {
		return add(n.children.values[index - 1], alpha + 1, path + 1, length - 1, observation);
	} else {
		shift_right(n.children.keys, (unsigned int) n.children.size, index);
		shift_right(n.children.values, (unsigned int) n.children.size, index);
		node<K, V>& child = n.children.values[index];
		if (!init(child, *alpha)) {
			fprintf(stderr, "add ERROR: Error creating new child.\n");
			return false;
		}
		n.children.keys[index] = *path;
		n.children.size++;
		/* TODO: this insertion algorithm can be made more efficient */
		return add(child, alpha + 1, path + 1, length - 1, observation);
	}
}

/**
 * Adds the given `observation` to the HDP hierarchy rooted at `h` to the node
 * at the end of the `path`. The `unsigned int` array `path`, with length
 * `depth - 1`, provides the indices to follow from each node to its child, in
 * order to locate the node to which the `observation` is added. If the node
 * does not exist, it is created, with its concentration parameter copied from
 * the appropriate index in hdp::alpha. Thus, `depth` cannot be larger than
 * `hdp::depth` of `h`. If `depth == 1`, the observation is added to the
 * root node.
 */
template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
inline bool add(hdp<BaseDistribution, DataDistribution, K, V>& h,
		const unsigned int* path, unsigned int depth, const K& observation)
{
	return add(h, h.alpha + 1, path, depth - 1, observation);
}

/**
 * Returns `true` if and only if the given `observation` is contained in the
 * HDP node at the end of the given `path`. The `unsigned int` array `path`,
 * with given `length`, provides the indices to follow from each node to its
 * child. The node at the end of this path is checked to determine if it
 * contains the given `observation`. Note that the path may contain
 * ::IMPLICIT_NODE elements to specify a *set* of paths. In this case, all
 * nodes at the end of a path in this set are checked.
 */
template<typename NodeType>
bool contains(NodeType& n,
	const unsigned int* path, unsigned int length,
	const typename NodeType::atom_type& observation)
{
	if (length == 0) return n.observations.contains(observation);

	if (*path == IMPLICIT_NODE) {
		for (unsigned int i = 0; i < n.children.size; i++)
			if (contains(n.children.values[i], path + 1, length - 1, observation)) return true;
		return false;
	} else {
		unsigned int index = strict_linear_search(n.children.keys, *path, 0, n.children.size);
		if (index > 0 && n.children.keys[index - 1] == *path)
			return contains(n.children.values[index - 1], path + 1, length - 1, observation);
		else return false;
	}
}

#endif /* HDP_H_ */
