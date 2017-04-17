/**
 * hdp.h - Hierarchical Dirichlet process implementation.
 *
 *  Created on: Jul 10, 2016
 *      Author: asaparov
 */

#ifndef HDP_H_
#define HDP_H_

#include <core/map.h>
#include <math/histogram.h>
#include <limits.h>

#define IMPLICIT_NODE UINT_MAX
#define UNION_NODE (UINT_MAX - 3)

using namespace core;

/* forward declarations */
template<typename K, typename V> struct node;
template<typename BaseDistribution, typename DataDistribution, typename K, typename V> struct hdp;

template<typename K, typename V>
struct node {
	typedef K atom_type;
	typedef V value_type;

	array_map<unsigned int, node<K, V>> children;

	V alpha;
	V log_alpha;
	array<K> observations;

	node(const V& alpha, unsigned int table_count, unsigned int table_capacity) :
		children(4), alpha(alpha), log_alpha(log(alpha)), observations(4)
	{ }

	~node() { free(); }

	inline V get_alpha() const {
		return alpha;
	}

	inline V get_log_alpha() const {
		return log_alpha;
	}

	static inline void move(const node<K, V>& src, node<K, V>& dst) {
		dst.alpha = src.alpha;
		dst.log_alpha = src.log_alpha;
		core::move(src.children, dst.children);
		core::move(src.observations, dst.observations);
	}

	template<typename Metric>
	static inline long unsigned int size_of(const node<K, V>& n, const Metric& metric) {
		long unsigned int sum = core::size_of(n.alpha) + core::size_of(n.log_alpha)
			+ core::size_of(n.children, make_key_value_metric(dummy_metric(), metric))
			+ core::size_of(n.customer_count) + core::size_of(n.observations, metric)
			+ core::size_of(n.table_count) + core::size_of(n.table_capacity)
			+ core::size_of(n.posterior);

		for (unsigned int i = 0; i < n.table_count; i++)
			sum += core::size_of(n.descendant_observations[i], metric);
		sum += sizeof(array_histogram<K>) * (n.table_capacity - n.table_count);

		sum += 3 * sizeof(unsigned int) * n.table_capacity;
		return sum;
	}

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

template<typename K, typename V>
inline bool init(node<K, V>& n, const V& alpha,
		unsigned int table_count, unsigned int table_capacity)
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

template<typename AtomScribe, typename KeyScribe>
struct node_scribe {
	AtomScribe& atom_scribe;
	KeyScribe& key_scribe;

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
	return true;
}

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

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
struct hdp {
	typedef K atom_type;
	typedef V value_type;
	typedef BaseDistribution base_distribution_type;
	typedef DataDistribution data_distribution_type;

	BaseDistribution pi;

	unsigned int depth;

	V* alpha;
	V log_alpha;
	array_map<unsigned int, node<K, V>> children;

	array<K> observations;

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

	inline V get_alpha() const {
		return alpha[0];
	}

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
		if (!init(child, *alpha, 1, 8)) {
			fprintf(stderr, "add ERROR: Error creating new child.\n");
			return false;
		}
		n.children.keys[index] = *path;
		n.children.size++;
		/* TODO: this insertion algorithm can be made more efficient */
		return add(child, alpha + 1, path + 1, length - 1, observation);
	}
}

template<typename BaseDistribution, typename DataDistribution, typename K, typename V>
inline bool add(hdp<BaseDistribution, DataDistribution, K, V>& h,
		const unsigned int* path, unsigned int length, const K& observation)
{
	return add(h, h.alpha + 1, path, length - 1, observation);
}

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
