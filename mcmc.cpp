/**
 * \file mcmc.cpp
 *
 *  <!-- Created on: Jul 31, 2015
 *           Author: asaparov -->
 */

#include "mcmc.h"

#include <core/timer.h>

template<typename V>
void test_log_rising_factorial() {
	unsigned int ITERATIONS = 1000;
	unsigned int MAX_EXPONENT = 10000;

	V* bases = (V*) malloc(sizeof(V) * ITERATIONS);
	unsigned int* exponents = (unsigned int*) malloc(sizeof(unsigned int) * ITERATIONS);
	V* results = (V*) malloc(sizeof(V) * ITERATIONS);

	printf("test_log_rising_factorial: Beginning test.\n");
	fflush(stdout);

	timer stopwatch;
	for (unsigned int i = 0; i < ITERATIONS; i++) {
		bases[i] = (double) rand() / RAND_MAX;
		exponents[i] = rand() % MAX_EXPONENT;
		results[i] = log_rising_factorial(bases[i], exponents[i]);
	}

	printf("test_log_rising_factorial: Test completed in %lf ms.\n", stopwatch.nanoseconds() / 10000000);
	printf("Results:\n");
	for (unsigned int i = 0; i < ITERATIONS; i++)
		printf("  %lf^(%u) = %lf\n", bases[i], exponents[i], results[i]);
}

int main(int argc, const char** argv) {
	srand(get_seed());
	printf("(seed = %u)\n", get_seed());
	if (!hdp_test<double>()) {
		fprintf(stderr, "hdp_test failed.\n");
		return EXIT_FAILURE;
	}
	printf("hdp_test completed.\n");
	fflush(stdout);
	return EXIT_SUCCESS;
}
