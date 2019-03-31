Documentation and code examples are available at <https://asaparov.org/docs/hdp>. The repository is located at <https://github.com/asaparov/hdp>.

This repository implements data structures to represent Dirichlet processes and hierarchical Dirichlet processes (see <a href="https://asaparov.org/docs/hdp/hdp.h.html">hdp.h</a> for an overview of these concepts). In addition, this repository provides a Markov chain Monte Carlo sampler to perform inference in [mcmc.h](https://asaparov.org/docs/hdp/mcmc.h.html).

### Dependencies

To use the code, simply download the files into a folder named "hdp". Add this folder to the include path for the compiler, for example by using the `-I` flag.

This library depends on [core](https://github.com/asaparov/core) and [math](https://github.com/asaparov/math). The code makes use of `C++11` and is regularly tested with `gcc 8` but I have previously compiled it with `gcc 4.8`, `clang 4.0`, and `Microsoft Visual C++ 14.0 (2015)`. The code is intended to be platform-independent, so please create an issue if there are any compilation bugs.

### Examples

There is an example in the documentation for [mcmc.h](https://asaparov.org/docs/hdp/mcmc.h.html), as well as a test program in `mcmc.cpp` which can be built using `make hdp_mcmc`.
