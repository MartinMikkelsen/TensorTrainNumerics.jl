# Contributing to TensorTrainNumerics.jl

Contributions to code, documentation, examples, and tests are welcome. Our practices are inspired by [ColPrac](https://github.com/SciML/ColPrac).

## Questions, bugs, and feature requests

Search the [existing issues](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl/issues) before opening a new one. Use issues for support questions and feature suggestions too.

For bugs, include a minimal reproducible example, expected and actual behavior, and your Julia and package versions.

## Pull requests

- Fork the repository and create a branch from `main`.

- Keep changes focused and follow the existing code style.

- Add tests for new functionality and bug fixes, and update relevant documentation.

- Discuss substantial features or breaking API changes in an issue first.

Run the tests from the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

For documentation changes, also build the documentation:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The maintainer reviews pull requests and manages releases.
