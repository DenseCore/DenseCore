# Contributing to DenseCore

We welcome contributions! Please follow these guidelines to keep the project clean and consistent.

## Development Setup

### Prerequisites
- **C++**: CMake 3.14+, GCC 9+/Clang 10+
- **Go**: 1.22+ (for server)
- **Python**: 3.8+

### Build Instructions

```bash
# 1. Clone & Init Submodules
git clone https://github.com/YOUR_USERNAME/DenseCore.git
cd DenseCore
git submodule update --init --recursive

# 2. Build C++ Core
make lib

# 3. Install Python Package (editable)
cd python
pip install -e ".[dev]"

# 4. Build Server (Optional)
make server
```

>**Note**: Default build uses `-march=native`. For portable builds, check CMake options.

## Pull Request Process

1.  **Fork** the repo and create a feature branch.
2.  **Commit** with clear messages (e.g., `feat: add nucleus sampling`).
3.  **Tests**: Ensure `make test` passes.
4.  **Submit PR** to `main`.

## Code Style

- **C++**: Google Style (use `clang-format`).
- **Python**: PEP 8 (use `black`, `ruff`).
- **Go**: `go fmt`.

## Reporting Issues

Please use the [GitHub Issue Tracker](https://github.com/Jake-Network/DenseCore/issues). Include reproduction steps and environment details.

## License

By contributing, you agree that your code will be licensed under the Apache 2.0 License.
