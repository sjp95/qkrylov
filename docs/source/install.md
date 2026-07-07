# Installation Guide

## System Requirements

- **Compiler**: A C++20 compliant compiler.
    - GCC 11+
    - Clang 13+
    - MSVC 19.30+
- **Build System**: CMake 3.20 or newer.
- **Python**: version 3.10 or newer (optional, for bindings).

## Building from Source

To build the C++ library and the test suite, run the following commands in the repository root:

```bash
mkdir build
cd build
cmake ..
make -j
```

Alternatively, use the provided `Makefile`:

```bash
make build
```

## Python Bindings

The Python bindings require the `nanobind` package.

1. **Install nanobind**:
   ```bash
   pip install nanobind
   ```

2. **Install qkrylov**:
   ```bash
pip install ./bindings/python
```

## System-wide C++ Installation

To install the library and headers to your system's include and library paths (usually `/usr/local`):

```bash
cmake --build build --target install
   ```

## Running Tests

### C++ Tests
```bash
make test
```

### Python Tests
```bash
pytest bindings/python/tests/test_basic.py
```
