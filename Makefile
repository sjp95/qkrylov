.PHONY: help build test docs clean

help:
	@echo "qkrylov development commands:"
	@echo "  build        Build the C++ library and tests"
	@echo "  test         Run C++ tests"
	@echo "  docs         Build the documentation"
	@echo "  clean        Remove build artifacts"

build:
	cmake -B build -S .
	cmake --build build

test: build
	cd build && ctest --output-on-failure

docs:
	sphinx-build -b html docs/source docs/build

clean:
	rm -rf build docs/build
