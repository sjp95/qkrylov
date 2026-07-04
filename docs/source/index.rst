Welcome to qkrylov's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorial
   api
   examples

Overview
--------

**qkrylov** is a high-performance C++20 library for matrix-free Krylov methods in quantum many-body physics.
It allows for the study of large-scale quantum systems by implementing the Hamiltonian action :math:`y = Hx`
without explicit matrix construction.

Key Features
------------

* **Multiple Physics Models**: Spin-Half, Fermions, Hubbard, and t-J models.
* **Symmetry Conservation**: Automated basis construction for various quantum number sectors.
* **Robust Solvers**: State-of-the-art Lanczos and Davidson implementations.
* **Spectral Functions**: Built-in support for dynamical structure factors.
* **Python Interface**: Easy-to-use Python bindings for rapid development.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
