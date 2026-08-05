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

Project Milestone: Final Implementation
---------------------------------------

This version marks the finalization of the **qkrylov** project core, providing a fully functional, high-performance framework.

Key Achievements
~~~~~~~~~~~~~~~~

1. **Diverse Physics**: Support for Spin-Half, spinless Fermion, Hubbard, and t-J models, including complex multi-site interactions like 3-spin DMI.
2. **Stable Numerical Solvers**: Lanczos solver with full reorthogonalization for guaranteed convergence. Includes Davidson, Continued Fraction (Dynamics), and FTLM solvers.
3. **Advanced Spectroscopy**: Tools for calculating momentum-resolved higher-order RIXS dynamical susceptibility :math:`S(q, \omega)`.
4. **Seamless Python Integration**: Native Python bindings via **nanobind** with a highly concise and user-friendly syntax.
5. **Professional Documentation**: Comprehensive Sphinx-based documentation including a beginners' tutorial, API reference, and physics examples.
6. **Data Storage & Interop**: Persistence via HDF5 and scaffolding for Julia interop.

Key Features
------------

* **Multiple Physics Models**: Spin-Half, Fermions, Hubbard, and t-J models.
* **Symmetry Conservation**: Automated basis construction for various quantum number sectors.
* **Robust Solvers**: State-of-the-art Lanczos and Davidson implementations.
* **Spectral Functions**: Built-in support for dynamical structure factors and RIXS.
* **Python Interface**: Easy-to-use Python bindings for rapid development.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
