# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Internal package: the optional compiled extensions.

Several extensions may be built here, gated by three feature options
in ``meson.options``::

    wht_kernel   ``wht_native``              Walsh-Hadamard butterfly
                 ``wht_native_v3``           same, -march=x86-64-v3
                 ``coeffs_native``           phase, scaling, threshold,
                                             index emission
                 ``coeffs_native_v3``        same, -march=x86-64-v3
                 ``gather_native``           dense XOR-row gather
                 ``pass1_scatter_native``    Pass-1 spill bit-scatter
                                             + large POSIX spill write
                 ``hermitian_check_native``  fused Hermiticity check
    native       ``pauli_label_native``           serial Pauli label strings
                 ``pauli_label_parallel_native``  oneTBB fill (optional)
    cache_probe  ``cache_probe``                  empirical cache sizing

Which are present depends on the toolchain available at build time.
The ``wht_kernel``, ``cache_probe``, and serial ``pauli_label_native``
modules need only a C compiler and Cython. The parallel label module
additionally needs C++ and oneTBB and is omitted from manylinux
wheels on purpose (e2e ``list[str]`` labeling is dominated by Python
string construction, not the C fill).

The ``_v3`` modules are the SAME source as their non-``_v3``
counterpart, compiled a second time under ``-march=x86-64-v3``
(AVX2 + BMI2 + FMA + POPCNT) when the build toolchain accepts that
flag. ``fwht.py`` picks between a module and its ``_v3`` twin at
IMPORT time, based on a runtime check of the CPU actually running the
process (not the machine that built the wheel) - never at build time,
so a wheel built on a newer CPU stays correct (just slower) on an
older one it later runs on.

Nothing outside ``paulikit.algorithms`` should import these directly.
Callers go through the public API, which handles a missing extension
by falling back - to NumPy for the kernels, and to declared cache
sizes for the probe.
"""
