# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Pauli decomposition algorithm implementations.

``fwht`` holds the decomposition itself, based on the Fast
Walsh-Hadamard Transform, O(N^2 log N) for an N x N matrix.
``autotune`` sizes chunks against the machine's measured cache
hierarchy and available memory.

Opt-in dense out-of-core input (do not import on the hot path unless
needed):

- ``operator_source`` — ``gather_chunk`` adapters (resident / sparse)
- ``dense_input`` — layout A/B validation (``resolve_dense_file``)
- ``dense_bucketed`` — spill buckets and ``from_dense_file``

See ``docs/dense_out_of_core.md`` in the documentation set.
"""
