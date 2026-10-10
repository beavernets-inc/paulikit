/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com> */

/* Pass-1 bit-indexed scatter: stream H by row-block into one spill tile.
 *
 * WHY THIS EXISTS. Spilled dense construction writes gather tiles
 * G[x, q] = H[x⊕q, q] into files of height R = spill_bucket_rows.
 * The Python/NumPy path did, for each operator row p and each tile,
 * a full-dim XOR + boolean mask + fancy-index store (~262k calls and
 * ~15 s of scatter alone at n=14, R=1024). That work is arithmetic
 * and indexing, not H I/O — windowing cannot help it.
 *
 * BIT ROUTING (R = 2^k). For a tile covering x ∈ [x_lo, x_lo+height)
 * with x_lo = b·R and R a power of two, the unique q that lands row p
 * into local row r is
 *
 *     q = p ⊕ (x_lo + r)
 *
 * (Hacker's Delight: power-of-two test (R & (R-1)) == 0; bucket id is
 * x >> ctz(R) and local row is x & (R-1). Division is never needed.)
 * Each tile cell (r, q) is written by exactly one p, so an OpenMP
 * parallel-for over the p-block is race-free on the destination.
 *
 * WHAT IS COMPUTED, matching the Python bit reference: for each
 * p = p_start .. p_start+n_p-1 and each r in [0, height),
 *
 *     bucket[r, p ⊕ (x_lo+r)] = H_block[p - p_start, p ⊕ (x_lo+r)]
 *
 * MEMORY. Allocates nothing; touches only the buffers given. GIL is
 * released around the call from Cython.
 */

#ifndef PAULIKIT_PASS1_SCATTER_H
#define PAULIKIT_PASS1_SCATTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Scatter one contiguous block of operator rows into one spill tile.
 *
 * `block_data` — n_p * dim complex128 values, row-major (pairs of
 *                doubles), C-contiguous. Read only.
 * `dim`        — power of two, >= 1.
 * `p_start`    — first operator row index represented in `block_data`.
 * `n_p`        — number of rows in the block (>= 0).
 * `x_lo`       — first gather-row x covered by this tile (b * R).
 * `height`     — tile height (1 .. R); may be < R on a short last tile.
 * `bucket_data`— height * dim complex128, row-major; cells written
 *                by this call are assigned exactly once across a full
 *                Pass-1 over all p (caller must zero the tile first
 *                if unread cells must stay zero).
 *
 * No power-of-two check on R here: the Python layer enforces
 * spill_bucket_rows = 2^k before calling. `x_lo` and `height` must
 * describe a half-open interval inside [0, dim).
 */
void paulikit_pass1_scatter_block_into_bucket(
    const double *block_data,
    int64_t dim,
    int64_t p_start,
    int64_t n_p,
    int64_t x_lo,
    int64_t height,
    double *bucket_data
);

/* Fill one spill tile from a full dense operator (RAM or memmap).
 *
 * For every p in [0, dim) and r in [0, height):
 *     bucket[r, p ⊕ (x_lo+r)] = operator[p, p ⊕ (x_lo+r)]
 *
 * Across all tiles covering [0, dim), each operator cell is read
 * exactly once — no Pass-1 re-read of H. OpenMP-scheduled over p.
 */
void paulikit_pass1_fill_bucket_from_operator(
    const double *operator_data,
    int64_t dim,
    int64_t x_lo,
    int64_t height,
    double *bucket_data
);

/* Write a contiguous complex128 buffer to `path` (create/truncate).
 *
 * Uses large POSIX write() chunks (and posix_fallocate when available)
 * so Pass-1 spill I/O is not limited by NumPy tofile's stdio path.
 * Returns 0 on success, -1 on failure with errno set.
 *
 * `n_complex` is the number of complex128 values (height * dim).
 * `data` is read-only for the duration of the call.
 */
int paulikit_spill_write_c128(
    const char *path,
    const double *data,
    int64_t n_complex
);

#ifdef __cplusplus
}
#endif

#endif /* PAULIKIT_PASS1_SCATTER_H */
