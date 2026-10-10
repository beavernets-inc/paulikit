/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com> */

/* Bit-indexed Pass-1 scatter + spill write. See pass1_scatter.h. */

#include "pass1_scatter.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Large sequential write chunks — measured Pass-1 was dominated by
 * NumPy tofile on 256 MiB tiles; one write() syscall stream hits the
 * disk closer to sequential bandwidth. */
#ifndef PAULIKIT_SPILL_WRITE_CHUNK
#define PAULIKIT_SPILL_WRITE_CHUNK ((size_t)16 << 20)  /* 16 MiB */
#endif

void paulikit_pass1_scatter_block_into_bucket(
    const double *block_data,
    int64_t dim,
    int64_t p_start,
    int64_t n_p,
    int64_t x_lo,
    int64_t height,
    double *bucket_data
) {
    if (dim < 1 || n_p < 1 || height < 1) {
        return;
    }

    /* Complex128 as adjacent double pairs — same convention as gather.c
     * / wht.c so the compiler sees a long contiguous store stream. */
    const int64_t dim_doubles = dim * 2;

    /* Schedule over p: each destination cell (r, q) has a unique writer
     * p = (x_lo+r) ⊕ q, so concurrent stores do not collide. Static
     * chunks keep each thread on a contiguous strip of H rows. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_p >= 32)
#endif
    for (int64_t i = 0; i < n_p; i++) {
        const int64_t p = p_start + i;
        const double *restrict row = block_data + i * dim_doubles;

        for (int64_t r = 0; r < height; r++) {
            const int64_t q = p ^ (x_lo + r);
            double *restrict dst = bucket_data + (r * dim + q) * 2;
            const double *restrict src = row + q * 2;
            dst[0] = src[0];
            dst[1] = src[1];
        }
    }
}

void paulikit_pass1_fill_bucket_from_operator(
    const double *operator_data,
    int64_t dim,
    int64_t x_lo,
    int64_t height,
    double *bucket_data
) {
    /* Full p-range: same kernel as the block scatter with p_start=0,
     * n_p=dim, block_data=operator_data. */
    paulikit_pass1_scatter_block_into_bucket(
        operator_data, dim, /*p_start=*/0, /*n_p=*/dim,
        x_lo, height, bucket_data
    );
}

int paulikit_spill_write_c128(
    const char *path,
    const double *data,
    int64_t n_complex
) {
    if (path == NULL || (n_complex > 0 && data == NULL) || n_complex < 0) {
        errno = EINVAL;
        return -1;
    }

    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0) {
        return -1;
    }

    /* complex128 = 16 bytes. */
    const size_t nbytes = (size_t)n_complex * (size_t)16;

#ifdef POSIX_FADV_SEQUENTIAL
    if (nbytes > 0) {
        (void)posix_fadvise(fd, 0, (off_t)nbytes, POSIX_FADV_SEQUENTIAL);
    }
#endif

    const char *p = (const char *)data;
    size_t left = nbytes;
    while (left > 0) {
        size_t chunk = left > PAULIKIT_SPILL_WRITE_CHUNK
            ? PAULIKIT_SPILL_WRITE_CHUNK
            : left;
        ssize_t n = write(fd, p, chunk);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            int saved = errno;
            close(fd);
            errno = saved;
            return -1;
        }
        if (n == 0) {
            close(fd);
            errno = EIO;
            return -1;
        }
        p += (size_t)n;
        left -= (size_t)n;
    }

    if (close(fd) != 0) {
        return -1;
    }
    return 0;
}
