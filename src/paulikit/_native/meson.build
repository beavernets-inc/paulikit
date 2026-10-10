# Serial pauli_label fast path (C + Cython). Optional via meson.options
# "native". Deliberately independent of oneTBB: the serial C kernel is
# what wheels and most installs need; the TBB parallel fill is a
# separate optional module below. fwht.py falls back to pure Python
# with a visible warning if pauli_label_native is absent.

native_opt = get_option('native')

# have_cython / have_c / have_cpp come from add_languages() in the
# top-level meson.build. Guarding on them matters: meson.get_compiler
# is a hard error when the language was never successfully added.
if not native_opt.disabled() and have_cython and have_c
  cy = meson.get_compiler('cython')
  cython_ok = cy.version().version_compare('>=3.0')

  if cython_ok
    np_include_path = run_command(
      py, ['-c', 'import numpy; print(numpy.get_include())'],
      check: true,
    ).stdout().strip()

    # Serial module: C only — ships in manylinux wheels.
    py.extension_module(
      'pauli_label_native',
      [
        'pauli_label_native.pyx',
        'pauli_label.c',
      ],
      include_directories: include_directories('.', np_include_path, is_system: true),
      install: true,
      subdir: 'paulikit/_native',
    )

    # Optional oneTBB parallel fill. Not required for -Dnative=enabled;
    # serial alone satisfies the option. Built when C++ and TBB exist.
    if have_cpp
      tbb_dep = dependency('tbb', required: false)
      if tbb_dep.found()
        py.extension_module(
          'pauli_label_parallel_native',
          [
            'pauli_label_parallel_native.pyx',
            'pauli_label_parallel.cpp',
          ],
          include_directories: include_directories('.', np_include_path, is_system: true),
          dependencies: [tbb_dep],
          override_options: ['cython_language=cpp'],
          install: true,
          subdir: 'paulikit/_native',
        )
      else
        message('paulikit: pauli_label_parallel_native not built (oneTBB not found)')
      endif
    else
      message('paulikit: pauli_label_parallel_native not built (no C++ compiler)')
    endif
  elif native_opt.enabled()
    error('native extension explicitly requested (-Dnative=enabled) but ' +
          'Cython >= 3.0 is missing (found: @0@)'.format(cython_ok))
  else
    message('paulikit: pauli_label_native not built (Cython >= 3.0 not found) ' +
            '- falling back to pure Python at import time')
  endif
elif native_opt.disabled()
  message('paulikit: native extension disabled via -Dnative=disabled')
else
  message('paulikit: pauli_label_native not built (C compiler or Cython unavailable)')
endif

# The Walsh-Hadamard butterfly kernel - the compiled fast path for
# the transform itself, as opposed to the label generation above.
# Deliberately independent of the `native` option: this needs only a C
# compiler and Cython, with no C++ and no oneTBB, so a machine that
# cannot build pauli_label_native can still get the kernel that
# carries most of the per-chunk work. fwht.py falls back to the
# pure-Python butterfly if it is absent.
wht_opt = get_option('wht_kernel')

# x86-64-v3 (AVX2 + BMI2 + FMA + POPCNT) compiler flag, used below to
# build a second copy of the WHT and coefficient kernels alongside the
# baseline ones. Same source, compiled twice under different module
# names - fwht.py picks between them at import time based on a runtime
# CPU-flags check (see _cpu_has_x86_64_v3 there), never at build time,
# since a wheel built on one machine may run on an older one. Checked
# with has_multi_arguments rather than assumed available: some cross
# compilers or exotic toolchains may not support this flag at all, and
# that must degrade to "just don't build the _v3 variant" rather than
# fail the whole build - the baseline kernel alone is still correct
# and was the only option before this change existed.
v3_args = ['-march=x86-64-v3']
have_v3_isa = false
if not wht_opt.disabled() and have_c
  cc = meson.get_compiler('c')
  have_v3_isa = cc.has_multi_arguments(v3_args)
endif

if not wht_opt.disabled() and have_cython and have_c
  cy3 = meson.get_compiler('cython')
  cython_ok3 = cy3.version().version_compare('>=3.0')

  if cython_ok3
    np_include_wht = run_command(
      py, ['-c', 'import numpy; print(numpy.get_include())'],
      check: true,
    ).stdout().strip()

    py.extension_module(
      'wht_native',
      [
        'wht_native.pyx',
        'wht.c',
      ],
      include_directories: include_directories('.', np_include_wht, is_system: true),
      install: true,
      subdir: 'paulikit/_native',
    )

    if have_v3_isa
      # Cython embeds the extension's PyInit_<name> symbol from the
      # .pyx module name at Cython-compile time, not from the meson
      # target name below - compiling the same wht_native.pyx twice
      # without this would produce two .so files that both export
      # PyInit_wht_native, so only whichever one Python's import
      # machinery finds first would actually load (found the hard
      # way: the CPU-capable branch silently imported the baseline
      # module). --module-name overrides what Cython embeds so the
      # resulting symbol matches this target's real name.
      py.extension_module(
        'wht_native_v3',
        [
          'wht_native.pyx',
          'wht.c',
        ],
        cython_args: ['--module-name', 'wht_native_v3'],
        c_args: v3_args,
        include_directories: include_directories('.', np_include_wht, is_system: true),
        install: true,
        subdir: 'paulikit/_native',
      )
    else
      message('paulikit: wht_native_v3 not built (compiler does not ' +
              'accept -march=x86-64-v3) - falling back to wht_native ' +
              'at import time on CPUs that support it')
    endif
  elif wht_opt.enabled()
    error('WHT kernel explicitly requested (-Dwht_kernel=enabled) but ' +
          'Cython >= 3.0 was not found')
  else
    message('paulikit: wht_native not built (Cython >= 3.0 not found) - ' +
            'falling back to the pure-Python butterfly at import time')
  endif
else
  message('paulikit: wht_native disabled via -Dwht_kernel=disabled')
endif

# The fully-dense per-chunk XOR-gather kernel - reads
# gathered_chunk[row, q] = operator[x_row ^ q, q] directly, replacing
# the NumPy fancy-index gather paulikit.algorithms.fwht's fully-dense
# fast path used. Same C-and-Cython-only dependency set as the WHT
# kernel and shares its option: this is upstream of the transform
# (gather, then transform, then emit), so there is no case for
# building it without the other two.
if not wht_opt.disabled() and have_cython and have_c
  cy5 = meson.get_compiler('cython')

  if cy5.version().version_compare('>=3.0')
    np_include_gather = run_command(
      py, ['-c', 'import numpy; print(numpy.get_include())'],
      check: true,
    ).stdout().strip()

    py.extension_module(
      'gather_native',
      [
        'gather_native.pyx',
        'gather.c',
      ],
      include_directories: include_directories('.', np_include_gather, is_system: true),
      install: true,
      subdir: 'paulikit/_native',
    )

    # Pass-1 spill scatter (bit-indexed, optional OpenMP over p-blocks).
    # Same C+Cython dependency set as gather; OpenMP is best-effort so
    # a toolchain without it still builds a correct serial kernel.
    omp_dep = dependency('openmp', required: false)
    if not omp_dep.found()
      message('paulikit: pass1_scatter_native building without OpenMP')
    endif

    py.extension_module(
      'pass1_scatter_native',
      [
        'pass1_scatter_native.pyx',
        'pass1_scatter.c',
      ],
      include_directories: include_directories('.', np_include_gather, is_system: true),
      dependencies: omp_dep,
      install: true,
      subdir: 'paulikit/_native',
    )
  endif
endif

# The Hermiticity-violation check kernel - a single fused pass over a
# chunk's surviving coefficients, replacing four separate NumPy passes
# (np.abs, np.abs, np.maximum, compare-and-any). Same
# C-and-Cython-only dependency set as the other kernels above and
# shares their option; used by both the sparse and dense input paths
# (fwht.py falls back to the NumPy 4-pass check if this module is
# absent, with identical results either way).
if not wht_opt.disabled() and have_cython and have_c
  cy6 = meson.get_compiler('cython')

  if cy6.version().version_compare('>=3.0')
    np_include_herm = run_command(
      py, ['-c', 'import numpy; print(numpy.get_include())'],
      check: true,
    ).stdout().strip()

    py.extension_module(
      'hermitian_check_native',
      [
        'hermitian_check_native.pyx',
        'hermitian_check.c',
      ],
      include_directories: include_directories('.', np_include_herm, is_system: true),
      install: true,
      subdir: 'paulikit/_native',
    )
  endif
endif

# The fused coefficient kernel - phase factor,
# scaling, thresholding and index emission in one pass. Same
# C-and-Cython-only dependency set as the WHT kernel above, and shares
# its option: the two are a pair (transform, then emit) and there is
# no case for building one without the other.
if not wht_opt.disabled() and have_cython and have_c
  cy4 = meson.get_compiler('cython')

  if cy4.version().version_compare('>=3.0')
    np_include_coeffs = run_command(
      py, ['-c', 'import numpy; print(numpy.get_include())'],
      check: true,
    ).stdout().strip()

    py.extension_module(
      'coeffs_native',
      [
        'coeffs_native.pyx',
        'coeffs.c',
      ],
      include_directories: include_directories('.', np_include_coeffs, is_system: true),
      install: true,
      subdir: 'paulikit/_native',
    )

    # Second copy at x86-64-v3: coeffs.c's popcount macro
    # (__builtin_popcountll) only lowers to the real POPCNT
    # instruction when the compiler is told the target has it: this
    # flag is what makes that happen. Measured directly this session:
    # ~44% fewer instructions, ~39-59% fewer cycles than the
    # SWAR-emulated baseline at this kernel's real per-chunk shape -
    # the single largest compiled-kernel win found in this
    # investigation. have_v3_isa is computed once above, shared with
    # the WHT kernel's own _v3 variant.
    if have_v3_isa
      # See wht_native_v3's own comment above: --module-name is
      # required so this variant's compiled-in PyInit_ symbol matches
      # its own target name rather than colliding with coeffs_native's.
      py.extension_module(
        'coeffs_native_v3',
        [
          'coeffs_native.pyx',
          'coeffs.c',
        ],
        cython_args: ['--module-name', 'coeffs_native_v3'],
        c_args: v3_args,
        include_directories: include_directories('.', np_include_coeffs, is_system: true),
        install: true,
        subdir: 'paulikit/_native',
      )
    else
      message('paulikit: coeffs_native_v3 not built (compiler does not ' +
              'accept -march=x86-64-v3) - falling back to coeffs_native ' +
              'at import time on CPUs that support it')
    endif
  endif
endif

# The empirical cache-latency probe - standalone,
# Cython-only, deliberately independent of the native/oneTBB option
# above (the probe has no relationship to oneTBB; see cache_probe.pyx's
# own docstring for why bundling them would be wrong).
cache_probe_opt = get_option('cache_probe')

if not cache_probe_opt.disabled() and have_cython and have_c
  cy2 = meson.get_compiler('cython')
  cython_ok2 = cy2.version().version_compare('>=3.0')

  if cython_ok2
    py.extension_module(
      'cache_probe',
      [
        'cache_probe.pyx',
        'cache_probe.c',
      ],
      include_directories: include_directories('.'),
      install: true,
      subdir: 'paulikit/_native',
    )
  elif cache_probe_opt.enabled()
    error('cache_probe extension explicitly requested (-Dcache_probe=enabled) but ' +
          'Cython >= 3.0 was not found')
  else
    message('paulikit: cache_probe extension not built (Cython >= 3.0 not found) - ' +
            'falling back to declared-size cache detection at runtime')
  endif
else
  message('paulikit: cache_probe extension disabled via -Dcache_probe=disabled')
endif

# The __init__.py for the _native package itself (plain Python,
# always installed regardless of whether the extension built).
py.install_sources(
  '__init__.py',
  subdir: 'paulikit/_native',
)
