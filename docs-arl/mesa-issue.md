# anv/brw: llama.cpp MoE matmul shader over-unrolls into a 5900-spill kernel, engine reset / VK_ERROR_DEVICE_LOST on Arrow Lake (Arc 140T, xe)

## System information

```
System:
  Kernel 7.1.4-1-cachyos arch x86_64 bits 64 compiler clang v 22.1.8
  Desktop Hyprland v 0.56.0 dm N/A Distro CachyOS base Arch Linux
CPU:
  Info 16-core model Intel Core Ultra 9 285H bits 64 type MCP arch Arrow Lake rev 2
Graphics:
  Device-1 Intel Arrow Lake-P [Arc Pro 130T/140T] vendor Lenovo driver xe v kernel
    arch Xe2-LPG ports active eDP-1 bus-ID 00:02.0 chip-ID 8086:7d51
  Display wayland server Xwayland v 24.1.13 compositor Hyprland v 0.56.0 driver gpu xe
  API OpenGL v 4.6 vendor intel mesa v 26.1.5-arch3.1 renderer Mesa Intel Graphics (ARL)
    device-ID 8086:7d51
  API Vulkan v 1.4.350 device 0 type integrated-gpu driver mesa intel device-ID 8086:7d51
```

* OS: CachyOS (Arch Linux based)
* GPU: `00:02.0 VGA compatible controller [0300]: Intel Corporation Arrow Lake-P [Arc Pro 130T/140T] [8086:7d51] (rev 03)`
* Kernel: `Linux 7.1.4-1-cachyos #1 SMP PREEMPT_DYNAMIC x86_64` (also reproduced on 6.18.38-LTS)
* KMD: `xe`
* Mesa: `26.1.5-arch3.1` (also reproduced on 26.0.6, 26.1.0, and main @ `0f7afad`)
* Desktop/compositor: Hyprland 0.56.0 (Wayland)
* Application: llama.cpp Vulkan backend, commit `0324696b8e5fe340dc94b64714e4c9aab03084a2`

## Describe the issue

A compute shader in llama.cpp's Vulkan backend (the quantized `mul_mat_id` tile matmul, used for
mixture-of-experts models) contains a loop marked `[[unroll]]` (SPIR-V `LoopControlUnroll`) whose
body already contains fully-unrolled nested loops. `brw` honours the unroll request and produces a
**109k-112k instruction kernel with ~5,900 register spills and ~2.4 MB of machine code**.

The resulting kernel cannot be preempted. Once the engine's preemption deadline expires the kernel
resets the engine, and the application gets `VK_ERROR_DEVICE_LOST`. Every affected model fails to
run at all, 100% reproducibly.

Building the same shader with that one loop kept rolled produces **5,990 instructions and 68
spills**, and the identical dispatch completes in milliseconds. So the shader is valid and the
workload is reasonable; the problem is the code generated for it.

Expected: the dispatch completes, as it does when that one loop is kept rolled.
Actual: engine reset and `VK_ERROR_DEVICE_LOST`, 100% of the time.

I have only Intel hardware, so I cannot say how other drivers handle this same SPIR-V.

### Reproduction

A standalone harness is attached (`repro.c`, ~200 lines, no llama.cpp dependency). It loads one
SPIR-V module, creates a compute pipeline with the exact specialization constants the application
uses, and dispatches it once with a 30 s fence wait:

```
cc -O2 -o repro repro.c -lvulkan
INTEL_DEBUG=cs MESA_SHADER_CACHE_DISABLE=1 ./repro buggy.spv
```

```
device: Intel(R) Graphics (ARL)
SIMD8 shader: 111793 instructions. 3 loops. 10864646 cycles. 5908:16243 spills:fills,
491 sends, scheduled with mode non-lifo. GRF registers: 128.
Non-SSA regs (after NIR): 6380. Compacted 2466368 to 2458976 bytes (0%)
RESULT: VK_ERROR_DEVICE_LOST (engine reset)
```

The same harness with `fixed.spv` (attached; byte-different, identical source except that one
loop is `[[dont_unroll]]`):

```
SIMD8 shader: 5990 instructions. 4 loops. 1030338 cycles. 68:185 spills:fills,
383 sends. GRF registers: 128. Non-SSA regs (after NIR): 1461. Compacted to 124992 bytes
RESULT: dispatch COMPLETED (no hang)
```

Same driver, same specialization constants, same dispatch, same buffers. Only the unroll hint
differs. (`MESA_SHADER_CACHE_DISABLE=1` is needed or the compile statistics are not re-emitted.)

Application-level reproduction, if preferred:

```
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test-backend-ops -j
build/bin/test-backend-ops test -o MUL_MAT_ID -b Vulkan0 -p 'type_a=iq3_s'
```

### The reset comes from the preemption timeout, not the job watchdog

Worth stating precisely, because it is easy to misdiagnose (I did, initially):

| `engines/rcs/preempt_timeout_us` | time from `vkQueueSubmit` to `DEVICE_LOST` |
|---|---|
| 640000 (default, 0.64 s) | 0.64 s |
| 10000000 (`preempt_timeout_max`, 10 s) | 10.01 s |

Varying `job_timeout_ms` (5000 vs 10000) has **no effect** on when the reset happens; raising
`preempt_timeout_us` moves it exactly in step. So the kernel is not simply overrunning the job
watchdog: it cannot be preempted, and the engine is reset when the preemption deadline expires.
Even at the 10 s maximum the kernel still does not finish.

### Affected Mesa versions (not a regression)

Same `buggy.spv`, same harness, four drivers:

| Mesa | instructions | spills:fills | code bytes | result |
|---|---:|---|---:|---|
| 26.0.6 | 109,277 | 5899:15706 | 2,504,640 | `VK_ERROR_DEVICE_LOST` |
| 26.1.0 | 111,793 | 5908:16243 | 2,458,976 | `VK_ERROR_DEVICE_LOST` |
| 26.1.5 (distro) | 111,793 | 5908:16243 | 2,458,976 | `VK_ERROR_DEVICE_LOST` |
| main `0f7afad` (locally built, unpatched) | 104,404 | 5907:14465 | 2,240,768 | `VK_ERROR_DEVICE_LOST` |

This is **not** a regression in the 26.1 cycle: 26.0.6 already produces a ~109k-instruction,
5,899-spill kernel and the same engine reset. Note this differs from #15609, where the RADV
over-unrolling *was* a 26.1-cycle regression.

## Regression

No. It reproduces on every Mesa version tested (26.0.6 through main) and on both kernels tested
(6.18.38-LTS and 7.1.4). I have not tested anything older than 26.0.6.

## The shader

* File: `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`, line 315:
  `[[unroll]] for (uint i = 0; i < BK / BK_STEP; i++)` (16 iterations for quantized types)
* Pipeline: `matmul_id_subgroup_iq3_s_f32_f16acc_aligned_s`
* Specialization constants (ids 0..11), captured from the application's pipeline creation:
  `[32, 32, 32, 32, 32, 32, 2, 2, 2, 1, 8, 1]`
  (`BLOCK_SIZE=32, BM=32, BN=32, BK=32, WM=32, WN=32, WMITER=2, TM=2, TN=2, ...`)
* Workgroup denominators `(32, 32, 1)`, 5 storage buffers, 13-uint push constant block

The loop body already contains fully-unrolled `WMITER x TM` and `WNITER x TN` nested loops, and the
`MUL_MAT_ID` variant keeps row-id gather addressing live across the loop. Unrolling the outer 16
iterations on top of that yields ~6,380 live vector values competing for 128 GRF, so the majority
must spill. "Spill less" and "do not fully unroll" are therefore the same lever here; this is not
register-allocator tuning.

Not specific to IQ quantizations: they merely always route through this shader because they have no
integer-dot `mul_mat_id` variant. Forcing a K-quant down the same path
(`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 ... -p 'type_a=q4_K'`) reproduces it. The `f16`/`f32`
`mul_mat_id` path (lighter body) and the mat-vec path do not.

## Analysis

**1. Explicit unroll requests bypass the cost heuristic.** In
`nir_opt_loop_unroll.c: check_unrolling_restrictions()`, `nir_loop_control_unroll` returns `true`
before reaching the `cost_limit = max_iter * LOOP_UNROLL_LIMIT` path, so an explicit unroll is
honoured regardless of resulting size.

**2. A NIR-cost ceiling on explicit unrolls is not an adequate fix.** I implemented one against a
locally built main and measured it:

| ceiling on `instr_cost x trip_count` | result |
|---|---|
| unlimited (stock) | `DEVICE_LOST` |
| 4096 | `DEVICE_LOST` (not caught) |
| 256 | no hang (caught) |

The culprit lands in (256, 4096], but legitimate explicit unrolls in the same shader set measure up
to ~1965, so any ceiling low enough to catch this also disables legitimate ones. NIR `instr_cost`
is a pre-register-allocation op count and cannot observe spilling, which is what actually explodes
the kernel. I am not proposing that patch.

**3. The usable signal exists only post-RA, and `brw` nearly has the plumbing.** In
`src/intel/compiler/brw/brw_compile_cs.cpp`, `brw_nir_quick_pressure_estimate()` produces per-SIMD
pressure and `simd_state.beyond_threshold[]`; `brw_simd_selection.cpp` consumes
`spilled[]`/`beyond_threshold[]` but only to select a **narrower SIMD**. For compute there is
nothing below SIMD8, so a SIMD8 shader that spills 5,908 times ships anyway. After RA,
`spilled_any_registers` and the spill counts are known, but there is no "spilled catastrophically,
fall back" path.

### Suggested direction

A two-phase compile: retain the pre-unroll NIR and, if the narrowest SIMD spills beyond a
catastrophic threshold, recompile from that NIR with unrolling suppressed. `brw_compile_cs` already
`nir_shader_clone`s per SIMD width and re-runs `brw_nir_optimize`, so the missing piece is the
"spills >> threshold -> re-run without unrolling" path. The obstacle is that `nir_opt_loop_unroll`
runs inside `brw_nir_optimize` and unrolling is destructive, so a "do not unroll" flag has to be
threaded through.

A much cheaper interim mitigation: emit an `INTEL_DEBUG=perf` warning when a shader spills beyond
some large threshold. Today a 2.4 MB / 5,900-spill kernel is emitted silently, which is the main
reason this took a long time to localise.

## Application-side workaround

```diff
-        [[unroll]] for (uint i = 0; i < BK / BK_STEP; i++) {
+#ifdef MUL_MAT_ID
+        [[dont_unroll]]
+#else
+        [[unroll]]
+#endif
+        for (uint i = 0; i < BK / BK_STEP; i++) {
```

Verified on this GPU with the fixed shader:

* `test-backend-ops -o MUL_MAT_ID -b Vulkan0`: 863/863 passed, 0 failures, 0 device losses
* `test-backend-ops -o MUL_MAT -b Vulkan0`: 982/982 passed (no regression on the dense path)

A 35B mixture-of-experts model (IQ3_XXS, experts resident on the GPU) goes from device-lost to
functioning, measured over two runs at 15.0-16.7 tok/s decode and 59.5-69.8 tok/s prefill
(`llama-bench -p 128 -n 64`, one repetition each). Previously it could only run with all expert
tensors forced onto the CPU, at 6.7 tok/s.

## GPU hang details

`xe` device coredump header from one occurrence (full dump attached; it is ~9 MB, almost all of it
the encoded GuC log):

```
**** Xe Device Coredump ****
Reason: Timedout job - seqno=4294967170, lrc_seqno=4294967170, guc_id=35, flags=0x0
kernel: 7.1.4-1-cachyos
module: xe
Process: repro [1516750]
PCI ID: 0x7d51
PCI revision: 0x03
GT id: 0
	Tile: 0
	Type: main
	IP ver: 12.74.4
	CS reference clock: 19200000
**** GuC Log ****
GuC firmware: i915/mtl_guc_70.bin
GuC version: 70.53.0 (wanted 70.53.0)
```

Note the kernel's reason string is the generic "Timedout job", but as shown above the timing tracks
`preempt_timeout_us` rather than `job_timeout_ms`.

No Mesa-side backtrace is included because Mesa does not crash: compilation succeeds and the
failure is the GPU-side reset of the submitted kernel.

## Relationship to #15550 / #15609 (different root cause)

#15609 was closed as a duplicate of #15550 ("[RADV] Regression: Infinite loop in NIR compiler
during nir_opt_dead_write_vars loading Gemma 4 via llama.cpp on gfx1100", llama.cpp issue
ggml-org/llama.cpp#23755). That one was bisected to 82b474c3 ("nir: remove is_only_uniform_src()
restriction", 2026-03-21), which made NIR unroll loops it previously did not, and it was fixed on
main by the NIR + ACO "big shaders" work. It manifested host-side, as compile time going from ~1 s
to 4-12 minutes.

This report is a **different problem**, on three independent grounds:

1. **It predates 82b474c3.** Mesa 26.0.6 does not contain that commit (26.0.6 is the version
   #15550 reports as good), yet the same SPIR-V produces a 109,277-instruction / 5,899-spill kernel
   and the same engine reset there. So this is not the 26.1-cycle unrolling regression.
2. **The #15550 fixes do not fix it.** My main build is `0f7afad` (2026-07-27), roughly four months
   after 82b474c3 and well after #15609 was closed (2026-06-04), so it contains that work. It still
   produces a 104,404-instruction / 5,907-spill kernel and still ends in `VK_ERROR_DEVICE_LOST`.
3. **The failure mode is GPU-side, not host-side.** #15550 was a compile-time explosion with an
   idle GPU. Here the shader does compile (in about 12 s for this pipeline, which is itself
   noticeable), and the resulting kernel then cannot be preempted, so the engine is reset.

Also possibly related:

* #15350 (open) - `[anv]/BMG 26.1.0-rc1 llama.cpp 26.1 regression`: additional unrolling of an
  `[[unroll]]`-tagged llama.cpp loop causing a performance regression on BMG. Same theme (anv
  honouring `[[unroll]]` on an expensive llama.cpp loop), different severity.
* #15216 (open) - anv coopmat performance regression on this same Arrow Lake Arc 140T.

## Attachments

* `repro.c` - standalone Vulkan harness
* `buggy.spv` - `matmul_id_subgroup_iq3_s_f32_f16acc`, from the unmodified shader
* `fixed.spv` - same pipeline, loop kept rolled, for A/B comparison
* `dmesg.txt` - kernel log across several occurrences
* `devcoredump.txt` - full `xe` device coredump (~9 MB; will upload as a snippet/attachment)
