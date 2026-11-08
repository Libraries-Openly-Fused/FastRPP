# FastRPP

FastRPP offers functions of AMD's [RPP](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rpp) tensor API
(`rppt_*`, RPP 3.2.0) with RPP's signatures, implemented with the
[Fused Kernel Library](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary) (FKL). It does for RPP what
[FastNPP](https://github.com/Libraries-Openly-Fused/FastNPP) does for NPP and cvGPUSpeedup does for OpenCV-CUDA.

Each `fastrpp::rppt_*` function takes RPP's run-time arguments, checks them on the host and, instead of running,
returns an operation. `fastrpp::executeOperations(handle, op1, op2, ...)` runs a chain of operations as **one kernel**:
- the destination of each operation is the source of the next one;
- the intermediate buffers are never written.

FastRPP is header only (`include/fast_rpp.h`), HIP only (AMD GPUs, ROCm), strict C++17.

## API coverage

The scope is what the RPP pre-processing benchmark of
[fkl-benchmarks](https://github.com/Libraries-Openly-Fused/fkl-benchmarks) needs (`rpp/benchmark_rpp_preprocessing.h`).

| RPP 3.2.0 | FastRPP | Implemented | Differences from RPP (see below) | Tests |
|---|---|---|---|---|
| `rppCreate` | `fastrpp::rppCreate` | `RPP_HIP_BACKEND` | FastRPP's own handle type; no device or scratch memory; batch size 0 and a null handle pointer give `rppStatusBadParm`, `RPP_HOST_BACKEND` gives `rppStatusNotImplemented` | `fastrpp_errors_test` |
| `rppDestroy` | `fastrpp::rppDestroy` | either back-end value frees the handle, as in RPP's GPU build | none (as in RPP, the stream given to `rppCreate` stays alive) | `fastrpp_errors_test` |
| `rppt_resize_mirror_normalize` | `fastrpp::rppt_resize_mirror_normalize<BATCH>` | `BILINEAR`; U8 NHWC, 3 channels -> F32 NCHW, 3 channels, dense destination of at most 4 GiB; one output size for the batch | the last columns and rows; no clearing, no synchronisation; ROIs are checked and never modified; host-readable parameter arrays; accepts `nStride = 0` (shared-source batch) | `fastrpp_resize_mirror_normalize_vs_rpp_test`, `fastrpp_chain_vs_rpp_test` |
| `rppt_channel_permute` | `fastrpp::rppt_channel_permute<BATCH>` | F32 NCHW, 3 channels -> F32 NCHW, 3 channels, dense, at most 4 GiB | only the six permutations; host-readable | `fastrpp_channel_permute_vs_rpp_test`, `fastrpp_chain_vs_rpp_test` |
| none | `fastrpp::executeOperations` | chains: `rppt_resize_mirror_normalize` or `rppt_channel_permute`, then any number of `rppt_channel_permute` | none | every test |
| none | `fastrpp::sharedSourceDesc` | the descriptor of a shared-source batch | FastRPP extension | `fastrpp_resize_mirror_normalize_vs_rpp_test`, `fastrpp_chain_vs_rpp_test` |

Naming another data type, layout or channel count in the template arguments stops at a `static_assert`. A run-time
descriptor that does not match the template arguments returns `RPP_ERROR_INVALID_{SRC,DST}_{DATATYPE,LAYOUT,CHANNELS}`.

## Use

```cpp
#include <fast_rpp.h>

// N crops of 60x120 at (i, i) of a 3840x2160 RGB frame, read in place: a shared-source batch (nStride = 0)
RpptDesc frameDesc = fastrpp::sharedSourceDesc(N, 3, 2160, 3840, RpptDataType::U8, RpptLayout::NHWC, 3840 * 3);
RpptDesc outDesc(N, 3, 128, 64, RpptDataType::F32, RpptLayout::NCHW);
for (int i = 0; i < N; ++i) rois[i].xywhROI = {{i, i}, 60, 120};

fastrpp::rppHandle_t handle;
fastrpp::rppCreate(&handle, N, 0, stream, RppBackend::RPP_HIP_BACKEND);
const auto rmn = fastrpp::rppt_resize_mirror_normalize<N>(frame, &frameDesc, resized, &outDesc, outSizes,
                                                          RpptInterpolationType::BILINEAR, mean, stdDev, mirror, rois,
                                                          RpptRoiType::XYWH, handle, RppBackend::RPP_HIP_BACKEND);
const auto cp = fastrpp::rppt_channel_permute<N>(resized, &outDesc, out, &outDesc, channelOrder, handle,
                                                 RppBackend::RPP_HIP_BACKEND);
fastrpp::executeOperations(handle, rmn, cp); // one kernel; `resized` is not written
hipStreamSynchronize(stream);
fastrpp::rppDestroy(handle, RppBackend::RPP_HIP_BACKEND);
```

The arguments are RPP's, and so are the types (`rppdefs.h`); FastRPP needs RPP's headers but not its library. To
share parameter arrays with RPP:
- use pinned host memory: RPP reads the arrays on the GPU, FastRPP reads them on the host;
- give RPP its own ROI array: RPP rewrites XYWH ROIs as LTRB in place, and needs room for `256 * ceil(n / 256)` ROIs (below).

## Semantics

### Template parameters

FKL builds its kernels at compile time, but an `RpptDesc` holds the batch size, data type and layout only at run time. So each
function takes them as template parameters, as FastNPP and cvGPUSpeedup do:

```cpp
rppt_resize_mirror_normalize<BATCH, SRC_TYPE = U8, SRC_LAYOUT = NHWC, DST_TYPE = F32, DST_LAYOUT = NCHW, CHANNELS = 3,
                             Impl = fastrpp::DefaultImplementation>
rppt_channel_permute<BATCH, TYPE = F32, SRC_LAYOUT = NCHW, DST_LAYOUT = NCHW, CHANNELS = 3,
                     Impl = fastrpp::DefaultImplementation>
```

`BATCH` must equal `srcDesc->n`, `dstDesc->n` and the handle's batch size; RPP itself processes the handle's batch size.
`Impl` selects the implementation of the per-image mirror and permutation (see "Implementations").

### Execution

An operation records the first check that failed in `op.status`, an `RppStatus`. `executeOperations` launches one
kernel on the stream of the handle given to it, and does not synchronise. Its result is the first of:
- an operation's failed check;
- `RPP_ERROR_INVALID_ARGUMENTS` for:
  - a null handle;
  - operations that do not form a chain (each source pointer and descriptor must be the previous destination's);
  - a handle whose batch size is not `BATCH`.

  Nothing is launched in these cases.
- `RPP_ERROR_HIP_LAUNCH` when a HIP error is pending after the launch. FKL checks `hipGetLastError()`, so the error
  can also come from earlier work, and the kernel may have been enqueued.

Every check and its status:

| Check | Status |
|---|---|
| `executionBackend` is not `RPP_HIP_BACKEND`; interpolation is not `BILINEAR` | `RPP_ERROR_NOT_IMPLEMENTED` |
| a null handle, pointer, descriptor or parameter array; `n` or the handle's batch size is not `BATCH`; a ROI type that is neither XYWH nor LTRB; a triple that is not a permutation | `RPP_ERROR_INVALID_ARGUMENTS` |
| `numDims` is not 4 | `RPP_ERROR_INVALID_{SRC,DST}_DIMS` |
| data type, layout or channels differ from the template arguments | `RPP_ERROR_INVALID_{SRC,DST}_{DATATYPE,LAYOUT,CHANNELS}` |
| width or height 0; a permutation whose source and destination sizes differ | `RPP_ERROR_INVALID_DIM_LENGTHS` |
| U8 source pixels not packed (`cStride` 1, `wStride` 3, `hStride >= 3 * w`); a destination (or permutation source) that is not dense (`RpptDesc`'s constructor strides; `nStride = 0` is rejected) | `RPP_ERROR_INVALID_{SRC,DST}_LAYOUT` |
| an ROI not inside its image, or of width or height below 1 | `RPP_ERROR_OUT_OF_BOUND_SRC_ROI` |
| per-image output sizes that differ from the destination's width and height; a float tensor over 4 GiB | `RPP_ERROR_NOT_IMPLEMENTED` |

### Shared-source batch

RPP's kernels address image `i` of a source at `srcPtr + offsetInBytes + i * nStride` elements, with the ROI of image `i`
inside it. RPP 3.2.0 does not document the stride fields, nor `nStride = 0`. FastRPP uses the same addressing and also
accepts `nStride = 0`: **n views of one image, each selected by its ROI**.

A batch of crops of one frame is then one descriptor (n crops, the frame's height and width) plus n ROIs, and FastRPP
reads the crops in place, without copies. `fastrpp::sharedSourceDesc` builds that descriptor.

The bilinear neighbours of a crop are clamped to its ROI, so FastRPP's result on a shared-source batch equals its result
on the same crops copied into a dense batch. The tests compare it with RPP's result on that dense batch outside RPP's
clamp region, where the two are bitwise equal. A destination never has `nStride = 0`.

### Parameter arrays and the handle

- The ROI, mean, standard-deviation, mirror, permutation and output-size arrays are read on the host when the operation is built. Their values are copied into the kernel's parameters.
- So they must be host-readable (pinned or pageable memory). Unlike RPP, which reads them on the GPU, they may change as soon as the function returns.
- The ROI array is never modified.
- `fastrpp::rppHandle_t` holds the batch size and the stream. It is not RPP's handle and allocates no device or scratch memory.

## Differences from RPP 3.2.0

Each was checked against RPP's source (ROCm/rocm-libraries, tag `therock-10.1`, `projects/rpp`, version 3.2.0).

- **The last columns and rows.** FastRPP samples at the pixel centres, the arithmetic of FKL's `ResizeHalfPixel`:
  - `src = max((x + 0.5) * roiW / dstW - 0.5, 0)` relative to the ROI;
  - `x1 = floor(src)`;
  - the second neighbour is clamped to the ROI's last column;
  - the same in y.

  RPP clamps both neighbours to `[0, roiW - 2]`, so it never reads the ROI's last column and row. Every output with
  `floor(src) >= roiW - 2` (or the same in y) differs: for 60x120 -> 64x128, the last 2 columns and rows. The
  difference reaches up to 255 / stdDev on noise, 126 with fkl-benchmarks' parameters. A mirrored image has these
  columns on the left. FastRPP does not reproduce this; its tests require every pixel within 1e-4 of a
  double-precision half-pixel reference.
- **Source positions.** RPP computes `x * r + (r - 1) / 2 + roiX` in float, in image coordinates; FastRPP computes
  `(x + 0.5) * r - 0.5`, in float, relative to the ROI. They give the same positions when `roiW / dstW` and
  `roiH / dstH` are exact in binary (60/64, 120/128, 640/128...) and the ROI origin is small. Otherwise the last bits
  differ, by up to a few times 1e-5 in the tests. RPP's absolute coordinates also lose precision for ROIs far from the
  image origin.
- **No clearing, no synchronisation.** RPP's `rppt_resize_mirror_normalize` zeroes its whole destination and synchronises
  its stream before its kernel. FastRPP writes every destination pixel, so there is nothing to clear, and
  `executeOperations` is asynchronous.
- **ROIs are checked.** An ROI outside its image, or of width or height below 1, gives `RPP_ERROR_OUT_OF_BOUND_SRC_ROI`.
  RPP checks nothing, and reads outside the image.
- **Mirror.** As in RPP, an image is mirrored only when `mirrorTensor[i] == 1`; other values are no mirror.
- **Normalisation.** As in RPP: `(v - mean) * (1 / stdDev)`, `1 / stdDev` in float, with no 1/255 scaling from U8 to F32
  and no check of `stdDev`.
- **Permutation.** As in RPP: output channel `k` is input channel `permutationTensor[3 * i + k]`.
  - RPP documents the six permutations and checks nothing: a repeated index copies a channel there, and an index above 2 is undefined.
  - FastRPP accepts only the six permutations and returns `RPP_ERROR_INVALID_ARGUMENTS` for any other triple.
- **Output sizes.** Every `dstImgSizes[i]` must equal the destination's width and height (`RPP_ERROR_NOT_IMPLEMENTED`
  otherwise). FKL bounds the threads of a batch by its largest image.
- **Dense destinations of at most 4 GiB.** The destination tensor must have the strides `RpptDesc`'s constructor gives,
  because FKL's planar write puts channel `k` at `k * w * h` elements. It must fit in 4 GiB, because FKL addresses
  images with 32-bit byte offsets.

Two defects of RPP 3.2.0 that a user of both libraries should know about (FastRPP has neither):
- RPP's kernels write 8 pixels per thread and check only the first one. So an output width that is not a multiple of 8
  makes RPP overwrite up to 7 pixels past each row.
- RPP's XYWH -> LTRB conversion runs `256 * ceil(n / 256)` threads with no bounds check. So an ROI array given to RPP
  must hold that many ROIs.

## Implementations of the per-image mirror and permutation

Both compute RPP's result. They differ in the code the GPU runs:
- `fastrpp::UniformBranches`, **the default**: the mirror and the six permutations are branches on a per-image value, uniform within a block.
- `fastrpp::Selects`: the mirror selects the source column, and the permutation is FKL's `VectorReorderRT`, with run-time channel indices.

The default was chosen by measurement. `tools/compare_implementations.hip` ran on an RX 9060 (gfx1200, ROCm 10.1) on
2026-11-08, on fkl-benchmarks' pipeline (a): N crops of 60x120 resized to 64x128, normalised and permuted. The times are
GPU time per launch, the median of 3 rotated rounds.

| N | `fkl_fused_halfpixel` (FKL, hand-composed) | FastRPP `UniformBranches` | FastRPP `Selects` |
|---|---|---|---|
| 10 | 0.0139 ms | 0.0136 ms | 0.0141 ms |
| 50 | 0.0293 ms | 0.0299 ms | 0.0340 ms |
| 100 | 0.0450 ms | 0.0470 ms | 0.0536 ms |
| 200 | 0.0764 ms | 0.0810 ms | 0.0954 ms |
| 300 | 0.1096 ms | 0.1151 ms | 0.1378 ms |

Results:
- `UniformBranches` is never slower than `Selects`, and is within about 5% of the hand-composed FKL chain at N = 300 (`Selects`: +26%).
- The difference with the hand-composed chain is the cost of RPP's per-image API: per-image constants and a run-time permutation, which double the kernel's parameter block (about 26 KB against about 14 KB at N = 300).
- Per-image parameters that vary within the batch (mirror 0/1/2, all six permutations) give the same times.
- Both implementations are bitwise equal to RPP outside its clamp region.

## Build

Requirements:
- ROCm 10.1, with `amdclang++` and RPP 3.2.0 (`amdrocm-rpp-dev10.1`);
- CMake 3.28 or later;
- an AMD GPU to run the tests (default architecture `gfx1200`, the RX 9060).

```sh
git clone --recursive https://github.com/Libraries-Openly-Fused/FastRPP.git
cmake -G Ninja -S FastRPP -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++
cmake --build build
ctest --test-dir build --output-on-failure
```

- **Strict C++17.** Every FastRPP target compiles with exactly `-std=c++17` (no extensions),
  `-Werror=c++20-extensions` and `-pedantic-errors`. FKL and RPP are included as system headers, so these flags apply
  in full to FastRPP's own code. FKL's `array.h` uses anonymous structs in unions, a GNU extension that
  `-pedantic-errors` rejects; FKL's own CI checks FKL.
- **FKL** is the git submodule `fkl`, pinned to the `LTS-C++17` branch at `64ad9195cd785d0c93a70b35e02d5a6c7c4a5048`.
  The configure fails unless the submodule is at that commit (`-DFASTRPP_CHECK_FKL_PIN=OFF` to skip the check).
  Nothing from FKL's `main` branch (C++20) is used.

## Tests

Every public function is compared with RPP's own output on the same inputs. The tolerance is 1e-4 absolute, as in
fkl-benchmarks; the test parameters keep the outputs below 256 in magnitude, where one float step is at most 2^-16
(about 1.5e-5). The agreement machinery is that of fkl-benchmarks, with RPP's clamp region computed per image instead
of "the last 2 columns and rows".

| Test | What it checks |
|---|---|
| `fastrpp_compile_test` | every public path instantiated under the strict flags, both implementations, and a three-operation chain that mixes them |
| `fastrpp_errors_test` | the handle, the statuses of the table above, and that nothing is launched after an error. Both buffers a launch could write hold a sentinel. `RPP_ERROR_HIP_LAUNCH` is not provoked |
| `fastrpp_resize_mirror_normalize_vs_rpp_test` | against RPP: 5 sizes, N = 1, 7, 300, noise and gradients, XYWH and LTRB, per-image mean, standard deviation and mirror (0, 1, 2), dense and shared-source batches, both implementations. Required: outside RPP's clamp region, and every pixel against the double reference. Where `src / dst` is not exact in binary, the RPP rows are informational and the reference is required within 1e-2. A FastRPP-only case checks the guard bands, with an output width that is not a multiple of 8, padded source rows and `offsetInBytes`, against the reference |
| `fastrpp_channel_permute_vs_rpp_test` | against RPP, bitwise: the six permutations mixed within a batch, N = 6, 12, 300, both implementations |
| `fastrpp_chain_vs_rpp_test` | the fused chain against RPP's calls, on the benchmark's pipeline (a), N = 10 and 300: the benchmark's parameters, and parameters that differ per image (every permutation, mirror 0/1/2). Also three operations with the two implementations mixed. The intermediate buffers must not be written |

## Continuous integration

`.github/workflows/rocm.yml` builds and runs the tests on the self-hosted ROCm runner (`[self-hosted, linux, x64, rocm]`),
for pull requests to `main` and by hand. That runner is also a benchmark machine, so the workflow never runs on a push.
It runs one job per pull request, or per branch for manual runs.

## License

Apache License 2.0 (`LICENSE`).
