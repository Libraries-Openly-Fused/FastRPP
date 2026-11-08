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
| `rppCreate` | `fastrpp::rppCreate` | `RPP_HIP_BACKEND` | FastRPP's own handle type; no scratch memory | `fastrpp_errors_test` |
| `rppDestroy` | `fastrpp::rppDestroy` | `RPP_HIP_BACKEND` | the stream given to `rppCreate` stays alive | `fastrpp_errors_test` |
| `rppt_resize_mirror_normalize` | `fastrpp::rppt_resize_mirror_normalize<BATCH>` | `BILINEAR`; U8 NHWC, 3 channels -> F32 NCHW, 3 channels, dense destination; one output size for the batch | the last columns and rows; no clearing, no synchronisation; the ROI array is not modified; host-readable parameter arrays; accepts `nStride = 0` (shared-source batch) | `fastrpp_resize_mirror_normalize_vs_rpp_test`, `fastrpp_chain_vs_rpp_test` |
| `rppt_channel_permute` | `fastrpp::rppt_channel_permute<BATCH>` | F32 NCHW, 3 channels -> F32 NCHW, 3 channels, dense | the triples are checked; host-readable | `fastrpp_channel_permute_vs_rpp_test`, `fastrpp_chain_vs_rpp_test` |
| none | `fastrpp::executeOperations` | chains: `rppt_resize_mirror_normalize` or `rppt_channel_permute`, then any number of `rppt_channel_permute` | none | every test |
| none | `fastrpp::sharedSourceDesc` | the descriptor of a shared-source batch | FastRPP extension | `fastrpp_resize_mirror_normalize_vs_rpp_test`, `fastrpp_chain_vs_rpp_test` |

Other data types, layouts and channel counts stop at a `static_assert`; the benchmark needs none of them.

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

The arguments are RPP's, and so are the arrays: the same `RpptDesc`, `RpptROI`, mean, standard-deviation, mirror and
permutation arrays work with RPP and with FastRPP. FastRPP needs RPP's headers (`rppdefs.h`) but not its library.

## Semantics

### Operations and the template parameters

FKL builds its kernels at compile time, but an `RpptDesc` holds the batch size, data type and layout only at run time. So each
function takes them as template parameters, as FastNPP and cvGPUSpeedup do:
- `BATCH`;
- the types and layouts, in RPP's vocabulary, defaulted to the implemented combination.

`BATCH` must equal `srcDesc->n`, `dstDesc->n` and the handle's batch size. RPP itself processes the handle's batch size.

### Execution

- An operation records the first check that failed (`op.status`, an `RppStatus`).
- `executeOperations` returns:
  - that status;
  - `RPP_ERROR_INVALID_ARGUMENTS` if the operations do not form a chain (each source pointer and descriptor must be the previous destination's);
  - `RPP_ERROR_HIP_LAUNCH` if the launch fails.

  After an error it launches nothing.
- It does not synchronise the stream.
- In a chain, the per-image work of every operation runs in the same thread. Only the last destination is written.

### Shared-source batch

FastRPP addresses image `i` of a source as RPP does: `(Rpp8u*)srcPtr + offsetInBytes + i * nStride` elements, the ROI
of image `i` inside it. RPP 3.2.0 documents `nStride` as the distance between images. FastRPP also accepts
`nStride = 0`: **n views of one image, each selected by its ROI**.

A batch of crops of one frame is then one descriptor (n crops, the frame's height and width) plus n ROIs, and FastRPP
reads the crops in place, without copies. `fastrpp::sharedSourceDesc` builds that descriptor.

As in RPP's dense batch, the bilinear neighbours of a crop are clamped to its ROI. So the result equals RPP's on the
same crops copied into a dense batch, which is what the tests compare. A destination never has `nStride = 0`.

### Parameter arrays and the handle

- The ROI, mean, standard-deviation, mirror, permutation and output-size arrays are read on the host when the operation is built. Their values are copied into the kernel's parameters.
- So they must be host-readable (pinned or pageable memory). Unlike RPP, which reads them on the GPU, they may change as soon as the function returns.
- The ROI array is never modified.
- `fastrpp::rppHandle_t` holds the batch size and the stream. It is not RPP's handle and allocates no memory.

## Differences from RPP 3.2.0

Each was checked against RPP's source (ROCm/rocm-libraries, tag `therock-10.1`, `projects/rpp`, version 3.2.0).

- **The last columns and rows.** FastRPP samples at the pixel centres, the arithmetic of FKL's `ResizeHalfPixel`:
  - `src = max((x + 0.5) * roiW / dstW - 0.5, 0)`;
  - `x1 = floor(src)`;
  - the second neighbour is clamped to the ROI's last column;
  - the same in y.

  RPP clamps both neighbours to `[0, roiW - 2]`, so it never reads the ROI's last column and row. Every output with
  `floor(src) >= roiW - 2` (or the same in y) differs: for 60x120 -> 64x128, the last 2 columns and rows (by up to 126
  on noise). A mirrored image has these columns on the left. FastRPP does not reproduce this; its tests require every
  pixel within 1e-4 of a double-precision half-pixel reference.
- **Source positions.** RPP computes `x * r + (r - 1) / 2` and FastRPP `(x + 0.5) * r - 0.5`, both in float. They give
  the same positions when `roiW / dstW` and `roiH / dstH` are exact in binary (60/64, 120/128, 640/128...). Otherwise the
  last bits differ, which on noisy images can exceed 1e-4.
- **No clearing, no synchronisation.** RPP's `rppt_resize_mirror_normalize` zeroes its whole destination and synchronises
  its stream before its kernel. FastRPP writes every destination pixel, so there is nothing to clear, and
  `executeOperations` is asynchronous.
- **Mirror.** As in RPP, an image is mirrored only when `mirrorTensor[i] == 1`; other values are no mirror.
- **Normalisation.** As in RPP: `(v - mean) * (1 / stdDev)`, `1 / stdDev` in float, with no 1/255 scaling from U8 to F32
  and no check of `stdDev`.
- **Permutation.** As in RPP: output channel `k` is input channel `permutationTensor[3 * i + k]`. FastRPP rejects triples
  that are not a permutation of 0, 1, 2; in RPP their result is undefined.
- **Output sizes.** Every `dstImgSizes[i]` must equal the destination's width and height (`RPP_ERROR_NOT_IMPLEMENTED`
  otherwise). FKL bounds the threads of a batch by its largest image.
- **Dense destinations.** The destination tensor must have the strides `RpptDesc`'s constructor gives. FKL's planar
  write puts channel `k` at `k * w * h` elements.

Two defects of RPP 3.2.0 that a user of both libraries should know about (FastRPP has neither):
- RPP's kernels write 8 pixels per thread and check only the first one. So an output width that is not a multiple of 8
  makes RPP overwrite up to 7 pixels past each row.
- RPP's XYWH -> LTRB conversion runs `256 * ceil(n / 256)` threads with no bounds check. So an ROI array given to RPP
  must hold that many ROIs.

## Implementations of the per-image mirror and permutation

Both compute RPP's result. They differ in the code the GPU runs:
- `fastrpp::UniformBranches`: the mirror and the six permutations are branches on a per-image value, uniform within a block.
- `fastrpp::Selects`: the mirror selects the source column, and the permutation is FKL's `VectorReorderRT`, with run-time channel indices.

The default, `fastrpp::DefaultImplementation`, is chosen by measurement: `tools/compare_implementations.hip` times both
against fkl-benchmarks' `fkl_fused_halfpixel` chain. The result and the reason will be recorded here.

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
fkl-benchmarks; the test parameters keep the outputs below 256 in magnitude, where one float step is about 3e-5. The
agreement machinery is that of fkl-benchmarks, with RPP's clamp region computed per image instead of "the last 2
columns and rows".

| Test | What it checks |
|---|---|
| `fastrpp_compile_test` | every public path instantiated under the strict flags, both implementations |
| `fastrpp_errors_test` | the handle, every documented status, and that nothing is launched after an error |
| `fastrpp_resize_mirror_normalize_vs_rpp_test` | against RPP: 5 sizes, N = 1, 7, 300, noise and gradients, XYWH and LTRB, per-image mean, standard deviation and mirror (0, 1, 2), dense and shared-source batches, both implementations. Required: outside RPP's clamp region, and every pixel against the double reference. The case where `src / dst` is not exact in binary is informational |
| `fastrpp_channel_permute_vs_rpp_test` | against RPP, bitwise: the six permutations mixed within a batch, N = 6, 12, 300, both implementations |
| `fastrpp_chain_vs_rpp_test` | the fused chain against RPP's two calls, on the benchmark's pipeline (a), N = 10 and 300: the benchmark's parameters, and parameters that differ per image (every permutation, mirror 0/1/2). The intermediate buffer must not be written |

## Continuous integration

`.github/workflows/rocm.yml` builds and runs the tests on the self-hosted ROCm runner (`[self-hosted, linux, x64, rocm]`),
for pull requests to `main` and by hand. That runner is also a benchmark machine, so the workflow never runs on a push.

## License

Apache License 2.0 (`LICENSE`).
