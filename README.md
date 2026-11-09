# FastRPP

FastRPP offers functions of AMD's [RPP](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rpp) tensor API
(`rppt_*`, RPP 3.2.0) with RPP's signatures but the handle and the back-end, as operations of the
[Fused Kernel Library](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary) (FKL). It does for RPP what
[FastNPP](https://github.com/Libraries-Openly-Fused/FastNPP) does for NPP and cvGPUSpeedup does for OpenCV-CUDA.

Each `fastrpp::rppt_*` function does three things:
- it takes RPP's arguments, all but the handle and the back-end, and checks them on the host;
- it translates RPP's objects into FKL's: descriptors and ROIs into views of the tensors, parameter arrays into the
  parameters of FKL operations;
- instead of running, it returns **one FKL instantiable operation (IOp)**: the FKL operations it needs, joined with
  `.then`.

FastRPP defines no operation of its own; new operations belong in FKL. The only ReadBack operation it uses is FKL's
resize.

`fastrpp::executeOperations(rppHandle, executionBackend, iOps...)` runs IOps **fused into one**, with
`fk::executeOperations`. Only this function takes RPP's own `rppHandle_t` and RPP's `RppBackend` (as FastNPP's
`executeOperations` takes the `NppStreamContext`):
- `RPP_HIP_BACKEND`: one GPU kernel on the handle's stream;
- `RPP_HOST_BACKEND`: FKL's CPU executor, multithreaded, on host memory.

As in FKL, a chain is one read IOp, any number of compute IOps and one write IOp:
- `rppt_resize_mirror_normalize` returns a read IOp;
- `rppt_channel_permute` returns a compute IOp;
- `rppt_copy` returns a write IOp.

Only a read IOp reads memory and only a write IOp writes it, so the intermediate tensors of a chain are never read or
written.

FastRPP is header only (`include/fast_rpp.h`), HIP only (AMD GPUs, ROCm), strict C++17.

## API coverage

The scope is what the RPP pre-processing benchmark of
[fkl-benchmarks](https://github.com/Libraries-Openly-Fused/fkl-benchmarks) needs (`rpp/benchmark_rpp_preprocessing.h`),
plus `rppt_copy` for the write, as FastNPP's `CopyBatch`.

| RPP 3.2.0 | FastRPP | Returns | Implemented | Tests |
|---|---|---|---|---|
| `rppt_resize_mirror_normalize` | `fastrpp::rppt_resize_mirror_normalize<BATCH>` | a read IOp: `fk::BatchRead<PlanePolicy::PROCESS_ALL>` of one chain per image, `fk::PerThreadRead` (the crop) `.then(fk::ResizeHalfPixel<INTER_LINEAR>)` `.then(fk::Sub)` `.then(fk::Mul)` | `BILINEAR`; U8 NHWC -> F32 NCHW, 3 channels; one output size for the batch; no mirror | `fastrpp_resize_mirror_normalize_vs_rpp_test`, `fastrpp_chain_vs_rpp_test` |
| `rppt_channel_permute` | `fastrpp::rppt_channel_permute` | a compute IOp: `fk::VectorReorderRT<float3>` | F32 NCHW -> F32 NCHW, 3 channels; one of the six permutations, the same for every image | `fastrpp_channel_permute_vs_rpp_test`, `fastrpp_chain_vs_rpp_test` |
| `rppt_copy` | `fastrpp::rppt_copy` | a write IOp: `fk::TensorSplit<float3>` | F32 NCHW -> F32 NCHW, 3 channels; a dense destination of at most 4 GiB | every test that runs on the GPU |
| none | `fastrpp::executeOperations` | runs FKL IOps fused into one: a GPU kernel on the stream of an RPP handle (`RPP_HIP_BACKEND`), or FKL's CPU executor on host memory (`RPP_HOST_BACKEND`) | any FKL chain, FastRPP's IOps or not; both back-ends | every test that runs IOps |
| none | `fastrpp::sharedSourceDesc` | an `RpptDesc` | the descriptor of a shared-source batch, a FastRPP extension | `fastrpp_resize_mirror_normalize_vs_rpp_test`, `fastrpp_chain_vs_rpp_test` |
| none | `fastrpp::Error` | | what every failed check throws, with RPP's status | `fastrpp_errors_test` |

Naming another data type, layout or channel count in the template arguments stops at a `static_assert`. A run-time
descriptor that does not match the template arguments throws `RPP_ERROR_INVALID_{SRC,DST}_{DATATYPE,LAYOUT,CHANNELS}`.

The handle is RPP's: `rppCreate` and `rppDestroy` are RPP's own functions, and FastRPP has none.

## Use

```cpp
#include <fast_rpp.h>

// N crops of 60x120 at (i, i) of a 3840x2160 RGB frame, read in place: a shared-source batch (nStride = 0)
RpptDesc frameDesc = fastrpp::sharedSourceDesc(N, 3, 2160, 3840, RpptDataType::U8, RpptLayout::NHWC, 3840 * 3);
RpptDesc outDesc(N, 3, 128, 64, RpptDataType::F32, RpptLayout::NCHW);
for (int i = 0; i < N; ++i) rois[i].xywhROI = {{i, i}, 60, 120};

rppHandle_t handle; // RPP's own handle, on stream
rppCreate(&handle, N, 0, stream, RppBackend::RPP_HIP_BACKEND);
// Three FKL IOps, built on the host from RPP's arguments but the handle
const auto rmn = fastrpp::rppt_resize_mirror_normalize<N>(frame, &frameDesc, resized, &outDesc, outSizes,
                                                          RpptInterpolationType::BILINEAR, mean, stdDev, mirror, rois,
                                                          RpptRoiType::XYWH);
const auto cp = fastrpp::rppt_channel_permute(resized, &outDesc, permuted, &outDesc, channelOrder);
const auto copy = fastrpp::rppt_copy(permuted, &outDesc, out, &outDesc);
// One kernel on the handle's stream: it reads the frame and writes out; resized and permuted are never accessed
fastrpp::executeOperations(handle, RppBackend::RPP_HIP_BACKEND, rmn, cp, copy);
hipStreamSynchronize(stream);
rppDestroy(handle, RppBackend::RPP_HIP_BACKEND);
```

With host memory and a host handle (`rppCreate(&handle, N, 0, nullptr, RppBackend::RPP_HOST_BACKEND)`), the same IOps
run on the CPU: `fastrpp::executeOperations(handle, RppBackend::RPP_HOST_BACKEND, rmn, cp, copy)`.

The arguments are RPP's but the handle and the back-end, and the types and the handle are RPP's (`rpp.h`). FastRPP
needs RPP's headers and its library: `executeOperations` gets the handle's stream with RPP's `rppGetStream`, so the
CMake target `FastRPP::FastRPP` links `rpp::rpp`. To share parameter arrays with RPP:
- use pinned host memory: RPP reads the arrays on the GPU, FastRPP reads them on the host;
- give RPP its own ROI array: RPP rewrites XYWH ROIs as LTRB in place, and needs room for `256 * ceil(n / 256)` ROIs (below).

`fastrpp::executeOperations` takes any FKL IOps, so FastRPP's IOps mix with FKL's own. For instance, a chain that
starts from an F32 NCHW tensor reads it with `fk::TensorPack<fk::float3>` (`fastrpp_channel_permute_vs_rpp_test`).

## Semantics

### What each IOp reads and writes

- `rppt_resize_mirror_normalize` reads the crops of the source. Its destination descriptor gives the output width and
  height and must describe F32 NCHW with 3 channels; `dstPtr` is never accessed.
- `rppt_channel_permute` accesses no memory. Its descriptors are checked; `srcPtr` and `dstPtr` are never accessed.
- `rppt_copy` writes `dstPtr`. `srcPtr` is never accessed: the values come from the IOps before it.

The pointers of the intermediate tensors can be any value, `nullptr` included. FastRPP does not check that the IOps
form RPP's chain (each source the previous destination): it builds IOps, and FKL checks at compile time that a chain
is a read, compute IOps and a write.

The IOps are values: each holds copies of its parameters, so it can be executed any number of times, and it can be
built once and executed later. They hold the pointers they were built with: these must be in the memory of the back-end
that runs them (device memory for `RPP_HIP_BACKEND`, host memory for `RPP_HOST_BACKEND`), as RPP's back-ends require.

### Back-ends

`executeOperations(rppHandle, executionBackend, iOps...)` runs the same IOps on either of RPP's back-ends:
- **`RPP_HIP_BACKEND`:** one kernel, `fk::TransformDPP<ParArch::GPU_AMD>`, on the stream of the handle. RPP's
  `rppGetStream` gives the stream: the one given to `rppCreate`, where `nullptr` is the null stream. It is put in an
  `fk::Stream`, which never destroys it. `executeOperations` does not synchronise.
- **`RPP_HOST_BACKEND`:** FKL's CPU executor, `fk::TransformDPP<ParArch::CPU_MULTITHREAD>`, on FKL's thread pool.
  `executeOperations` returns when the work is done, as RPP's host back-end does. The pool is FKL's: one for the
  process, of FKL's default size. RPP's API gives no access to the handle's `numThreads`, so the handle is not used.

`ParArch::CPU_MULTITHREAD` needs `FK_CPU_MULTITHREAD_ENABLED`, FKL's `thread-pool` submodule and the threads library.
FastRPP's CMake option `FASTRPP_CPU_MULTITHREAD` (ON by default) sets all three, as FKL's own `ENABLE_CPU_MULTITHREAD`
does. Without `FK_CPU_MULTITHREAD_ENABLED`, `RPP_HOST_BACKEND` runs on FKL's single-threaded `ParArch::CPU`; the HIP
back-end does not depend on it.

### Template parameters

FKL builds its kernels at compile time, but an `RpptDesc` holds the data type and layout only at run time. So each
function takes them as template parameters, as FastNPP and cvGPUSpeedup do:

```cpp
rppt_resize_mirror_normalize<BATCH, SRC_TYPE = U8, SRC_LAYOUT = NHWC, DST_TYPE = F32, DST_LAYOUT = NCHW, CHANNELS = 3>
rppt_channel_permute<TYPE = F32, SRC_LAYOUT = NCHW, DST_LAYOUT = NCHW, CHANNELS = 3>
rppt_copy<TYPE = F32, SRC_LAYOUT = NCHW, DST_LAYOUT = NCHW, CHANNELS = 3>
```

`rppt_resize_mirror_normalize`'s IOp holds one chain per image, so it needs the batch size at compile time: `BATCH`
must equal `srcDesc->n` and `dstDesc->n`. For the other two functions, the source and the destination must have the
same `n`.

In a chain, every `n` must be the batch of the read IOp (`BATCH`): the kernel runs one image per plane of the read, and
FKL's write writes every image it is given, with no bounds check. Each FastRPP function sees only its own arguments, so
none of them can check this: a destination of `rppt_copy` with `n` below `BATCH` is written past its end. The handle's
batch size is not used.

### Errors

A failed check throws `fastrpp::Error`, a `std::runtime_error`. Its `status()` is the `RppStatus` of the failure:
- RPP's own status for that failure;
- or `RPP_ERROR_NOT_IMPLEMENTED` for what FastRPP does not implement.

Its `what()` names the function and the check.

`executeOperations` throws `fastrpp::Error` with `RPP_ERROR_INVALID_ARGUMENTS` for a null handle, and with `RPP_ERROR`
when `rppGetStream` fails (the message gives its `rppStatus_t`). RPP's API does not tell a host handle from a HIP handle
on the null stream, so with `RPP_HIP_BACKEND` a host handle runs on the null stream. When a HIP error is pending after
the launch, it throws FKL's `std::runtime_error`; FKL checks `hipGetLastError()`, so the error can also come from
earlier work.

Every check and its status:

| Check | Status |
|---|---|
| interpolation is not `BILINEAR` | `RPP_ERROR_NOT_IMPLEMENTED` |
| a null descriptor or parameter array; a null source (`rppt_resize_mirror_normalize`) or destination (`rppt_copy`); `n` is not `BATCH` (`rppt_resize_mirror_normalize`) or the source's `n` (`rppt_channel_permute`, `rppt_copy`); a ROI type that is neither XYWH nor LTRB; a triple that is not a permutation; a null handle (`executeOperations`) | `RPP_ERROR_INVALID_ARGUMENTS` |
| `numDims` is not 4 | `RPP_ERROR_INVALID_{SRC,DST}_DIMS` |
| data type, layout or channels differ from the template arguments | `RPP_ERROR_INVALID_{SRC,DST}_{DATATYPE,LAYOUT,CHANNELS}` |
| `n`, width or height 0; a source and destination of different sizes (`rppt_channel_permute`, `rppt_copy`) | `RPP_ERROR_INVALID_DIM_LENGTHS` |
| U8 source pixels not packed (`cStride` 1, `wStride` 3, `hStride >= 3 * w`); a destination of `rppt_copy` that is not dense (`RpptDesc`'s constructor strides; `nStride = 0` is rejected) | `RPP_ERROR_INVALID_{SRC,DST}_LAYOUT` |
| an ROI not inside its image, or of width or height below 1 | `RPP_ERROR_OUT_OF_BOUND_SRC_ROI` |
| a mirrored image (`mirrorTensor[i] == 1`); per-image output sizes that differ from the destination's width and height; permutations that differ between images; a destination of `rppt_copy` over 4 GiB | `RPP_ERROR_NOT_IMPLEMENTED` |
| `rppGetStream` fails (`executeOperations`) | `RPP_ERROR` |

### Shared-source batch

RPP's kernels address image `i` of a source at `srcPtr + offsetInBytes + i * nStride` elements, with the ROI of image `i`
inside it. RPP 3.2.0 does not document the stride fields, nor `nStride = 0`. FastRPP uses the same addressing and also
accepts `nStride = 0`: **n views of one image, each selected by its ROI**.

A batch of crops of one frame is then one descriptor (n crops, the frame's height and width) plus n ROIs, and FastRPP
reads the crops in place, without copies. `fastrpp::sharedSourceDesc` builds that descriptor.

The bilinear neighbours of a crop are clamped to its ROI, so FastRPP's result on a shared-source batch equals its result
on the same crops copied into a dense batch. The tests compare it with RPP's result on that dense batch outside RPP's
clamp region, where the two are bitwise equal. The destination of `rppt_copy` never has `nStride = 0`.

### Parameter arrays and the handle

- The ROI, mean, standard-deviation, mirror, permutation and output-size arrays are read on the host when the IOp is built. Their values are copied into the IOp.
- So they must be host-readable (pinned or pageable memory). Unlike RPP, which reads them on the GPU, they may change as soon as the function returns.
- The ROI array is never modified.
- The handle is RPP's `rppHandle_t`, made by RPP's `rppCreate`. Only `executeOperations` takes it, with the back-end. It uses only the handle's stream, for `RPP_HIP_BACKEND`: not its batch size, its number of threads, nor the scratch memory RPP allocates for it.

## Differences from RPP 3.2.0

Each was checked against RPP's source (ROCm/rocm-libraries, tag `therock-10.1`, `projects/rpp`, version 3.2.0).

- **IOps instead of runs.** Each function returns an FKL IOp and runs nothing: `executeOperations` runs a chain as one
  kernel. A failed check throws `fastrpp::Error` instead of returning a status.
- **The handle and the back-end only in `executeOperations`.** RPP's functions take the handle and the back-end and run.
  FastRPP's take neither: `executeOperations` takes both and runs the chain, on the handle's stream or on the CPU.
- **Host threads.** RPP's host back-end runs on the handle's `numThreads` (OpenMP); FastRPP's on FKL's thread pool.
- **The intermediate tensors.** In RPP, each call reads its source and writes its destination. A FastRPP chain reads
  only the source of its read IOp and writes only the destination of its write IOp (`rppt_copy`).
- **`rppt_copy`** writes the values of the IOps before it to its destination; it does not read its source.
- **No mirror.** RPP's HIP back-end mirrors image `i` when `mirrorTensor[i] == 1`, and its host back-end when
  `mirrorTensor[i] != 0`. FastRPP throws `RPP_ERROR_NOT_IMPLEMENTED` for 1, and any other value means no mirror, on both
  back-ends: the IOp is built before the back-end is chosen, and it follows RPP's HIP back-end. FKL (at the pinned commit) has no operation that maps column
  `x` to `w - 1 - x`, except its warping, which is a ReadBack operation, and FastRPP uses no ReadBack operation but the
  resize.
- **One permutation per batch.** RPP takes a triple per image; FastRPP needs the same triple for every image, because
  FKL's compute operations have one set of parameters for the whole batch. Output channel `k` is input channel
  `permutationTensor[k]`, as in RPP.
  - RPP documents the six permutations and checks nothing: a repeated index copies a channel there, and an index above 2 is undefined.
  - FastRPP accepts only the six permutations and throws `RPP_ERROR_INVALID_ARGUMENTS` for any other triple.
- **The last columns and rows.** FastRPP samples at the pixel centres, the arithmetic of FKL's `ResizeHalfPixel`:
  - `src = max((x + 0.5) * roiW / dstW - 0.5, 0)` relative to the ROI;
  - `x1 = floor(src)`;
  - the second neighbour is clamped to the ROI's last column;
  - the same in y.

  RPP's HIP kernels clamp both neighbours to `[0, roiW - 2]`, so they never read the ROI's last column and row. RPP's
  host kernels clamp both neighbours into the ROI, as FastRPP does. On the HIP back-end, every output with
  `floor(src) >= roiW - 2` (or the same in y) differs: for 60x120 -> 64x128, the last 2 columns and rows. The
  difference reaches up to 255 / stdDev on noise, 126 with fkl-benchmarks' parameters. FastRPP does not reproduce
  this; its tests require every pixel within 1e-4 of a double-precision half-pixel reference.
- **Source positions.** RPP computes `x * r + (r - 1) / 2 + roiX` in float, in image coordinates; FastRPP computes
  `(x + 0.5) * r - 0.5`, in float, relative to the ROI. They give the same positions when `roiW / dstW` and
  `roiH / dstH` are exact in binary (60/64, 120/128, 640/128...) and the ROI origin is small. Otherwise the last bits
  differ, by up to a few times 1e-5 in the tests. RPP's absolute coordinates also lose precision for ROIs far from the
  image origin.
- **No clearing, no synchronisation.** RPP's `rppt_resize_mirror_normalize` zeroes its whole destination and synchronises
  its stream before its kernel. A FastRPP chain writes every pixel of its destination, so there is nothing to clear,
  and `executeOperations` is asynchronous.
- **ROIs are checked.** An ROI outside its image, or of width or height below 1, throws `RPP_ERROR_OUT_OF_BOUND_SRC_ROI`.
  RPP checks nothing, and reads outside the image.
- **Normalisation.** As in RPP: `(v - mean) * (1 / stdDev)`, `1 / stdDev` in float, with no 1/255 scaling from U8 to F32
  and no check of `stdDev`.
- **Output sizes.** Every `dstImgSizes[i]` must equal the destination's width and height (`RPP_ERROR_NOT_IMPLEMENTED`
  otherwise). FKL bounds the threads of a batch by its largest image.
- **Dense destinations of at most 4 GiB.** The destination of `rppt_copy` must have the strides `RpptDesc`'s
  constructor gives, because FKL's planar write puts channel `k` at `k * w * h` elements. It must fit in 4 GiB, because
  FKL addresses images with 32-bit byte offsets.

What RPP has and FastRPP does not would need FKL first: a mirror (an operation, or a mirror option of FKL's resize so
that the resize stays the only ReadBack operation), and compute operations with per-image parameters.

Two defects of RPP 3.2.0 that a user of both libraries should know about (FastRPP has neither):
- RPP's kernels write 8 pixels per thread and check only the first one. So an output width that is not a multiple of 8
  makes RPP overwrite up to 7 pixels past each row.
- RPP's XYWH -> LTRB conversion runs `256 * ceil(n / 256)` threads with no bounds check. So an ROI array given to RPP
  must hold that many ROIs.

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
- **`FASTRPP_CPU_MULTITHREAD`** (ON by default) runs `RPP_HOST_BACKEND` on FKL's `ParArch::CPU_MULTITHREAD`. It defines
  `FK_CPU_MULTITHREAD_ENABLED`, adds the headers of FKL's own submodule `fkl/thread-pool` (at the commit FKL pins) and
  links the threads library. The configure fails if that submodule is missing: `git submodule update --init --recursive`.
  OFF runs `RPP_HOST_BACKEND` on FKL's single-threaded `ParArch::CPU`.

## Tests

Every function that builds an IOp is compared with RPP's own output on the same inputs, on each back-end: FastRPP's
chain on the GPU against RPP's HIP back-end, and on the CPU (host memory) against RPP's host back-end. The tolerance is 1e-4
absolute, as in fkl-benchmarks; the test parameters keep the outputs below 256 in magnitude, where one float step is at
most 2^-16 (about 1.5e-5). The agreement machinery is that of fkl-benchmarks, with the clamp region of RPP's HIP kernels
computed per image instead of "the last 2 columns and rows". RPP's host kernels have no clamp region, so on the host
back-end every pixel is compared. RPP's host back-end mirrors on any non-zero flag, so its comparisons use the flag 0.

| Test | What it checks |
|---|---|
| `fastrpp_compile_test` | every public path instantiated under the strict flags, the GPU kernels and FKL's CPU executor included. Each function returns exactly the IOp written in the test with FKL alone, and takes the parameters RPP's header declares for it, without the handle and the back-end; `executeOperations` takes RPP's `rppHandle_t` and `RppBackend`; the host back-end is `CPU_MULTITHREAD` when the build enables it |
| `fastrpp_errors_test` | every check of the table above throws `fastrpp::Error` with its status. `executeOperations` with RPP's handle, `RPP_HIP_BACKEND`: a null handle launches nothing; the kernel goes to the handle's stream (captured into a graph while that stream is captured, and run when the graph is launched); a handle made with `stream = nullptr` runs on the null stream. `RPP_HOST_BACKEND`, with a host handle and pageable host memory: a null handle runs nothing; the chain has finished when `executeOperations` returns, and its output equals the GPU's within the tolerance. On both, a valid chain writes only the destination of `rppt_copy`. A HIP launch error and a failing `rppGetStream` are not provoked |
| `fastrpp_resize_mirror_normalize_vs_rpp_test` | on each back-end, `rppt_resize_mirror_normalize` then `rppt_copy`, against RPP: 5 sizes, N = 1, 7, 300, noise and gradients, XYWH and LTRB, per-image mean and standard deviation, mirror flag 0 and 2 (HIP) or 0 (host), dense and shared-source batches. Required: outside the clamp region of RPP's HIP kernels (HIP) or on every pixel (host), and every pixel against the double reference. Where `src / dst` is not exact in binary, the RPP rows are informational and the reference is required within 1e-2. A FastRPP-only case checks the guard bands, with an output width that is not a multiple of 8, padded source rows and `offsetInBytes`, against the reference |
| `fastrpp_channel_permute_vs_rpp_test` | on each back-end, an FKL read, `rppt_channel_permute`, `rppt_copy`, against RPP, bitwise: each of the six permutations, N = 6, 12, 300 |
| `fastrpp_chain_vs_rpp_test` | on each back-end, the chain `rppt_resize_mirror_normalize`, `rppt_channel_permute`, `rppt_copy` against RPP's calls, on the benchmark's pipeline (a), N = 10 and 300: the benchmark's parameters, and parameters that differ per image (mean, standard deviation, mirror flag 0/2 on HIP) with the permutation (1, 2, 0). Also two permutations in a row. The intermediate tensors must not be written |

## Continuous integration

`.github/workflows/rocm.yml` builds and runs the tests on the self-hosted ROCm runner (`[self-hosted, linux, x64, rocm]`),
for pull requests to `main` and by hand. That runner is also a benchmark machine, so the workflow never runs on a push.
It runs one job per pull request, or per branch for manual runs. It checks the submodules out recursively, for FKL's
`thread-pool`.

## License

Apache License 2.0 (`LICENSE`).
