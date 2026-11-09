/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// The fused chain: fastrpp::executeOperations(handle, backend, rppt_resize_mirror_normalize, rppt_channel_permute,
// rppt_copy), one GPU kernel or FKL's CPU executor on host memory, against RPP's two calls on the same back-end, on
// pipeline (a) of fkl-benchmarks (rpp/benchmark_rpp_preprocessing.h): N crops of 60x120 at (i, i) of a 3840x2160 frame
// of the benchmark's seeded noise, resized to 64x128. FastRPP reads the crops in place (shared-source batch); RPP reads
// them copied into a dense batch, RPP's documented batched form.
//   benchmark parameters  the benchmark's: mean = sub / 0.3, stdDev = div / 0.3 in the source channel order, mirror 0,
//                         permutation (2, 1, 0)
//   mixed parameters      per image a different mean and standard deviation, a mirror flag of 0 or 2 on the HIP
//                         back-end (no mirror in either library) and 0 on the host back-end (RPP's host back-end mirrors
//                         on 2), and the permutation (1, 2, 0), which is not its own inverse
// Rows as in fastrpp_resize_mirror_normalize_vs_rpp_test. The intermediate tensors (the destination of
// rppt_resize_mirror_normalize, the source and destination of rppt_channel_permute) must not be written: only the write
// IOp, rppt_copy's, writes memory.
#include <common/test_common.h>

#include <memory>

using namespace fastrpp_test;

namespace {

constexpr int SRC_W = 3840, SRC_H = 2160, CROP_W = 60, CROP_H = 120, OUT_W = 64, OUT_H = 128;
constexpr float ALPHA = 0.3f;
constexpr float SUB[3] = {1.f, 4.f, 3.2f}, DIV[3] = {3.2f, 0.6f, 11.8f};
constexpr uint32_t SENTINEL = 0xFFFFFFFFu; // the bytes the intermediate tensors are filled with (a NaN)

// The crops of the frame copied into a dense batch, and their ROIs (each the whole crop): RPP's input
struct RppInput {
    Buffer<unsigned char> crops;
    RpptDesc desc;
    Pinned<RpptROI> rois;
    RppInput(const Buffer<unsigned char>& frame, const size_t n)
        : crops(static_cast<size_t>(CROP_W) * CROP_H * 3 * n, frame.backend()),
          desc(static_cast<Rpp32u>(n), 3, CROP_H, CROP_W, RpptDataType::U8, RpptLayout::NHWC), rois(rppRoiCapacity(n)) {
        const size_t cropBytes = static_cast<size_t>(CROP_W) * CROP_H * 3;
        for (size_t i = 0; i < n; ++i) {
            copy2D(frame.backend(), crops.data() + cropBytes * i, static_cast<size_t>(CROP_W) * 3, frame.data() + (i * SRC_W + i) * 3,
                   static_cast<size_t>(SRC_W) * 3, static_cast<size_t>(CROP_W) * 3, CROP_H);
            rois[i].xywhROI = {{0, 0}, CROP_W, CROP_H};
        }
    }
};

// The specs of the double reference and RPP's parameter arrays for pipeline (a); output channel k of image i is the
// normalised channel spec.perm[k]
template <size_t N>
struct Parameters {
    std::vector<ImageSpec> specs = std::vector<ImageSpec>(N);
    Pinned<Rpp32f> mean{3 * N}, stdDev{3 * N};
    Pinned<Rpp32u> mirror{N};
    Pinned<RpptImagePatch> sizes{N};
    std::vector<RpptROI> rois = std::vector<RpptROI>(N);

    Parameters(const std::vector<unsigned char>& frameHost, const bool mixed, const uint32_t (&perm)[3], const RppBackend backend) {
        for (size_t i = 0; i < N; ++i) {
            ImageSpec& s = specs[i];
            s.image = frameHost.data();
            s.pitch = static_cast<size_t>(SRC_W) * 3;
            s.roiX = static_cast<int>(i);
            s.roiY = static_cast<int>(i);
            s.roiW = CROP_W;
            s.roiH = CROP_H;
            for (size_t c = 0; c < 3; ++c) {
                if (mixed) {
                    s.mean[c] = static_cast<float>((i * 11 + c * 5) % 40);
                    s.stdDev[c] = 1.0f + 0.25f * static_cast<float>((i + c) % 7);
                } else {
                    // The permutation (2, 1, 0) comes after the normalisation: output channel 2 - c is (alpha * v_c - sub[2 - c]) / div[2 - c]
                    s.mean[c] = SUB[2 - c] / ALPHA;
                    s.stdDev[c] = DIV[2 - c] / ALPHA;
                }
                s.perm[c] = perm[c];
                mean[3 * i + c] = s.mean[c];
                stdDev[3 * i + c] = s.stdDev[c];
            }
            mirror[i] = mixed && backend == RppBackend::RPP_HIP_BACKEND ? static_cast<Rpp32u>((i % 2) * 2) : 0u;
            sizes[i] = RpptImagePatch{OUT_W, OUT_H};
            rois[i].xywhROI = {{s.roiX, s.roiY}, CROP_W, CROP_H};
        }
    }
};

void rows(Report& report, const std::vector<ImageSpec>& specs, const std::vector<float>& fast, const std::vector<float>& rpp,
          const RppBackend backend) {
    const size_t n = specs.size();
    const size_t imageValues = 3 * static_cast<size_t>(OUT_W) * OUT_H;
    std::vector<double> expected(n * imageValues);
    for (size_t i = 0; i < n; ++i) {
        referenceImage(specs[i], OUT_W, OUT_H, expected.data() + i * imageValues);
    }
    auto inClampRegion = [&](const size_t i, const int x, const int y) { return rppEdge(specs[i], OUT_W, OUT_H, x, y); };
    auto outsideClampRegion = [&](const size_t i, const int x, const int y) { return !rppEdge(specs[i], OUT_W, OUT_H, x, y); };
    auto everywhere = [](const size_t, const int, const int) { return true; };
    if (backend == RppBackend::RPP_HIP_BACKEND) {
        report.row("FastRPP vs RPP, outside RPP's clamp region", compare(fast.data(), rpp.data(), n, OUT_W, OUT_H, outsideClampRegion), true);
        report.row("FastRPP vs RPP, in RPP's clamp region", compare(fast.data(), rpp.data(), n, OUT_W, OUT_H, inClampRegion), false);
    } else {
        report.row("FastRPP vs RPP, every pixel (RPP's host kernels have no clamp region)",
                   compare(fast.data(), rpp.data(), n, OUT_W, OUT_H, everywhere), true);
    }
    report.row("FastRPP vs the double half-pixel reference, every pixel", compare(fast.data(), expected.data(), n, OUT_W, OUT_H, everywhere), true);
    if (backend == RppBackend::RPP_HIP_BACKEND) {
        report.row("RPP vs the double half-pixel reference, outside RPP's clamp region",
                   compare(rpp.data(), expected.data(), n, OUT_W, OUT_H, outsideClampRegion), false);
    } else {
        report.row("RPP vs the double half-pixel reference, every pixel", compare(rpp.data(), expected.data(), n, OUT_W, OUT_H, everywhere), false);
    }
}

template <size_t N>
void runCase(Report& report, const std::vector<unsigned char>& frameHost, const Buffer<unsigned char>& frame, const bool mixed) {
    const RppBackend backend = frame.backend();
    std::cout << backendName(backend) << ": chain rppt_resize_mirror_normalize -> rppt_channel_permute -> rppt_copy, pipeline (a), N " << N
              << ", " << (mixed ? "mixed per-image parameters, permutation (1, 2, 0)" : "the benchmark's parameters") << std::endl;
    const uint32_t perms[2][3] = {{2, 1, 0}, {1, 2, 0}}; // the benchmark's, the mixed case's
    Parameters<N> prm(frameHost, mixed, perms[mixed ? 1 : 0], backend);
    Pinned<Rpp32u> permutation(3 * N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            permutation[3 * i + c] = prm.specs[i].perm[c];
        }
    }
    const size_t outCount = N * 3 * static_cast<size_t>(OUT_W) * OUT_H;
    RpptDesc outDesc(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
    hipStream_t stream{};
    FASTRPP_TEST_HIP(hipStreamCreate(&stream));

    // FastRPP: the frame in place, three IOps, one kernel
    RpptDesc frameDesc = fastrpp::sharedSourceDesc(N, 3, SRC_H, SRC_W, RpptDataType::U8, RpptLayout::NHWC, SRC_W * 3);
    Buffer<float> resized(outCount, backend), permuted(outCount, backend), fastOut(outCount, backend);
    resized.fillBytes(0xFF);
    permuted.fillBytes(0xFF);
    fastOut.fillBytes(0xFF);
    const rppHandle_t fastHandle = createHandle(N, backend, stream);
    const auto rmn = fastrpp::rppt_resize_mirror_normalize<N>(frame.data(), &frameDesc, resized.data(), &outDesc, prm.sizes.data(),
                                                              RpptInterpolationType::BILINEAR, prm.mean.data(), prm.stdDev.data(),
                                                              prm.mirror.data(), prm.rois.data(), RpptRoiType::XYWH);
    const auto cp = fastrpp::rppt_channel_permute(resized.data(), &outDesc, permuted.data(), &outDesc, permutation.data());
    const auto copy = fastrpp::rppt_copy(permuted.data(), &outDesc, fastOut.data(), &outDesc);
    fastrpp::executeOperations(fastHandle, backend, rmn, cp, copy);
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> fast = fastOut.download();
    report.check("the intermediate tensors are not written", holdsSentinel(resized.download(), SENTINEL) && holdsSentinel(permuted.download(), SENTINEL));
    FASTRPP_TEST_RPP(rppDestroy(fastHandle, backend));

    // RPP: the crops copied into a dense batch, then its two calls
    RppInput in(frame, N);
    Buffer<float> rppResized(outCount, backend), rppOut(outCount, backend);
    const rppHandle_t rppHandle = createHandle(N, backend, stream);
    FASTRPP_TEST_RPP(rppt_resize_mirror_normalize(in.crops.data(), &in.desc, rppResized.data(), &outDesc, prm.sizes.data(),
                                                  RpptInterpolationType::BILINEAR, prm.mean.data(), prm.stdDev.data(), prm.mirror.data(),
                                                  in.rois.data(), RpptRoiType::XYWH, rppHandle, backend));
    FASTRPP_TEST_RPP(rppt_channel_permute(rppResized.data(), &outDesc, rppOut.data(), &outDesc, permutation.data(), rppHandle, backend));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> rpp = rppOut.download();
    FASTRPP_TEST_RPP(rppDestroy(rppHandle, backend));
    FASTRPP_TEST_HIP(hipStreamDestroy(stream));
    rows(report, prm.specs, fast, rpp, backend);
}

// Two permutations: rppt_resize_mirror_normalize -> rppt_channel_permute (P1) -> rppt_channel_permute (P2) -> rppt_copy,
// one kernel, against RPP's three calls; mixed per-image parameters. Output channel k is the normalised channel
// P1[P2[k]]; none of the three intermediate tensors may be written.
template <size_t N>
void runTwoPermutations(Report& report, const std::vector<unsigned char>& frameHost, const Buffer<unsigned char>& frame) {
    const RppBackend backend = frame.backend();
    const uint32_t a[3] = {1, 2, 0}, b[3] = {0, 2, 1};
    std::cout << backendName(backend)
              << ": chain rppt_resize_mirror_normalize -> rppt_channel_permute (1, 2, 0) -> rppt_channel_permute (0, 2, 1) -> rppt_copy, "
                 "pipeline (a), N "
              << N << ", mixed per-image parameters" << std::endl;
    const uint32_t combined[3] = {a[b[0]], a[b[1]], a[b[2]]};
    Parameters<N> prm(frameHost, true, combined, backend);
    Pinned<Rpp32u> p1(3 * N), p2(3 * N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            p1[3 * i + c] = a[c];
            p2[3 * i + c] = b[c];
        }
    }
    const size_t outCount = N * 3 * static_cast<size_t>(OUT_W) * OUT_H;
    RpptDesc outDesc(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
    hipStream_t stream{};
    FASTRPP_TEST_HIP(hipStreamCreate(&stream));

    RpptDesc frameDesc = fastrpp::sharedSourceDesc(N, 3, SRC_H, SRC_W, RpptDataType::U8, RpptLayout::NHWC, SRC_W * 3);
    Buffer<float> mid1(outCount, backend), mid2(outCount, backend), mid3(outCount, backend), fastOut(outCount, backend);
    for (Buffer<float>* buffer : {&mid1, &mid2, &mid3, &fastOut}) {
        buffer->fillBytes(0xFF);
    }
    const rppHandle_t fastHandle = createHandle(N, backend, stream);
    const auto rmn = fastrpp::rppt_resize_mirror_normalize<N>(frame.data(), &frameDesc, mid1.data(), &outDesc, prm.sizes.data(),
                                                              RpptInterpolationType::BILINEAR, prm.mean.data(), prm.stdDev.data(),
                                                              prm.mirror.data(), prm.rois.data(), RpptRoiType::XYWH);
    const auto cp1 = fastrpp::rppt_channel_permute(mid1.data(), &outDesc, mid2.data(), &outDesc, p1.data());
    const auto cp2 = fastrpp::rppt_channel_permute(mid2.data(), &outDesc, mid3.data(), &outDesc, p2.data());
    const auto copy = fastrpp::rppt_copy(mid3.data(), &outDesc, fastOut.data(), &outDesc);
    fastrpp::executeOperations(fastHandle, backend, rmn, cp1, cp2, copy);
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> fast = fastOut.download();
    report.check("none of the three intermediate tensors is written", holdsSentinel(mid1.download(), SENTINEL) &&
                                                                          holdsSentinel(mid2.download(), SENTINEL) &&
                                                                          holdsSentinel(mid3.download(), SENTINEL));
    FASTRPP_TEST_RPP(rppDestroy(fastHandle, backend));

    RppInput in(frame, N);
    Buffer<float> r1(outCount, backend), r2(outCount, backend), rppOut(outCount, backend);
    const rppHandle_t rppHandle = createHandle(N, backend, stream);
    FASTRPP_TEST_RPP(rppt_resize_mirror_normalize(in.crops.data(), &in.desc, r1.data(), &outDesc, prm.sizes.data(), RpptInterpolationType::BILINEAR,
                                                  prm.mean.data(), prm.stdDev.data(), prm.mirror.data(), in.rois.data(), RpptRoiType::XYWH,
                                                  rppHandle, backend));
    FASTRPP_TEST_RPP(rppt_channel_permute(r1.data(), &outDesc, r2.data(), &outDesc, p1.data(), rppHandle, backend));
    FASTRPP_TEST_RPP(rppt_channel_permute(r2.data(), &outDesc, rppOut.data(), &outDesc, p2.data(), rppHandle, backend));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> rpp = rppOut.download();
    FASTRPP_TEST_RPP(rppDestroy(rppHandle, backend));
    FASTRPP_TEST_HIP(hipStreamDestroy(stream));
    rows(report, prm.specs, fast, rpp, backend);
}

} // namespace

int main() {
    try {
        // The benchmark's frame: seeded noise (LCG, seed 12345), dense rows
        const std::vector<unsigned char> frameHost = noiseImage(SRC_W, SRC_H, 12345u);
        Report report;
        for (const RppBackend backend : BACKENDS) {
            Buffer<unsigned char> frame(frameHost.size(), backend);
            frame.upload(frameHost);
            for (const bool mixed : {false, true}) {
                runCase<10>(report, frameHost, frame, mixed);
                runCase<300>(report, frameHost, frame, mixed);
            }
            runTwoPermutations<10>(report, frameHost, frame);
        }
        return report.exitCode();
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
