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

// The fused chain: fastrpp::executeOperations(handle, rppt_resize_mirror_normalize, rppt_channel_permute), one kernel,
// against RPP's two calls, on pipeline (a) of fkl-benchmarks (rpp/benchmark_rpp_preprocessing.h): N crops of 60x120 at
// (i, i) of a 3840x2160 frame of the benchmark's seeded noise, resized to 64x128. FastRPP reads the crops in place
// (shared-source batch); RPP reads them copied into a dense batch, RPP's documented batched form.
//   benchmark parameters  the benchmark's: mean = sub / 0.3, stdDev = div / 0.3 in the source channel order, mirror 0,
//                         permutation (2, 1, 0)
//   mixed parameters      per image a different mean and standard deviation, a mirror flag of 0, 1 or 2 (RPP mirrors
//                         only on 1) and one of the six permutations, all mixed within the batch
// Rows as in fastrpp_resize_mirror_normalize_vs_rpp_test, and the intermediate buffer (the destination of
// rppt_resize_mirror_normalize, the source of rppt_channel_permute) must not be written by the fused chain.
#include <common/test_common.h>

#include <memory>

using namespace fastrpp_test;

namespace {

constexpr int SRC_W = 3840, SRC_H = 2160, CROP_W = 60, CROP_H = 120, OUT_W = 64, OUT_H = 128;
constexpr float ALPHA = 0.3f;
constexpr float SUB[3] = {1.f, 4.f, 3.2f}, DIV[3] = {3.2f, 0.6f, 11.8f};

template <size_t N, typename Impl>
void runCase(Report& report, const std::vector<unsigned char>& frameHost, const Device<unsigned char>& frame, const bool mixed) {
    std::cout << "chain rppt_resize_mirror_normalize -> rppt_channel_permute, pipeline (a), N " << N << ", "
              << (mixed ? "mixed per-image parameters" : "the benchmark's parameters") << ", " << implementationName(Impl{}) << std::endl;
    const uint32_t perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    std::vector<ImageSpec> specs(N);
    Pinned<Rpp32f> mean(3 * N), stdDev(3 * N);
    Pinned<Rpp32u> mirror(N), permutation(3 * N);
    Pinned<RpptImagePatch> sizes(N);
    std::vector<RpptROI> rois(N);
    for (size_t i = 0; i < N; ++i) {
        ImageSpec& s = specs[i];
        s.image = frameHost.data();
        s.pitch = static_cast<size_t>(SRC_W) * 3;
        s.roiX = static_cast<int>(i);
        s.roiY = static_cast<int>(i);
        s.roiW = CROP_W;
        s.roiH = CROP_H;
        const uint32_t* const p = mixed ? perms[(i * 5) % 6] : perms[5];
        for (size_t c = 0; c < 3; ++c) {
            if (mixed) {
                s.mean[c] = static_cast<float>((i * 11 + c * 5) % 40);
                s.stdDev[c] = 1.0f + 0.25f * static_cast<float>((i + c) % 7);
            } else {
                // The permutation (2, 1, 0) comes after the normalisation: output channel 2 - c is (alpha * v_c - sub[2 - c]) / div[2 - c]
                s.mean[c] = SUB[2 - c] / ALPHA;
                s.stdDev[c] = DIV[2 - c] / ALPHA;
            }
            s.perm[c] = p[c];
            mean[3 * i + c] = s.mean[c];
            stdDev[3 * i + c] = s.stdDev[c];
            permutation[3 * i + c] = p[c];
        }
        s.mirror = mixed ? static_cast<uint32_t>(i % 3) : 0u;
        mirror[i] = s.mirror;
        sizes[i] = RpptImagePatch{OUT_W, OUT_H};
        rois[i].xywhROI = {{s.roiX, s.roiY}, CROP_W, CROP_H};
    }
    const size_t outCount = N * 3 * static_cast<size_t>(OUT_W) * OUT_H;
    RpptDesc outDesc(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
    hipStream_t stream{};
    FASTRPP_TEST_HIP(hipStreamCreate(&stream));

    // FastRPP: the frame in place, one kernel
    RpptDesc frameDesc = fastrpp::sharedSourceDesc(N, 3, SRC_H, SRC_W, RpptDataType::U8, RpptLayout::NHWC, SRC_W * 3);
    Device<float> resized(outCount), fastOut(outCount);
    resized.fillBytes(0xFF);
    fastOut.fillBytes(0xFF);
    fastrpp::rppHandle_t fastHandle{};
    FASTRPP_TEST_RPP(fastrpp::rppCreate(&fastHandle, N, 0, stream, RppBackend::RPP_HIP_BACKEND));
    const auto rmn = fastrpp::rppt_resize_mirror_normalize<N, RpptDataType::U8, RpptLayout::NHWC, RpptDataType::F32, RpptLayout::NCHW, 3, Impl>(
        frame.data(), &frameDesc, resized.data(), &outDesc, sizes.data(), RpptInterpolationType::BILINEAR, mean.data(), stdDev.data(),
        mirror.data(), rois.data(), RpptRoiType::XYWH, fastHandle, RppBackend::RPP_HIP_BACKEND);
    const auto cp = fastrpp::rppt_channel_permute<N, RpptDataType::F32, RpptLayout::NCHW, RpptLayout::NCHW, 3, Impl>(
        resized.data(), &outDesc, fastOut.data(), &outDesc, permutation.data(), fastHandle, RppBackend::RPP_HIP_BACKEND);
    report.check("both operations return RPP_SUCCESS", rmn.status == RPP_SUCCESS && cp.status == RPP_SUCCESS);
    FASTRPP_TEST_RPP(fastrpp::executeOperations(fastHandle, rmn, cp));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> fast = fastOut.download();
    const std::vector<float> intermediate = resized.download();
    bool untouched = true;
    for (const float v : intermediate) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        untouched &= bits == 0xFFFFFFFFu;
    }
    report.check("the intermediate buffer is not written", untouched);
    FASTRPP_TEST_RPP(fastrpp::rppDestroy(fastHandle, RppBackend::RPP_HIP_BACKEND));

    // RPP: the crops copied into a dense batch, then its two calls
    const size_t cropBytes = static_cast<size_t>(CROP_W) * CROP_H * 3;
    Device<unsigned char> crops(cropBytes * N);
    for (size_t i = 0; i < N; ++i) {
        FASTRPP_TEST_HIP(hipMemcpy2D(crops.data() + cropBytes * i, static_cast<size_t>(CROP_W) * 3,
                                     frame.data() + (i * SRC_W + i) * 3, static_cast<size_t>(SRC_W) * 3, static_cast<size_t>(CROP_W) * 3,
                                     CROP_H, hipMemcpyDeviceToDevice));
    }
    RpptDesc cropDesc(N, 3, CROP_H, CROP_W, RpptDataType::U8, RpptLayout::NHWC);
    Pinned<RpptROI> rppRois(rppRoiCapacity(N));
    for (size_t i = 0; i < N; ++i) {
        rppRois[i].xywhROI = {{0, 0}, CROP_W, CROP_H};
    }
    Device<float> rppResized(outCount), rppOut(outCount);
    rppHandle_t rppHandle{};
    FASTRPP_TEST_RPP(rppCreate(&rppHandle, N, 0, stream, RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_RPP(rppt_resize_mirror_normalize(crops.data(), &cropDesc, rppResized.data(), &outDesc, sizes.data(),
                                                  RpptInterpolationType::BILINEAR, mean.data(), stdDev.data(), mirror.data(), rppRois.data(),
                                                  RpptRoiType::XYWH, rppHandle, RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_RPP(rppt_channel_permute(rppResized.data(), &outDesc, rppOut.data(), &outDesc, permutation.data(), rppHandle,
                                          RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> rpp = rppOut.download();
    FASTRPP_TEST_RPP(rppDestroy(rppHandle, RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_HIP(hipStreamDestroy(stream));

    const size_t imageValues = 3 * static_cast<size_t>(OUT_W) * OUT_H;
    std::vector<double> expected(outCount);
    for (size_t i = 0; i < N; ++i) {
        referenceImage(specs[i], OUT_W, OUT_H, expected.data() + i * imageValues);
    }
    auto inClampRegion = [&](const size_t i, const int x, const int y) { return rppEdge(specs[i], OUT_W, OUT_H, x, y); };
    auto outsideClampRegion = [&](const size_t i, const int x, const int y) { return !rppEdge(specs[i], OUT_W, OUT_H, x, y); };
    auto everywhere = [](const size_t, const int, const int) { return true; };
    report.row("FastRPP vs RPP, outside RPP's clamp region", compare(fast.data(), rpp.data(), N, OUT_W, OUT_H, outsideClampRegion), true);
    report.row("FastRPP vs RPP, in RPP's clamp region", compare(fast.data(), rpp.data(), N, OUT_W, OUT_H, inClampRegion), false);
    report.row("FastRPP vs the double half-pixel reference, every pixel", compare(fast.data(), expected.data(), N, OUT_W, OUT_H, everywhere), true);
    report.row("RPP vs the double half-pixel reference, outside RPP's clamp region",
               compare(rpp.data(), expected.data(), N, OUT_W, OUT_H, outsideClampRegion), false);
}

template <typename Impl>
void runAll(Report& report, const std::vector<unsigned char>& frameHost, const Device<unsigned char>& frame) {
    for (const bool mixed : {false, true}) {
        runCase<10, Impl>(report, frameHost, frame, mixed);
        runCase<300, Impl>(report, frameHost, frame, mixed);
    }
}

} // namespace

int main() {
    try {
        // The benchmark's frame: seeded noise (LCG, seed 12345), dense rows
        const std::vector<unsigned char> frameHost = noiseImage(SRC_W, SRC_H, 12345u);
        Device<unsigned char> frame(frameHost.size());
        frame.upload(frameHost);
        Report report;
        runAll<fastrpp::UniformBranches>(report, frameHost, frame);
        runAll<fastrpp::Selects>(report, frameHost, frame);
        return report.exitCode();
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
