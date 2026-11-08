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

// fastrpp::rppt_resize_mirror_normalize against RPP's rppt_resize_mirror_normalize on the same inputs, for both
// implementations of FastRPP's per-image mirror (UniformBranches, Selects).
//
// Source U8 NHWC, destination F32 NCHW, BILINEAR; per image a different ROI position, mean and standard deviation, and a
// mirror flag of 0, 1 or 2 (RPP mirrors only on 1), mixed within the batch. Two source forms:
//   dense batch    both libraries read the same tensor of N images
//   shared source  FastRPP reads N ROIs of one frame in place (nStride = 0, fastrpp::sharedSourceDesc); RPP reads the
//                  same crops copied into a dense batch, RPP's documented batched form
// Rows: FastRPP vs RPP outside RPP's clamp region (required), inside it (informational: RPP never reads the last column
// and row of a ROI, README.md), and FastRPP vs a double-precision reference of FKL's half-pixel sampling (required). The
// required rows need src / dst exact in binary for both libraries to compute the same source positions; the case that is
// not is informational. RPP's output widths are multiples of 8 (its kernels write 8 pixels per thread).
#include <common/test_common.h>

#include <memory>

using namespace fastrpp_test;

namespace {

struct Case {
    const char* name;
    int srcW, srcH;  // each image (dense) or a quarter of the frame (shared)
    int roiW, roiH;  // the ROI of every image
    int dstW, dstH;
    bool required;
};

RpptROI makeRoi(const int x, const int y, const int w, const int h, const RpptRoiType roiType) {
    RpptROI roi{};
    if (roiType == RpptRoiType::XYWH) {
        roi.xywhROI = {{x, y}, w, h};
    } else {
        roi.ltrbROI = {{x, y}, {x + w - 1, y + h - 1}};
    }
    return roi;
}

template <size_t N, typename Impl>
void runCase(Report& report, const Case& c, const bool noise, const RpptRoiType roiType, const bool shared) {
    std::cout << c.name << ", N " << N << ", " << (noise ? "noise" : "gradient") << ", " << (roiType == RpptRoiType::XYWH ? "XYWH" : "LTRB")
              << ", " << (shared ? "shared source (nStride 0) vs RPP on the crops copied into a dense batch" : "dense batch") << ", "
              << implementationName(Impl{}) << std::endl;
    const int frameW = shared ? 2 * c.srcW : c.srcW, frameH = shared ? 2 * c.srcH : c.srcH;
    const size_t imageBytes = static_cast<size_t>(frameW) * frameH * 3;
    std::vector<unsigned char> host;
    if (shared) {
        host = noise ? noiseImage(frameW, frameH, 777u) : gradientImage(frameW, frameH, 0);
    } else {
        host.resize(imageBytes * N);
        for (size_t i = 0; i < N; ++i) {
            const std::vector<unsigned char> img = noise ? noiseImage(frameW, frameH, 12345u + static_cast<uint32_t>(i))
                                                         : gradientImage(frameW, frameH, static_cast<int>(i));
            std::copy(img.begin(), img.end(), host.begin() + static_cast<std::ptrdiff_t>(i * imageBytes));
        }
    }
    Device<unsigned char> src(host.size());
    src.upload(host);

    // Per image: the ROI, the parameters, and the same values in the double-precision reference
    std::vector<ImageSpec> specs(N);
    Pinned<Rpp32f> mean(3 * N), stdDev(3 * N);
    Pinned<Rpp32u> mirror(N);
    Pinned<RpptImagePatch> sizes(N);
    std::vector<RpptROI> fastRois(N);
    for (size_t i = 0; i < N; ++i) {
        ImageSpec& s = specs[i];
        s.image = shared ? host.data() : host.data() + i * imageBytes;
        s.pitch = static_cast<size_t>(frameW) * 3;
        s.roiX = static_cast<int>((i * 7) % static_cast<size_t>(frameW - c.roiW + 1));
        s.roiY = static_cast<int>((i * 5) % static_cast<size_t>(frameH - c.roiH + 1));
        s.roiW = c.roiW;
        s.roiH = c.roiH;
        for (int ch = 0; ch < 3; ++ch) {
            s.mean[ch] = static_cast<float>((i * 13 + static_cast<size_t>(ch) * 7) % 50);
            s.stdDev[ch] = 1.0f + 0.5f * static_cast<float>((i + static_cast<size_t>(ch)) % 5); // |output| < 256
            s.perm[ch] = static_cast<uint32_t>(ch);
            mean[3 * i + static_cast<size_t>(ch)] = s.mean[ch];
            stdDev[3 * i + static_cast<size_t>(ch)] = s.stdDev[ch];
        }
        s.mirror = static_cast<uint32_t>((i + N) % 3); // 0, 1 and 2 mixed; RPP mirrors only on 1
        mirror[i] = s.mirror;
        sizes[i] = RpptImagePatch{static_cast<Rpp32u>(c.dstW), static_cast<Rpp32u>(c.dstH)};
        fastRois[i] = makeRoi(s.roiX, s.roiY, s.roiW, s.roiH, roiType);
    }
    const size_t outCount = N * 3 * static_cast<size_t>(c.dstW) * c.dstH;
    RpptDesc dstDesc(N, 3, static_cast<Rpp32u>(c.dstH), static_cast<Rpp32u>(c.dstW), RpptDataType::F32, RpptLayout::NCHW);
    hipStream_t stream{};
    FASTRPP_TEST_HIP(hipStreamCreate(&stream));

    // FastRPP: the parameter arrays are pinned (RPP needs them so) except the ROIs, in pageable memory
    RpptDesc srcDesc = shared ? fastrpp::sharedSourceDesc(N, 3, static_cast<Rpp32u>(frameH), static_cast<Rpp32u>(frameW), RpptDataType::U8,
                                                          RpptLayout::NHWC, static_cast<Rpp32u>(frameW) * 3)
                              : RpptDesc(N, 3, static_cast<Rpp32u>(frameH), static_cast<Rpp32u>(frameW), RpptDataType::U8, RpptLayout::NHWC);
    Device<float> fastOut(outCount);
    fastOut.fillBytes(0xFF); // NaN: an element FastRPP does not write fails every row
    fastrpp::rppHandle_t fastHandle{};
    FASTRPP_TEST_RPP(fastrpp::rppCreate(&fastHandle, N, 0, stream, RppBackend::RPP_HIP_BACKEND));
    const std::vector<RpptROI> roisBefore = fastRois;
    const auto op = fastrpp::rppt_resize_mirror_normalize<N, RpptDataType::U8, RpptLayout::NHWC, RpptDataType::F32, RpptLayout::NCHW, 3, Impl>(
        src.data(), &srcDesc, fastOut.data(), &dstDesc, sizes.data(), RpptInterpolationType::BILINEAR, mean.data(), stdDev.data(), mirror.data(),
        fastRois.data(), roiType, fastHandle, RppBackend::RPP_HIP_BACKEND);
    report.check("fastrpp::rppt_resize_mirror_normalize returns RPP_SUCCESS", op.status == RPP_SUCCESS);
    FASTRPP_TEST_RPP(fastrpp::executeOperations(fastHandle, op));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> fast = fastOut.download();
    report.check("the ROI array is unchanged", std::memcmp(roisBefore.data(), fastRois.data(), sizeof(RpptROI) * N) == 0);
    FASTRPP_TEST_RPP(fastrpp::rppDestroy(fastHandle, RppBackend::RPP_HIP_BACKEND));

    // RPP: the same tensor, or the crops copied into a dense batch; its ROI array holds 256 * ceil(N / 256) ROIs
    std::unique_ptr<Device<unsigned char>> crops;
    RpptDesc rppSrcDesc = srcDesc;
    unsigned char* rppSrc = src.data();
    Pinned<RpptROI> rppRois(rppRoiCapacity(N));
    for (size_t i = 0; i < N; ++i) {
        rppRois[i] = shared ? makeRoi(0, 0, c.roiW, c.roiH, roiType) : fastRois[i];
    }
    if (shared) {
        const size_t cropBytes = static_cast<size_t>(c.roiW) * c.roiH * 3;
        crops = std::make_unique<Device<unsigned char>>(cropBytes * N);
        for (size_t i = 0; i < N; ++i) {
            FASTRPP_TEST_HIP(hipMemcpy2D(crops->data() + cropBytes * i, static_cast<size_t>(c.roiW) * 3,
                                         src.data() + static_cast<size_t>(specs[i].roiY) * specs[i].pitch + static_cast<size_t>(specs[i].roiX) * 3,
                                         specs[i].pitch, static_cast<size_t>(c.roiW) * 3, static_cast<size_t>(c.roiH), hipMemcpyDeviceToDevice));
        }
        rppSrcDesc = RpptDesc(N, 3, static_cast<Rpp32u>(c.roiH), static_cast<Rpp32u>(c.roiW), RpptDataType::U8, RpptLayout::NHWC);
        rppSrc = crops->data();
    }
    Device<float> rppOut(outCount);
    rppOut.fillBytes(0xFF);
    rppHandle_t rppHandle{};
    FASTRPP_TEST_RPP(rppCreate(&rppHandle, N, 0, stream, RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_RPP(rppt_resize_mirror_normalize(rppSrc, &rppSrcDesc, rppOut.data(), &dstDesc, sizes.data(), RpptInterpolationType::BILINEAR,
                                                  mean.data(), stdDev.data(), mirror.data(), rppRois.data(), roiType, rppHandle,
                                                  RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> rpp = rppOut.download();
    FASTRPP_TEST_RPP(rppDestroy(rppHandle, RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_HIP(hipStreamDestroy(stream));

    // The double-precision reference, and the rows
    const size_t imageValues = 3 * static_cast<size_t>(c.dstW) * c.dstH;
    std::vector<double> expected(outCount);
    for (size_t i = 0; i < N; ++i) {
        referenceImage(specs[i], c.dstW, c.dstH, expected.data() + i * imageValues);
    }
    auto inClampRegion = [&](const size_t i, const int x, const int y) { return rppEdge(specs[i], c.dstW, c.dstH, x, y); };
    auto outsideClampRegion = [&](const size_t i, const int x, const int y) { return !rppEdge(specs[i], c.dstW, c.dstH, x, y); };
    auto everywhere = [](const size_t, const int, const int) { return true; };
    report.row("FastRPP vs RPP, outside RPP's clamp region", compare(fast.data(), rpp.data(), N, c.dstW, c.dstH, outsideClampRegion), c.required);
    report.row("FastRPP vs RPP, in RPP's clamp region", compare(fast.data(), rpp.data(), N, c.dstW, c.dstH, inClampRegion), false);
    report.row("FastRPP vs the double half-pixel reference, every pixel", compare(fast.data(), expected.data(), N, c.dstW, c.dstH, everywhere),
               c.required);
    report.row("RPP vs the double half-pixel reference, outside RPP's clamp region",
               compare(rpp.data(), expected.data(), N, c.dstW, c.dstH, outsideClampRegion), false);
}

const Case CASES[] = {
    {"60x120 -> 64x128 (the benchmark's crop, upscale)", 60, 120, 60, 120, 64, 128, true},
    {"640x360 -> 128x72 (downscale by 5)", 640, 360, 640, 360, 128, 72, true},
    {"97x53 ROI of 128x80 -> 64x32", 128, 80, 97, 53, 64, 32, true},
    {"16x16 ROI of 40x30 -> 64x64 (upscale by 4: a wide clamp region)", 40, 30, 16, 16, 64, 64, true},
    {"100x60 -> 48x40 (100 / 48 is not exact in binary: informational)", 100, 60, 100, 60, 48, 40, false},
};

template <typename Impl>
void runAll(Report& report) {
    for (const Case& c : CASES) {
        runCase<1, Impl>(report, c, true, RpptRoiType::XYWH, false);
        runCase<7, Impl>(report, c, false, RpptRoiType::LTRB, false);
        runCase<7, Impl>(report, c, true, RpptRoiType::XYWH, true);
    }
    runCase<300, Impl>(report, CASES[0], true, RpptRoiType::XYWH, true);
    runCase<300, Impl>(report, CASES[0], true, RpptRoiType::LTRB, false);
}

} // namespace

int main() {
    try {
        Report report;
        runAll<fastrpp::UniformBranches>(report);
        runAll<fastrpp::Selects>(report);
        return report.exitCode();
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
