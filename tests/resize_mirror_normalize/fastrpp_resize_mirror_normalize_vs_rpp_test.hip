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

// fastrpp::rppt_resize_mirror_normalize against RPP's rppt_resize_mirror_normalize on the same inputs, on each execution
// back-end: FastRPP's read IOp runs with fastrpp::rppt_copy's write IOp, executeOperations(handle, backend, rmn, copy),
// one GPU kernel (RPP_HIP_BACKEND) or FKL's CPU executor on host memory (RPP_HOST_BACKEND), against RPP on the same
// back-end.
//
// Source U8 NHWC, destination F32 NCHW, BILINEAR; per image a different ROI position, mean and standard deviation. On the
// HIP back-end, a mirror flag of 0 or 2 mixed within the batch (RPP's HIP back-end mirrors only on 1; FastRPP rejects 1:
// fastrpp_errors_test); on the host back-end, 0 (RPP's host back-end mirrors on any value but 0). Two source forms:
//   dense batch    both libraries read the same tensor of N images
//   shared source  FastRPP reads N ROIs of one frame in place (nStride = 0, fastrpp::sharedSourceDesc); RPP reads the
//                  same crops copied into a dense batch, RPP's documented batched form
// Rows, HIP back-end: FastRPP vs RPP outside RPP's clamp region (required), inside it (informational: RPP's HIP kernels
// never read the last column and row of a ROI, README.md). Host back-end: FastRPP vs RPP on every pixel (required): RPP's
// host kernels clamp both neighbours into the ROI, as FKL does, so they have no clamp region. Both: FastRPP vs a
// double-precision reference of FKL's half-pixel sampling (required). The RPP rows are required when src / dst is exact
// in binary, so that both libraries compute the same source positions;
// otherwise they are informational, and FastRPP vs the reference is required within 1e-2 (both libraries compute the
// position in float, the reference in double: one float step of the position, times a gradient of at most 255 per
// pixel, times 1 / stdDev <= 1, stays far below 1e-2). RPP's output widths are multiples of 8 (its kernels write 8
// pixels per thread).
// A last case, on each back-end, is FastRPP's only, against the reference: an output width that is not a multiple of 8,
// padded source rows, offsetInBytes on the source and the destination, and guard bands around the destination that must
// stay intact.
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

template <size_t N>
void runCase(Report& report, const Case& c, const bool noise, const RpptRoiType roiType, const bool shared, const RppBackend backend) {
    std::cout << backendName(backend) << ": " << c.name << ", N " << N << ", " << (noise ? "noise" : "gradient") << ", "
              << (roiType == RpptRoiType::XYWH ? "XYWH" : "LTRB") << ", "
              << (shared ? "shared source (nStride 0) vs RPP on the crops copied into a dense batch" : "dense batch") << std::endl;
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
    Buffer<unsigned char> src(host.size(), backend);
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
        // HIP: 0 and 2 mixed, no mirror in either library; host: 0, as RPP's host back-end mirrors on 2
        mirror[i] = backend == RppBackend::RPP_HIP_BACKEND ? static_cast<Rpp32u>(((i + N) % 2) * 2) : 0u;
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
    Buffer<float> fastOut(outCount, backend);
    fastOut.fillBytes(0xFF); // NaN: an element FastRPP does not write fails every row
    const rppHandle_t fastHandle = createHandle(N, backend, stream);
    const std::vector<RpptROI> roisBefore = fastRois;
    const auto rmn = fastrpp::rppt_resize_mirror_normalize<N>(src.data(), &srcDesc, fastOut.data(), &dstDesc, sizes.data(),
                                                              RpptInterpolationType::BILINEAR, mean.data(), stdDev.data(), mirror.data(),
                                                              fastRois.data(), roiType);
    const auto copy = fastrpp::rppt_copy(fastOut.data(), &dstDesc, fastOut.data(), &dstDesc);
    fastrpp::executeOperations(fastHandle, backend, rmn, copy);
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> fast = fastOut.download();
    report.check("the ROI array is unchanged", std::memcmp(roisBefore.data(), fastRois.data(), sizeof(RpptROI) * N) == 0);
    FASTRPP_TEST_RPP(rppDestroy(fastHandle, backend));

    // RPP: the same tensor, or the crops copied into a dense batch; its ROI array holds 256 * ceil(N / 256) ROIs
    std::unique_ptr<Buffer<unsigned char>> crops;
    RpptDesc rppSrcDesc = srcDesc;
    unsigned char* rppSrc = src.data();
    Pinned<RpptROI> rppRois(rppRoiCapacity(N));
    for (size_t i = 0; i < N; ++i) {
        rppRois[i] = shared ? makeRoi(0, 0, c.roiW, c.roiH, roiType) : fastRois[i];
    }
    if (shared) {
        const size_t cropBytes = static_cast<size_t>(c.roiW) * c.roiH * 3;
        crops = std::make_unique<Buffer<unsigned char>>(cropBytes * N, backend);
        for (size_t i = 0; i < N; ++i) {
            copy2D(backend, crops->data() + cropBytes * i, static_cast<size_t>(c.roiW) * 3,
                   src.data() + static_cast<size_t>(specs[i].roiY) * specs[i].pitch + static_cast<size_t>(specs[i].roiX) * 3, specs[i].pitch,
                   static_cast<size_t>(c.roiW) * 3, static_cast<size_t>(c.roiH));
        }
        rppSrcDesc = RpptDesc(N, 3, static_cast<Rpp32u>(c.roiH), static_cast<Rpp32u>(c.roiW), RpptDataType::U8, RpptLayout::NHWC);
        rppSrc = crops->data();
    }
    Buffer<float> rppOut(outCount, backend);
    rppOut.fillBytes(0xFF);
    const rppHandle_t rppHandle = createHandle(N, backend, stream);
    FASTRPP_TEST_RPP(rppt_resize_mirror_normalize(rppSrc, &rppSrcDesc, rppOut.data(), &dstDesc, sizes.data(), RpptInterpolationType::BILINEAR,
                                                  mean.data(), stdDev.data(), mirror.data(), rppRois.data(), roiType, rppHandle, backend));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> rpp = rppOut.download();
    FASTRPP_TEST_RPP(rppDestroy(rppHandle, backend));
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
    if (backend == RppBackend::RPP_HIP_BACKEND) {
        report.row("FastRPP vs RPP, outside RPP's clamp region", compare(fast.data(), rpp.data(), N, c.dstW, c.dstH, outsideClampRegion),
                   c.required);
        report.row("FastRPP vs RPP, in RPP's clamp region", compare(fast.data(), rpp.data(), N, c.dstW, c.dstH, inClampRegion), false);
    } else {
        report.row("FastRPP vs RPP, every pixel (RPP's host kernels have no clamp region)",
                   compare(fast.data(), rpp.data(), N, c.dstW, c.dstH, everywhere), c.required);
    }
    report.row("FastRPP vs the double half-pixel reference, every pixel", compare(fast.data(), expected.data(), N, c.dstW, c.dstH, everywhere), true,
               c.required ? TOLERANCE : 1e-2);
    if (backend == RppBackend::RPP_HIP_BACKEND) {
        report.row("RPP vs the double half-pixel reference, outside RPP's clamp region",
                   compare(rpp.data(), expected.data(), N, c.dstW, c.dstH, outsideClampRegion), false);
    } else {
        report.row("RPP vs the double half-pixel reference, every pixel", compare(rpp.data(), expected.data(), N, c.dstW, c.dstH, everywhere), false);
    }
}

// FastRPP only, against the double reference: output 60x40 (60 is not a multiple of 8) from 30x20 ROIs (an exact 2x
// upscale), images of 47x33 pixels with padded rows (hStride = 3 * 47 + 5 bytes), offsetInBytes 64 on the source and 128
// on the destination of rppt_copy, the parameter arrays in pageable memory, and guard bands before and after that
// destination tensor
template <size_t N>
void runFastRppOnly(Report& report, const RppBackend backend) {
    std::cout << backendName(backend) << ": FastRPP only: 30x20 ROIs of 47x33 images with padded rows -> 60x40, offsets, guard bands, N " << N
              << std::endl;
    constexpr int W = 47, H = 33, ROI_W = 30, ROI_H = 20, DST_W = 60, DST_H = 40;
    constexpr Rpp32u H_STRIDE = 3 * W + 5, SRC_OFFSET = 64, DST_OFFSET = 128;
    constexpr size_t GUARD = 4096;
    const size_t nStride = static_cast<size_t>(H_STRIDE) * H;
    std::vector<unsigned char> host(SRC_OFFSET + N * nStride);
    uint32_t state = 99u;
    for (unsigned char& b : host) {
        state = state * 1664525u + 1013904223u;
        b = static_cast<unsigned char>(state >> 24);
    }
    Buffer<unsigned char> src(host.size(), backend);
    src.upload(host);
    std::vector<ImageSpec> specs(N);
    std::vector<Rpp32f> mean(3 * N), stdDev(3 * N);
    std::vector<Rpp32u> mirror(N);
    std::vector<RpptImagePatch> sizes(N, RpptImagePatch{DST_W, DST_H});
    std::vector<RpptROI> rois(N);
    for (size_t i = 0; i < N; ++i) {
        ImageSpec& s = specs[i];
        s.image = host.data() + SRC_OFFSET + i * nStride;
        s.pitch = H_STRIDE;
        s.roiX = static_cast<int>((i * 5) % static_cast<size_t>(W - ROI_W + 1));
        s.roiY = static_cast<int>((i * 3) % static_cast<size_t>(H - ROI_H + 1));
        s.roiW = ROI_W;
        s.roiH = ROI_H;
        for (size_t c = 0; c < 3; ++c) {
            s.mean[c] = static_cast<float>((i + c * 3) % 20);
            s.stdDev[c] = 1.0f + 0.5f * static_cast<float>((i * 2 + c) % 4);
            s.perm[c] = static_cast<uint32_t>(c);
            mean[3 * i + c] = s.mean[c];
            stdDev[3 * i + c] = s.stdDev[c];
        }
        mirror[i] = static_cast<Rpp32u>((i % 2) * 2);
        rois[i].xywhROI = {{s.roiX, s.roiY}, ROI_W, ROI_H};
    }
    RpptDesc srcDesc(N, 3, H, W, RpptDataType::U8, RpptLayout::NHWC, SRC_OFFSET, RpptStrides{static_cast<Rpp32u>(nStride), 1, H_STRIDE, 3});
    RpptDesc dstDesc(N, 3, DST_H, DST_W, RpptDataType::F32, RpptLayout::NCHW, DST_OFFSET,
                     RpptStrides{3 * DST_W * DST_H, DST_W * DST_H, DST_W, 1});
    const size_t tensorBytes = N * 3 * static_cast<size_t>(DST_W) * DST_H * sizeof(float);
    Buffer<unsigned char> dst(DST_OFFSET + tensorBytes + GUARD, backend);
    dst.fillBytes(0x5A);
    hipStream_t stream{};
    FASTRPP_TEST_HIP(hipStreamCreate(&stream));
    const rppHandle_t handle = createHandle(N, backend, stream);
    // The explicit template arguments are the defaults
    const auto rmn = fastrpp::rppt_resize_mirror_normalize<N, RpptDataType::U8, RpptLayout::NHWC, RpptDataType::F32, RpptLayout::NCHW, 3>(
        src.data(), &srcDesc, nullptr, &dstDesc, sizes.data(), RpptInterpolationType::BILINEAR, mean.data(), stdDev.data(), mirror.data(),
        rois.data(), RpptRoiType::XYWH); // its dstPtr is never accessed: nullptr is fine
    const auto copy = fastrpp::rppt_copy<RpptDataType::F32, RpptLayout::NCHW, RpptLayout::NCHW, 3>(nullptr, &dstDesc, dst.data(), &dstDesc);
    fastrpp::executeOperations(handle, backend, rmn, copy);
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    FASTRPP_TEST_RPP(rppDestroy(handle, backend));
    FASTRPP_TEST_HIP(hipStreamDestroy(stream));
    const std::vector<unsigned char> bytes = dst.download();
    bool guards = true;
    for (size_t k = 0; k < DST_OFFSET; ++k) {
        guards &= bytes[k] == 0x5A;
    }
    for (size_t k = DST_OFFSET + tensorBytes; k < bytes.size(); ++k) {
        guards &= bytes[k] == 0x5A;
    }
    report.check("the bytes before offsetInBytes and after the tensor are untouched", guards);
    std::vector<float> fast(tensorBytes / sizeof(float));
    std::memcpy(fast.data(), bytes.data() + DST_OFFSET, tensorBytes);
    const size_t imageValues = 3 * static_cast<size_t>(DST_W) * DST_H;
    std::vector<double> expected(fast.size());
    for (size_t i = 0; i < N; ++i) {
        referenceImage(specs[i], DST_W, DST_H, expected.data() + i * imageValues);
    }
    report.row("FastRPP vs the double half-pixel reference, every pixel",
               compare(fast.data(), expected.data(), N, DST_W, DST_H, [](const size_t, const int, const int) { return true; }), true);
}

const Case CASES[] = {
    {"60x120 -> 64x128 (the benchmark's crop, upscale)", 60, 120, 60, 120, 64, 128, true},
    {"640x360 -> 128x72 (downscale by 5)", 640, 360, 640, 360, 128, 72, true},
    {"97x53 ROI of 128x80 -> 64x32", 128, 80, 97, 53, 64, 32, true},
    {"16x16 ROI of 40x30 -> 64x64 (upscale by 4: a wide clamp region)", 40, 30, 16, 16, 64, 64, true},
    {"100x60 -> 48x40 (100 / 48 is not exact in binary: informational)", 100, 60, 100, 60, 48, 40, false},
};

} // namespace

int main() {
    try {
        Report report;
        for (const RppBackend backend : BACKENDS) {
            for (const Case& c : CASES) {
                runCase<1>(report, c, true, RpptRoiType::XYWH, false, backend);
                runCase<7>(report, c, false, RpptRoiType::LTRB, false, backend);
                runCase<7>(report, c, true, RpptRoiType::XYWH, true, backend);
            }
            runCase<300>(report, CASES[0], true, RpptRoiType::XYWH, true, backend);
            runCase<300>(report, CASES[0], true, RpptRoiType::LTRB, false, backend);
            runFastRppOnly<5>(report, backend);
        }
        return report.exitCode();
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
