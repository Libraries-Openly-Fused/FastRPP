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

// The handle and every documented status: fastrpp::rppCreate / rppDestroy, the checks of each operation, and
// executeOperations, which returns the first failure and launches nothing (the destination keeps its sentinel).
#include <common/test_common.h>

using namespace fastrpp_test;

namespace {

constexpr size_t N = 4;
constexpr int W = 32, H = 24, OUT_W = 16, OUT_H = 8;

struct Fixture {
    Device<unsigned char> src{N * W * H * 3};
    Device<float> mid{N * 3 * OUT_W * OUT_H};
    Device<float> out{N * 3 * OUT_W * OUT_H};
    Pinned<Rpp32f> mean{3 * N}, stdDev{3 * N};
    Pinned<Rpp32u> mirror{N}, permutation{3 * N};
    Pinned<RpptImagePatch> sizes{N};
    std::vector<RpptROI> rois = std::vector<RpptROI>(N);
    RpptDesc srcDesc{N, 3, H, W, RpptDataType::U8, RpptLayout::NHWC};
    RpptDesc outDesc{N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW};
    fastrpp::rppHandle_t handle{};

    Fixture() {
        for (size_t i = 0; i < N; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                mean[3 * i + c] = 1.0f;
                stdDev[3 * i + c] = 2.0f;
                permutation[3 * i + c] = static_cast<Rpp32u>(2 - c);
            }
            mirror[i] = 0;
            sizes[i] = RpptImagePatch{OUT_W, OUT_H};
            rois[i].xywhROI = {{0, 0}, W, H};
        }
        FASTRPP_TEST_RPP(fastrpp::rppCreate(&handle, N, 0, nullptr, RppBackend::RPP_HIP_BACKEND));
        out.fillBytes(0x5A);
    }
    ~Fixture() { static_cast<void>(fastrpp::rppDestroy(handle, RppBackend::RPP_HIP_BACKEND)); }

    auto rmn(RpptDesc* s, RpptDesc* d, const RpptInterpolationType interpolation = RpptInterpolationType::BILINEAR,
             const RpptRoiType roiType = RpptRoiType::XYWH, const RppBackend backend = RppBackend::RPP_HIP_BACKEND) {
        return fastrpp::rppt_resize_mirror_normalize<N>(src.data(), s, mid.data(), d, sizes.data(), interpolation, mean.data(), stdDev.data(),
                                                        mirror.data(), rois.data(), roiType, handle, backend);
    }
    auto permute(RpptDesc* s, RpptDesc* d, void* srcPtr = nullptr) {
        return fastrpp::rppt_channel_permute<N>(srcPtr ? srcPtr : mid.data(), s, out.data(), d, permutation.data(), handle,
                                                RppBackend::RPP_HIP_BACKEND);
    }
    // The destination still holds its sentinel: nothing was launched
    bool outUntouched() const {
        FASTRPP_TEST_HIP(hipDeviceSynchronize());
        const std::vector<float> host = out.download();
        for (const float v : host) {
            uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            if (bits != 0x5A5A5A5Au) {
                return false;
            }
        }
        return true;
    }
};

void expect(Report& report, const std::string& what, const int status, const int expected) {
    report.check(what + " -> " + std::to_string(expected) + " (got " + std::to_string(status) + ")", status == expected);
}

} // namespace

int main() {
    try {
        Report report;
        std::cout << "handle" << std::endl;
        fastrpp::rppHandle_t h{};
        expect(report, "rppCreate with a null handle pointer", fastrpp::rppCreate(nullptr, 1, 0, nullptr, RppBackend::RPP_HIP_BACKEND),
               rppStatusBadParm);
        expect(report, "rppCreate with batch size 0", fastrpp::rppCreate(&h, 0, 0, nullptr, RppBackend::RPP_HIP_BACKEND), rppStatusBadParm);
        expect(report, "rppCreate with the host back-end", fastrpp::rppCreate(&h, 1, 0, nullptr, RppBackend::RPP_HOST_BACKEND),
               rppStatusNotImplemented);
        expect(report, "rppCreate with the HIP back-end", fastrpp::rppCreate(&h, 1, 0, nullptr, RppBackend::RPP_HIP_BACKEND), rppStatusSuccess);
        expect(report, "rppDestroy with the host back-end", fastrpp::rppDestroy(h, RppBackend::RPP_HOST_BACKEND), rppStatusNotImplemented);
        expect(report, "rppDestroy", fastrpp::rppDestroy(h, RppBackend::RPP_HIP_BACKEND), rppStatusSuccess);
        expect(report, "rppDestroy of a null handle", fastrpp::rppDestroy(nullptr, RppBackend::RPP_HIP_BACKEND), rppStatusBadParm);

        std::cout << "rppt_resize_mirror_normalize" << std::endl;
        Fixture f;
        expect(report, "valid arguments", f.rmn(&f.srcDesc, &f.outDesc).status, RPP_SUCCESS);
        expect(report, "host back-end", f.rmn(&f.srcDesc, &f.outDesc, RpptInterpolationType::BILINEAR, RpptRoiType::XYWH, RppBackend::RPP_HOST_BACKEND).status,
               RPP_ERROR_NOT_IMPLEMENTED);
        expect(report, "NEAREST_NEIGHBOR", f.rmn(&f.srcDesc, &f.outDesc, RpptInterpolationType::NEAREST_NEIGHBOR).status, RPP_ERROR_NOT_IMPLEMENTED);
        {
            RpptDesc s = f.srcDesc;
            s.numDims = 3;
            expect(report, "source numDims 3", f.rmn(&s, &f.outDesc).status, RPP_ERROR_INVALID_SRC_DIMS);
        }
        {
            RpptDesc s(N, 3, H, W, RpptDataType::F32, RpptLayout::NHWC);
            expect(report, "source F32", f.rmn(&s, &f.outDesc).status, RPP_ERROR_INVALID_SRC_DATATYPE);
        }
        {
            RpptDesc s(N, 3, H, W, RpptDataType::U8, RpptLayout::NCHW);
            expect(report, "source NCHW", f.rmn(&s, &f.outDesc).status, RPP_ERROR_INVALID_SRC_LAYOUT);
        }
        {
            RpptDesc s(N, 1, H, W, RpptDataType::U8, RpptLayout::NHWC);
            expect(report, "source with 1 channel", f.rmn(&s, &f.outDesc).status, RPP_ERROR_INVALID_SRC_CHANNELS);
        }
        {
            RpptDesc s(N - 1, 3, H, W, RpptDataType::U8, RpptLayout::NHWC);
            expect(report, "source n != BATCH", f.rmn(&s, &f.outDesc).status, RPP_ERROR_INVALID_ARGUMENTS);
        }
        {
            RpptDesc s = f.srcDesc;
            s.strides.wStride = 4;
            expect(report, "source pixels not packed (wStride 4)", f.rmn(&s, &f.outDesc).status, RPP_ERROR_INVALID_SRC_LAYOUT);
        }
        {
            RpptDesc d(N, 3, OUT_H, OUT_W, RpptDataType::F16, RpptLayout::NCHW);
            expect(report, "destination F16", f.rmn(&f.srcDesc, &d).status, RPP_ERROR_INVALID_DST_DATATYPE);
        }
        {
            RpptDesc d(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NHWC);
            expect(report, "destination NHWC", f.rmn(&f.srcDesc, &d).status, RPP_ERROR_INVALID_DST_LAYOUT);
        }
        {
            RpptDesc d = f.outDesc;
            d.strides.hStride = OUT_W + 8;
            d.strides.cStride = (OUT_W + 8) * OUT_H;
            d.strides.nStride = 3 * (OUT_W + 8) * OUT_H;
            expect(report, "destination with padded rows", f.rmn(&f.srcDesc, &d).status, RPP_ERROR_INVALID_DST_LAYOUT);
        }
        {
            RpptDesc d = f.outDesc;
            d.strides.nStride = 0;
            expect(report, "destination nStride 0", f.rmn(&f.srcDesc, &d).status, RPP_ERROR_INVALID_DST_LAYOUT);
        }
        f.sizes[1] = RpptImagePatch{OUT_W - 8, OUT_H};
        expect(report, "per-image output sizes", f.rmn(&f.srcDesc, &f.outDesc).status, RPP_ERROR_NOT_IMPLEMENTED);
        f.sizes[1] = RpptImagePatch{OUT_W, OUT_H};
        f.rois[2].xywhROI = {{4, 0}, W, H};
        expect(report, "ROI out of the image", f.rmn(&f.srcDesc, &f.outDesc).status, RPP_ERROR_OUT_OF_BOUND_SRC_ROI);
        f.rois[2].xywhROI = {{-1, 0}, 8, 8};
        expect(report, "ROI at x = -1", f.rmn(&f.srcDesc, &f.outDesc).status, RPP_ERROR_OUT_OF_BOUND_SRC_ROI);
        f.rois[2].xywhROI = {{0, 0}, 0, 8};
        expect(report, "ROI of width 0", f.rmn(&f.srcDesc, &f.outDesc).status, RPP_ERROR_OUT_OF_BOUND_SRC_ROI);
        f.rois[2].xywhROI = {{0, 0}, W, H};
        for (size_t i = 0; i < N; ++i) {
            f.rois[i].ltrbROI = {{0, 0}, {W - 1, H - 1}}; // LTRB is inclusive: the whole image
        }
        expect(report, "LTRB ROIs of the whole image", f.rmn(&f.srcDesc, &f.outDesc, RpptInterpolationType::BILINEAR, RpptRoiType::LTRB).status,
               RPP_SUCCESS);
        f.rois[3].ltrbROI = {{2, 3}, {W, H - 1}};
        expect(report, "an LTRB ROI one column past the image", f.rmn(&f.srcDesc, &f.outDesc, RpptInterpolationType::BILINEAR, RpptRoiType::LTRB).status,
               RPP_ERROR_OUT_OF_BOUND_SRC_ROI);
        for (size_t i = 0; i < N; ++i) {
            f.rois[i].xywhROI = {{0, 0}, W, H};
        }
        {
            fastrpp::rppHandle_t other{};
            FASTRPP_TEST_RPP(fastrpp::rppCreate(&other, N + 1, 0, nullptr, RppBackend::RPP_HIP_BACKEND));
            const auto op = fastrpp::rppt_resize_mirror_normalize<N>(f.src.data(), &f.srcDesc, f.mid.data(), &f.outDesc, f.sizes.data(),
                                                                     RpptInterpolationType::BILINEAR, f.mean.data(), f.stdDev.data(),
                                                                     f.mirror.data(), f.rois.data(), RpptRoiType::XYWH, other,
                                                                     RppBackend::RPP_HIP_BACKEND);
            expect(report, "handle batch size != BATCH", op.status, RPP_ERROR_INVALID_ARGUMENTS);
            FASTRPP_TEST_RPP(fastrpp::rppDestroy(other, RppBackend::RPP_HIP_BACKEND));
        }

        std::cout << "rppt_channel_permute" << std::endl;
        expect(report, "valid arguments", f.permute(&f.outDesc, &f.outDesc).status, RPP_SUCCESS);
        {
            RpptDesc d(N, 3, OUT_H, OUT_W + 8, RpptDataType::F32, RpptLayout::NCHW);
            expect(report, "source and destination of different sizes", f.permute(&f.outDesc, &d).status, RPP_ERROR_INVALID_DIM_LENGTHS);
        }
        {
            RpptDesc s(N, 3, OUT_H, OUT_W, RpptDataType::U8, RpptLayout::NCHW);
            expect(report, "source U8", f.permute(&s, &f.outDesc).status, RPP_ERROR_INVALID_SRC_DATATYPE);
        }
        f.permutation[3] = 0;
        f.permutation[4] = 0;
        f.permutation[5] = 1;
        expect(report, "triple (0, 0, 1)", f.permute(&f.outDesc, &f.outDesc).status, RPP_ERROR_INVALID_ARGUMENTS);
        f.permutation[3] = 3;
        f.permutation[4] = 1;
        f.permutation[5] = 0;
        expect(report, "triple (3, 1, 0)", f.permute(&f.outDesc, &f.outDesc).status, RPP_ERROR_INVALID_ARGUMENTS);

        std::cout << "executeOperations: the first failure, nothing launched" << std::endl;
        const auto badPermute = f.permute(&f.outDesc, &f.outDesc);
        expect(report, "an operation that failed its checks", fastrpp::executeOperations(f.handle, badPermute), RPP_ERROR_INVALID_ARGUMENTS);
        report.check("nothing was launched", f.outUntouched());
        f.permutation[3] = 2;
        f.permutation[4] = 1;
        f.permutation[5] = 0;
        const auto good = f.rmn(&f.srcDesc, &f.outDesc);
        const auto wrongSource = f.permute(&f.outDesc, &f.outDesc, f.out.data()); // reads out, not rmn's destination mid
        expect(report, "a broken chain (the permutation does not read the resize's destination)",
               fastrpp::executeOperations(f.handle, good, wrongSource), RPP_ERROR_INVALID_ARGUMENTS);
        report.check("nothing was launched", f.outUntouched());
        expect(report, "a null handle", fastrpp::executeOperations(nullptr, good), RPP_ERROR_INVALID_ARGUMENTS);
        const auto chained = f.permute(&f.outDesc, &f.outDesc);
        expect(report, "a valid chain", fastrpp::executeOperations(f.handle, good, chained), RPP_SUCCESS);
        report.check("the valid chain wrote its destination", !f.outUntouched());
        return report.exitCode();
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
