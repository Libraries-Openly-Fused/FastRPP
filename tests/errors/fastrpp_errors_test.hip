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

// The handle and the documented errors: fastrpp::rppCreate / rppDestroy return RPP's statuses; every check of the
// functions that build IOps throws fastrpp::Error with RPP's status; executeOperations throws on a null handle. A valid
// chain writes only the destination of rppt_copy: the intermediate tensor `mid` (the destination of
// rppt_resize_mirror_normalize, the source and destination of rppt_channel_permute, the source of rppt_copy) holds a
// sentinel that must survive.
#include <common/test_common.h>

using namespace fastrpp_test;

namespace {

constexpr size_t N = 4;
constexpr int W = 32, H = 24, OUT_W = 16, OUT_H = 8;
constexpr uint32_t SENTINEL = 0x5A5A5A5Au;

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
        mid.fillBytes(0x5A);
        out.fillBytes(0x5A);
    }
    ~Fixture() { static_cast<void>(fastrpp::rppDestroy(handle, RppBackend::RPP_HIP_BACKEND)); }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;

    auto rmn(RpptDesc* s, RpptDesc* d, const RpptInterpolationType interpolation = RpptInterpolationType::BILINEAR,
             const RpptRoiType roiType = RpptRoiType::XYWH, const RppBackend backend = RppBackend::RPP_HIP_BACKEND) {
        return fastrpp::rppt_resize_mirror_normalize<N>(src.data(), s, mid.data(), d, sizes.data(), interpolation, mean.data(), stdDev.data(),
                                                        mirror.data(), rois.data(), roiType, handle, backend);
    }
    auto permute(RpptDesc* s, RpptDesc* d, fastrpp::rppHandle_t h = nullptr, const RppBackend backend = RppBackend::RPP_HIP_BACKEND) {
        return fastrpp::rppt_channel_permute(mid.data(), s, mid.data(), d, permutation.data(), h ? h : handle, backend);
    }
    auto copy(RpptDesc* s, RpptDesc* d, void* const dstPtr, fastrpp::rppHandle_t h = nullptr,
              const RppBackend backend = RppBackend::RPP_HIP_BACKEND) {
        return fastrpp::rppt_copy(mid.data(), s, dstPtr, d, h ? h : handle, backend);
    }
};

void expectStatus(Report& report, const std::string& what, const int status, const int expected) {
    report.check(what + " -> " + std::to_string(expected) + " (got " + std::to_string(status) + ")", status == expected);
}

// f builds an IOp (or runs executeOperations); it must throw fastrpp::Error with the expected status, or nothing for
// RPP_SUCCESS
template <typename F>
void expect(Report& report, const std::string& what, F&& f, const RppStatus expected) {
    std::string message;
    const RppStatus got = statusOf(std::forward<F>(f), message);
    report.check(what + " -> " + std::to_string(static_cast<int>(expected)) + " (got " + std::to_string(static_cast<int>(got)) + ": " +
                     message + ")",
                 got == expected);
}

} // namespace

int main() {
    try {
        Report report;
        std::cout << "handle" << std::endl;
        fastrpp::rppHandle_t h{};
        expectStatus(report, "rppCreate with a null handle pointer", fastrpp::rppCreate(nullptr, 1, 0, nullptr, RppBackend::RPP_HIP_BACKEND),
                     rppStatusBadParm);
        expectStatus(report, "rppCreate with batch size 0", fastrpp::rppCreate(&h, 0, 0, nullptr, RppBackend::RPP_HIP_BACKEND), rppStatusBadParm);
        expectStatus(report, "rppCreate with the host back-end", fastrpp::rppCreate(&h, 1, 0, nullptr, RppBackend::RPP_HOST_BACKEND),
                     rppStatusNotImplemented);
        expectStatus(report, "rppCreate with the HIP back-end", fastrpp::rppCreate(&h, 1, 0, nullptr, RppBackend::RPP_HIP_BACKEND),
                     rppStatusSuccess);
        expectStatus(report, "rppDestroy with the default (host) back-end value frees the handle, as RPP's GPU build does",
                     fastrpp::rppDestroy(h), rppStatusSuccess);
        expectStatus(report, "rppCreate with the HIP back-end, again", fastrpp::rppCreate(&h, 1, 0, nullptr, RppBackend::RPP_HIP_BACKEND),
                     rppStatusSuccess);
        expectStatus(report, "rppDestroy with the HIP back-end", fastrpp::rppDestroy(h, RppBackend::RPP_HIP_BACKEND), rppStatusSuccess);
        expectStatus(report, "rppDestroy of a null handle", fastrpp::rppDestroy(nullptr, RppBackend::RPP_HIP_BACKEND), rppStatusBadParm);

        Fixture f;
        fastrpp::rppHandle_t other{};
        FASTRPP_TEST_RPP(fastrpp::rppCreate(&other, N + 1, 0, nullptr, RppBackend::RPP_HIP_BACKEND));

        std::cout << "rppt_resize_mirror_normalize" << std::endl;
        expect(report, "valid arguments", [&] { f.rmn(&f.srcDesc, &f.outDesc); }, RPP_SUCCESS);
        expect(report, "host back-end",
               [&] { f.rmn(&f.srcDesc, &f.outDesc, RpptInterpolationType::BILINEAR, RpptRoiType::XYWH, RppBackend::RPP_HOST_BACKEND); },
               RPP_ERROR_NOT_IMPLEMENTED);
        expect(report, "NEAREST_NEIGHBOR", [&] { f.rmn(&f.srcDesc, &f.outDesc, RpptInterpolationType::NEAREST_NEIGHBOR); },
               RPP_ERROR_NOT_IMPLEMENTED);
        expect(report, "a null destination descriptor", [&] { f.rmn(&f.srcDesc, nullptr); }, RPP_ERROR_INVALID_ARGUMENTS);
        expect(report, "a null handle",
               [&] {
                   fastrpp::rppt_resize_mirror_normalize<N>(f.src.data(), &f.srcDesc, f.mid.data(), &f.outDesc, f.sizes.data(),
                                                            RpptInterpolationType::BILINEAR, f.mean.data(), f.stdDev.data(), f.mirror.data(),
                                                            f.rois.data(), RpptRoiType::XYWH, nullptr, RppBackend::RPP_HIP_BACKEND);
               },
               RPP_ERROR_INVALID_ARGUMENTS);
        expect(report, "a handle whose batch size is not BATCH",
               [&] {
                   fastrpp::rppt_resize_mirror_normalize<N>(f.src.data(), &f.srcDesc, f.mid.data(), &f.outDesc, f.sizes.data(),
                                                            RpptInterpolationType::BILINEAR, f.mean.data(), f.stdDev.data(), f.mirror.data(),
                                                            f.rois.data(), RpptRoiType::XYWH, other, RppBackend::RPP_HIP_BACKEND);
               },
               RPP_ERROR_INVALID_ARGUMENTS);
        {
            RpptDesc s = f.srcDesc;
            s.numDims = 3;
            expect(report, "source numDims 3", [&] { f.rmn(&s, &f.outDesc); }, RPP_ERROR_INVALID_SRC_DIMS);
        }
        {
            RpptDesc s(N, 3, H, W, RpptDataType::F32, RpptLayout::NHWC);
            expect(report, "source F32", [&] { f.rmn(&s, &f.outDesc); }, RPP_ERROR_INVALID_SRC_DATATYPE);
        }
        {
            RpptDesc s(N, 3, H, W, RpptDataType::U8, RpptLayout::NCHW);
            expect(report, "source NCHW", [&] { f.rmn(&s, &f.outDesc); }, RPP_ERROR_INVALID_SRC_LAYOUT);
        }
        {
            RpptDesc s(N, 1, H, W, RpptDataType::U8, RpptLayout::NHWC);
            expect(report, "source with 1 channel", [&] { f.rmn(&s, &f.outDesc); }, RPP_ERROR_INVALID_SRC_CHANNELS);
        }
        {
            RpptDesc s(N - 1, 3, H, W, RpptDataType::U8, RpptLayout::NHWC);
            expect(report, "source n != BATCH", [&] { f.rmn(&s, &f.outDesc); }, RPP_ERROR_INVALID_ARGUMENTS);
        }
        {
            RpptDesc s(N, 3, H, 0, RpptDataType::U8, RpptLayout::NHWC);
            expect(report, "source of width 0", [&] { f.rmn(&s, &f.outDesc); }, RPP_ERROR_INVALID_DIM_LENGTHS);
        }
        {
            RpptDesc s = f.srcDesc;
            s.strides.wStride = 4;
            expect(report, "source pixels not packed (wStride 4)", [&] { f.rmn(&s, &f.outDesc); }, RPP_ERROR_INVALID_SRC_LAYOUT);
        }
        {
            RpptDesc s = f.srcDesc;
            s.strides.hStride = 3 * W - 1;
            expect(report, "source rows shorter than 3 * w", [&] { f.rmn(&s, &f.outDesc); }, RPP_ERROR_INVALID_SRC_LAYOUT);
        }
        {
            RpptDesc d(N, 3, OUT_H, OUT_W, RpptDataType::F16, RpptLayout::NCHW);
            expect(report, "destination F16", [&] { f.rmn(&f.srcDesc, &d); }, RPP_ERROR_INVALID_DST_DATATYPE);
        }
        {
            RpptDesc d(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NHWC);
            expect(report, "destination NHWC", [&] { f.rmn(&f.srcDesc, &d); }, RPP_ERROR_INVALID_DST_LAYOUT);
        }
        {
            RpptDesc d = f.outDesc;
            d.strides.hStride = OUT_W + 8;
            d.strides.cStride = (OUT_W + 8) * OUT_H;
            d.strides.nStride = 3 * (OUT_W + 8) * OUT_H;
            expect(report, "a destination with padded rows: accepted, the read IOp writes nothing", [&] { f.rmn(&f.srcDesc, &d); }, RPP_SUCCESS);
        }
        f.sizes[1] = RpptImagePatch{OUT_W - 8, OUT_H};
        expect(report, "per-image output sizes", [&] { f.rmn(&f.srcDesc, &f.outDesc); }, RPP_ERROR_NOT_IMPLEMENTED);
        f.sizes[1] = RpptImagePatch{OUT_W, OUT_H};
        f.mirror[2] = 1;
        expect(report, "an image mirrored (mirrorTensor[2] == 1)", [&] { f.rmn(&f.srcDesc, &f.outDesc); }, RPP_ERROR_NOT_IMPLEMENTED);
        f.mirror[2] = 2;
        expect(report, "mirrorTensor[2] == 2: no mirror, as in RPP", [&] { f.rmn(&f.srcDesc, &f.outDesc); }, RPP_SUCCESS);
        f.mirror[2] = 0;
        f.rois[2].xywhROI = {{4, 0}, W, H};
        expect(report, "ROI out of the image", [&] { f.rmn(&f.srcDesc, &f.outDesc); }, RPP_ERROR_OUT_OF_BOUND_SRC_ROI);
        f.rois[2].xywhROI = {{-1, 0}, 8, 8};
        expect(report, "ROI at x = -1", [&] { f.rmn(&f.srcDesc, &f.outDesc); }, RPP_ERROR_OUT_OF_BOUND_SRC_ROI);
        f.rois[2].xywhROI = {{0, 0}, 0, 8};
        expect(report, "ROI of width 0", [&] { f.rmn(&f.srcDesc, &f.outDesc); }, RPP_ERROR_OUT_OF_BOUND_SRC_ROI);
        f.rois[2].xywhROI = {{0, 0}, W, H};
        for (size_t i = 0; i < N; ++i) {
            f.rois[i].ltrbROI = {{0, 0}, {W - 1, H - 1}}; // LTRB is inclusive: the whole image
        }
        expect(report, "LTRB ROIs of the whole image", [&] { f.rmn(&f.srcDesc, &f.outDesc, RpptInterpolationType::BILINEAR, RpptRoiType::LTRB); },
               RPP_SUCCESS);
        f.rois[3].ltrbROI = {{2, 3}, {W, H - 1}};
        expect(report, "an LTRB ROI one column past the image",
               [&] { f.rmn(&f.srcDesc, &f.outDesc, RpptInterpolationType::BILINEAR, RpptRoiType::LTRB); }, RPP_ERROR_OUT_OF_BOUND_SRC_ROI);
        for (size_t i = 0; i < N; ++i) {
            f.rois[i].xywhROI = {{0, 0}, W, H};
        }

        std::cout << "rppt_channel_permute" << std::endl;
        expect(report, "valid arguments", [&] { f.permute(&f.outDesc, &f.outDesc); }, RPP_SUCCESS);
        expect(report, "host back-end", [&] { f.permute(&f.outDesc, &f.outDesc, nullptr, RppBackend::RPP_HOST_BACKEND); },
               RPP_ERROR_NOT_IMPLEMENTED);
        expect(report, "a handle whose batch size is not the descriptors' n", [&] { f.permute(&f.outDesc, &f.outDesc, other); },
               RPP_ERROR_INVALID_ARGUMENTS);
        expect(report, "a null source descriptor", [&] { f.permute(nullptr, &f.outDesc); }, RPP_ERROR_INVALID_ARGUMENTS);
        expect(report, "a null permutation array",
               [&] { fastrpp::rppt_channel_permute(f.mid.data(), &f.outDesc, f.mid.data(), &f.outDesc, nullptr, f.handle, RppBackend::RPP_HIP_BACKEND); },
               RPP_ERROR_INVALID_ARGUMENTS);
        {
            RpptDesc d(N, 3, OUT_H, OUT_W + 8, RpptDataType::F32, RpptLayout::NCHW);
            expect(report, "source and destination of different sizes", [&] { f.permute(&f.outDesc, &d); }, RPP_ERROR_INVALID_DIM_LENGTHS);
        }
        {
            RpptDesc s(N, 3, OUT_H, OUT_W, RpptDataType::U8, RpptLayout::NCHW);
            expect(report, "source U8", [&] { f.permute(&s, &f.outDesc); }, RPP_ERROR_INVALID_SRC_DATATYPE);
        }
        {
            RpptDesc d(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NHWC);
            expect(report, "destination NHWC", [&] { f.permute(&f.outDesc, &d); }, RPP_ERROR_INVALID_DST_LAYOUT);
        }
        {
            RpptDesc d = f.outDesc;
            d.strides.nStride += 16;
            expect(report, "a destination with padding between images: accepted, the compute IOp accesses no memory",
                   [&] { f.permute(&f.outDesc, &d); }, RPP_SUCCESS);
        }
        f.permutation[3] = 0;
        f.permutation[4] = 0;
        f.permutation[5] = 1;
        expect(report, "triple (0, 0, 1)", [&] { f.permute(&f.outDesc, &f.outDesc); }, RPP_ERROR_INVALID_ARGUMENTS);
        f.permutation[3] = 3;
        f.permutation[4] = 1;
        f.permutation[5] = 0;
        expect(report, "triple (3, 1, 0)", [&] { f.permute(&f.outDesc, &f.outDesc); }, RPP_ERROR_INVALID_ARGUMENTS);
        f.permutation[3] = 0;
        f.permutation[4] = 1;
        f.permutation[5] = 2;
        expect(report, "permutations that differ between images ((2, 1, 0), then (0, 1, 2))", [&] { f.permute(&f.outDesc, &f.outDesc); },
               RPP_ERROR_NOT_IMPLEMENTED);
        f.permutation[3] = 2;
        f.permutation[4] = 1;
        f.permutation[5] = 0;

        std::cout << "rppt_copy" << std::endl;
        expect(report, "valid arguments", [&] { f.copy(&f.outDesc, &f.outDesc, f.out.data()); }, RPP_SUCCESS);
        expect(report, "host back-end", [&] { f.copy(&f.outDesc, &f.outDesc, f.out.data(), nullptr, RppBackend::RPP_HOST_BACKEND); },
               RPP_ERROR_NOT_IMPLEMENTED);
        expect(report, "a null destination", [&] { f.copy(&f.outDesc, &f.outDesc, nullptr); }, RPP_ERROR_INVALID_ARGUMENTS);
        expect(report, "a null destination descriptor", [&] { f.copy(&f.outDesc, nullptr, f.out.data()); }, RPP_ERROR_INVALID_ARGUMENTS);
        expect(report, "a handle whose batch size is not the descriptors' n", [&] { f.copy(&f.outDesc, &f.outDesc, f.out.data(), other); },
               RPP_ERROR_INVALID_ARGUMENTS);
        {
            RpptDesc d(N, 3, OUT_H + 1, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
            expect(report, "source and destination of different sizes", [&] { f.copy(&f.outDesc, &d, f.out.data()); },
                   RPP_ERROR_INVALID_DIM_LENGTHS);
        }
        {
            RpptDesc s(N, 3, OUT_H, OUT_W, RpptDataType::F16, RpptLayout::NCHW);
            expect(report, "source F16", [&] { f.copy(&s, &f.outDesc, f.out.data()); }, RPP_ERROR_INVALID_SRC_DATATYPE);
        }
        {
            RpptDesc d = f.outDesc;
            d.strides.hStride = OUT_W + 8;
            d.strides.cStride = (OUT_W + 8) * OUT_H;
            d.strides.nStride = 3 * (OUT_W + 8) * OUT_H;
            expect(report, "destination with padded rows", [&] { f.copy(&f.outDesc, &d, f.out.data()); }, RPP_ERROR_INVALID_DST_LAYOUT);
        }
        {
            RpptDesc d = f.outDesc;
            d.strides.nStride = 0;
            expect(report, "destination nStride 0", [&] { f.copy(&f.outDesc, &d, f.out.data()); }, RPP_ERROR_INVALID_DST_LAYOUT);
        }
        {
            RpptDesc d = f.outDesc;
            d.strides.nStride += 16;
            expect(report, "destination with padding between images", [&] { f.copy(&f.outDesc, &d, f.out.data()); }, RPP_ERROR_INVALID_DST_LAYOUT);
        }
        {
            RpptDesc big(N, 3, 16384, 16384, RpptDataType::F32, RpptLayout::NCHW); // 12 GiB: past FKL's 32-bit tensor offsets
            expect(report, "a destination larger than 4 GiB", [&] { f.copy(&big, &big, f.out.data()); }, RPP_ERROR_NOT_IMPLEMENTED);
        }

        std::cout << "executeOperations" << std::endl;
        const auto rmn = f.rmn(&f.srcDesc, &f.outDesc);
        const auto cp = f.permute(&f.outDesc, &f.outDesc);
        const auto copy = f.copy(&f.outDesc, &f.outDesc, f.out.data());
        expect(report, "a null handle", [&] { fastrpp::executeOperations(nullptr, rmn, cp, copy); }, RPP_ERROR_INVALID_ARGUMENTS);
        FASTRPP_TEST_HIP(hipDeviceSynchronize());
        report.check("nothing was launched", holdsSentinel(f.mid.download(), SENTINEL) && holdsSentinel(f.out.download(), SENTINEL));
        expect(report, "a valid chain", [&] { fastrpp::executeOperations(f.handle, rmn, cp, copy); }, RPP_SUCCESS);
        FASTRPP_TEST_HIP(hipDeviceSynchronize());
        report.check("the valid chain wrote the destination of rppt_copy and not the intermediate tensor",
                     holdsSentinel(f.mid.download(), SENTINEL) && !holdsSentinel(f.out.download(), SENTINEL));
        FASTRPP_TEST_RPP(fastrpp::rppDestroy(other, RppBackend::RPP_HIP_BACKEND));
        return report.exitCode();
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
