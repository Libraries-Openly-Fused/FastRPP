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

// The documented errors, the handle and the back-ends: every check of the functions that build IOps throws
// fastrpp::Error with RPP's status. executeOperations takes RPP's own handle and execution back-end and throws on a null
// handle; with RPP_HIP_BACKEND it launches on the handle's stream, with RPP_HOST_BACKEND it runs the same chain, built on
// host memory, on FKL's CPU executor and returns when it is done. A valid chain writes only the destination of
// rppt_copy: the intermediate tensor `mid` (the destination of rppt_resize_mirror_normalize, the source and destination of
// rppt_channel_permute, the source of rppt_copy) holds a sentinel that must survive.
#include <common/test_common.h>

using namespace fastrpp_test;

namespace {

constexpr size_t N = 4;
constexpr int W = 32, H = 24, OUT_W = 16, OUT_H = 8;
constexpr uint32_t SENTINEL = 0x5A5A5A5Au;

struct Fixture {
    std::vector<unsigned char> srcHost = noiseImage(W, H * static_cast<int>(N), 31u); // N images of W x H
    Device<unsigned char> src{N * W * H * 3};
    Device<float> mid{N * 3 * OUT_W * OUT_H};
    Device<float> out{N * 3 * OUT_W * OUT_H};
    Pinned<Rpp32f> mean{3 * N}, stdDev{3 * N};
    Pinned<Rpp32u> mirror{N}, permutation{3 * N};
    Pinned<RpptImagePatch> sizes{N};
    std::vector<RpptROI> rois = std::vector<RpptROI>(N);
    RpptDesc srcDesc{N, 3, H, W, RpptDataType::U8, RpptLayout::NHWC};
    RpptDesc outDesc{N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW};
    hipStream_t stream{};
    rppHandle_t handle{}; // RPP's own handle, on its own stream

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
        FASTRPP_TEST_HIP(hipStreamCreate(&stream));
        FASTRPP_TEST_RPP(rppCreate(&handle, N, 0, stream, RppBackend::RPP_HIP_BACKEND));
        src.upload(srcHost);
        mid.fillBytes(0x5A);
        out.fillBytes(0x5A);
    }
    ~Fixture() {
        static_cast<void>(rppDestroy(handle, RppBackend::RPP_HIP_BACKEND));
        static_cast<void>(hipStreamDestroy(stream));
    }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;

    auto rmn(RpptDesc* s, RpptDesc* d, const RpptInterpolationType interpolation = RpptInterpolationType::BILINEAR,
             const RpptRoiType roiType = RpptRoiType::XYWH) {
        return fastrpp::rppt_resize_mirror_normalize<N>(src.data(), s, mid.data(), d, sizes.data(), interpolation, mean.data(), stdDev.data(),
                                                        mirror.data(), rois.data(), roiType);
    }
    auto permute(RpptDesc* s, RpptDesc* d) { return fastrpp::rppt_channel_permute(mid.data(), s, mid.data(), d, permutation.data()); }
    auto copy(RpptDesc* s, RpptDesc* d, void* const dstPtr) { return fastrpp::rppt_copy(mid.data(), s, dstPtr, d); }
};

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
        Fixture f;

        std::cout << "rppt_resize_mirror_normalize" << std::endl;
        expect(report, "valid arguments", [&] { f.rmn(&f.srcDesc, &f.outDesc); }, RPP_SUCCESS);
        expect(report, "NEAREST_NEIGHBOR", [&] { f.rmn(&f.srcDesc, &f.outDesc, RpptInterpolationType::NEAREST_NEIGHBOR); },
               RPP_ERROR_NOT_IMPLEMENTED);
        expect(report, "a null destination descriptor", [&] { f.rmn(&f.srcDesc, nullptr); }, RPP_ERROR_INVALID_ARGUMENTS);
        expect(report, "a null source",
               [&] {
                   fastrpp::rppt_resize_mirror_normalize<N>(nullptr, &f.srcDesc, f.mid.data(), &f.outDesc, f.sizes.data(),
                                                            RpptInterpolationType::BILINEAR, f.mean.data(), f.stdDev.data(), f.mirror.data(),
                                                            f.rois.data(), RpptRoiType::XYWH);
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
            RpptDesc d(N + 1, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
            expect(report, "destination n != BATCH", [&] { f.rmn(&f.srcDesc, &d); }, RPP_ERROR_INVALID_ARGUMENTS);
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
        expect(report, "a null source descriptor", [&] { f.permute(nullptr, &f.outDesc); }, RPP_ERROR_INVALID_ARGUMENTS);
        expect(report, "a null permutation array",
               [&] { fastrpp::rppt_channel_permute(f.mid.data(), &f.outDesc, f.mid.data(), &f.outDesc, nullptr); },
               RPP_ERROR_INVALID_ARGUMENTS);
        {
            RpptDesc d(N - 1, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
            expect(report, "destination n != source n", [&] { f.permute(&f.outDesc, &d); }, RPP_ERROR_INVALID_ARGUMENTS);
        }
        {
            RpptDesc s(0, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
            RpptDesc d(0, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
            expect(report, "n = 0", [&] { f.permute(&s, &d); }, RPP_ERROR_INVALID_DIM_LENGTHS);
        }
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
        expect(report, "a null destination", [&] { f.copy(&f.outDesc, &f.outDesc, nullptr); }, RPP_ERROR_INVALID_ARGUMENTS);
        expect(report, "a null destination descriptor", [&] { f.copy(&f.outDesc, nullptr, f.out.data()); }, RPP_ERROR_INVALID_ARGUMENTS);
        {
            RpptDesc d(N + 1, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
            expect(report, "destination n != source n", [&] { f.copy(&f.outDesc, &d, f.out.data()); }, RPP_ERROR_INVALID_ARGUMENTS);
        }
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

        std::cout << "executeOperations, RPP_HIP_BACKEND" << std::endl;
        constexpr RppBackend hipBackend = RppBackend::RPP_HIP_BACKEND;
        constexpr RppBackend hostBackend = RppBackend::RPP_HOST_BACKEND;
        const auto rmn = f.rmn(&f.srcDesc, &f.outDesc);
        const auto cp = f.permute(&f.outDesc, &f.outDesc);
        const auto copy = f.copy(&f.outDesc, &f.outDesc, f.out.data());
        expect(report, "a null handle", [&] { fastrpp::executeOperations(nullptr, hipBackend, rmn, cp, copy); }, RPP_ERROR_INVALID_ARGUMENTS);
        FASTRPP_TEST_HIP(hipDeviceSynchronize());
        report.check("nothing was launched", holdsSentinel(f.mid.download(), SENTINEL) && holdsSentinel(f.out.download(), SENTINEL));
        expect(report, "a valid chain", [&] { fastrpp::executeOperations(f.handle, hipBackend, rmn, cp, copy); }, RPP_SUCCESS);
        FASTRPP_TEST_HIP(hipDeviceSynchronize());
        const std::vector<float> gpuOut = f.out.download();
        report.check("the valid chain wrote the destination of rppt_copy and not the intermediate tensor",
                     holdsSentinel(f.mid.download(), SENTINEL) && !holdsSentinel(gpuOut, SENTINEL));

        // The kernel goes to the stream of the handle (rppGetStream): while that stream is captured into a graph, the
        // launch is captured and does not run; it runs when the graph is launched
        f.out.fillBytes(0x5A);
        FASTRPP_TEST_HIP(hipDeviceSynchronize());
        hipGraph_t graph{};
        FASTRPP_TEST_HIP(hipStreamBeginCapture(f.stream, hipStreamCaptureModeRelaxed));
        std::string captureMessage;
        const RppStatus captured = statusOf([&] { fastrpp::executeOperations(f.handle, hipBackend, rmn, cp, copy); }, captureMessage);
        FASTRPP_TEST_HIP(hipStreamEndCapture(f.stream, &graph));
        size_t nodes = 0;
        FASTRPP_TEST_HIP(hipGraphGetNodes(graph, nullptr, &nodes));
        report.check("under a capture of the handle's stream: one kernel captured (" + std::to_string(nodes) + " nodes" +
                         (captured == RPP_SUCCESS ? std::string() : ", " + captureMessage) + "), nothing run",
                     captured == RPP_SUCCESS && nodes == 1 && holdsSentinel(f.out.download(), SENTINEL));
        hipGraphExec_t exec{};
        FASTRPP_TEST_HIP(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        FASTRPP_TEST_HIP(hipGraphLaunch(exec, f.stream));
        FASTRPP_TEST_HIP(hipStreamSynchronize(f.stream));
        report.check("the captured kernel wrote the destination of rppt_copy, and only it, when the graph ran",
                     holdsSentinel(f.mid.download(), SENTINEL) && !holdsSentinel(f.out.download(), SENTINEL));
        FASTRPP_TEST_HIP(hipGraphExecDestroy(exec));
        FASTRPP_TEST_HIP(hipGraphDestroy(graph));

        // A handle made with stream nullptr (rppCreate's default) runs on the null stream
        rppHandle_t nullStreamHandle{};
        FASTRPP_TEST_RPP(rppCreate(&nullStreamHandle, N, 0, nullptr, hipBackend));
        f.out.fillBytes(0x5A);
        FASTRPP_TEST_HIP(hipDeviceSynchronize());
        expect(report, "a handle on the null stream", [&] { fastrpp::executeOperations(nullStreamHandle, hipBackend, rmn, cp, copy); }, RPP_SUCCESS);
        FASTRPP_TEST_HIP(hipDeviceSynchronize());
        report.check("the chain on the null stream wrote the destination of rppt_copy", !holdsSentinel(f.out.download(), SENTINEL));
        FASTRPP_TEST_RPP(rppDestroy(nullStreamHandle, hipBackend));

        // RPP_HOST_BACKEND: the same chain built on pageable host memory, on FKL's CPU executor (CPU_MULTITHREAD when
        // FastRPP's CMake enables it), with an RPP host handle. It returns when the work is done: no synchronisation
        std::cout << "executeOperations, RPP_HOST_BACKEND (FKL ParArch::" << fk::toStrView(fastrpp::detail::HOST_PAR_ARCH) << ")" << std::endl;
        Buffer<unsigned char> hostSrc(f.src.size(), hostBackend);
        Buffer<float> hostMid(f.mid.size(), hostBackend), hostOut(f.out.size(), hostBackend);
        hostSrc.upload(f.srcHost);
        hostMid.fillBytes(0x5A);
        hostOut.fillBytes(0x5A);
        const auto hostRmn = fastrpp::rppt_resize_mirror_normalize<N>(hostSrc.data(), &f.srcDesc, hostMid.data(), &f.outDesc, f.sizes.data(),
                                                                      RpptInterpolationType::BILINEAR, f.mean.data(), f.stdDev.data(),
                                                                      f.mirror.data(), f.rois.data(), RpptRoiType::XYWH);
        const auto hostCp = fastrpp::rppt_channel_permute(hostMid.data(), &f.outDesc, hostMid.data(), &f.outDesc, f.permutation.data());
        const auto hostCopy = fastrpp::rppt_copy(hostMid.data(), &f.outDesc, hostOut.data(), &f.outDesc);
        rppHandle_t hostHandle{};
        FASTRPP_TEST_RPP(rppCreate(&hostHandle, N, 0, nullptr, hostBackend));
        expect(report, "a null handle", [&] { fastrpp::executeOperations(nullptr, hostBackend, hostRmn, hostCp, hostCopy); },
               RPP_ERROR_INVALID_ARGUMENTS);
        report.check("nothing was run", holdsSentinel(hostOut.download(), SENTINEL));
        expect(report, "a valid chain", [&] { fastrpp::executeOperations(hostHandle, hostBackend, hostRmn, hostCp, hostCopy); }, RPP_SUCCESS);
        const std::vector<float> cpuOut = hostOut.download();
        report.check("the valid chain wrote the destination of rppt_copy and not the intermediate tensor, before returning",
                     holdsSentinel(hostMid.download(), SENTINEL) && !holdsSentinel(cpuOut, SENTINEL));
        report.row("the host back-end's output vs the HIP back-end's, every value",
                   compare(cpuOut.data(), gpuOut.data(), N, OUT_W, OUT_H, [](const size_t, const int, const int) { return true; }), true);
        FASTRPP_TEST_RPP(rppDestroy(hostHandle, hostBackend));
        return report.exitCode();
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
