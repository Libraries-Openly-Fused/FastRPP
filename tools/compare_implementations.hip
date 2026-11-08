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

// Which implementation of the per-image mirror and permutation FastRPP uses by default (README.md, "Implementations"):
// fastrpp::UniformBranches and fastrpp::Selects, and a replica of fkl_fused_halfpixel (fkl-benchmarks 082ce201,
// rpp/benchmark_rpp_preprocessing.h), on pipeline (a): N crops of 60x120 at (i, i) of a 3840x2160 frame of the
// benchmark's seeded noise, resized to 64x128, NCHW float. Every variant reads the same fk::Ptr2D frame (FastRPP
// through sharedSourceDesc and its pitch). Two parameter sets:
//   benchmark  the benchmark's: RGB to BGR, * 0.3, - {1, 4, 3.2}, / {3.2, 0.6, 11.8} (FastRPP: mean = sub / 0.3,
//              stdDev = div / 0.3, permutation (2, 1, 0), mirror 0); the replica, UniformBranches and Selects
//   mixed      per image a mirror flag of 0, 1 or 2, one of the six permutations, its own mean and stdDev: what differs
//              between the two implementations; UniformBranches and Selects (the replica has no per-image parameters)
// Measures, for each N:
//   agreement       max |diff| against the replica (benchmark set, every pixel) and against RPP's two calls on the crops
//                   copied into a dense batch (outside RPP's clamp region, required <= 1e-4; inside it, informational);
//                   every output starts as NaN, so an element not written fails
//   GPU time        events around the fused kernel, the operations built beforehand: "single", one launch on an idle
//                   stream (it includes the dispatch of the launch); "back-to-back", 20 launches between the events,
//                   divided by 20 (the dispatch of a launch overlaps the previous kernel when the kernel is longer)
//   public path     host wall clock of the API: build (FastRPP: the two rppt_* calls; the replica: its crop views and
//                   operations), run (executeOperations + stream synchronisation; FastRPP composes its per-image chains
//                   inside executeOperations, so its run holds that host work) and total = build + run, per run
// Every timed loop: a warm-up of at least 0.3 s, 3 rounds in a rotated order, the median of the round medians and their
// range. Not the benchmark protocol (fkl-benchmarks measures FastRPP against RPP); a tool to choose the implementation.
// Writes compare_implementations.csv in the working directory.
#include <fast_rpp.h>
#include <rpp/rpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using DPP = fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>;

constexpr int SRC_W = 3840, SRC_H = 2160, CROP_W = 60, CROP_H = 120, OUT_W = 64, OUT_H = 128;
constexpr float ALPHA = 0.3f;
constexpr float SUB[3] = {1.f, 4.f, 3.2f}, DIV[3] = {3.2f, 0.6f, 11.8f};
constexpr int ROUNDS = 3, SINGLE_SAMPLES = 200, B2B_SAMPLES = 50, B2B_LAUNCHES = 20;
constexpr double WARMUP_MS = 300.0;

void hipCheck(const hipError_t e, const char* what) {
    if (e != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
    }
}

double msBetween(const Clock::time_point a, const Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? std::nan("") : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

struct Stat {
    std::vector<double> rounds;
    double med() const { return median(rounds); }
    double lo() const { return rounds.empty() ? std::nan("") : *std::min_element(rounds.begin(), rounds.end()); }
    double hi() const { return rounds.empty() ? std::nan("") : *std::max_element(rounds.begin(), rounds.end()); }
};

struct Variant {
    std::string params, name;
    size_t argBytes{0};
    std::function<void()> launch;                              // the fused kernel alone, operations built beforehand
    std::function<void(double&, double&)> publicPath;          // build, run (+ sync): host wall clock
    Stat single, backToBack, build, run, total;
    double vsReplica{std::nan("")}, vsRppOutside{0}, vsRppInside{0};
};

// The parameter arrays of one FastRPP variant, as RPP's API takes them
template <size_t N>
struct Params {
    fastrpp::rppHandle_t handle{};
    RpptDesc frameDesc, outDesc;
    Rpp32f mean[3 * N], stdDev[3 * N];
    Rpp32u mirror[N], permutation[3 * N];
    RpptImagePatch sizes[N];
    RpptROI rois[N];
};

template <size_t N>
void fillParams(Params<N>& p, const bool mixed, const fk::RawPtr<fk::ND::_2D, fk::uchar3>& frameView, const hipStream_t stream) {
    const Rpp32u perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    p.frameDesc = fastrpp::sharedSourceDesc(N, 3, SRC_H, SRC_W, RpptDataType::U8, RpptLayout::NHWC, frameView.dims.pitch);
    p.outDesc = RpptDesc(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
    for (size_t i = 0; i < N; ++i) {
        const Rpp32u* const perm = mixed ? perms[(i * 5) % 6] : perms[5];
        for (size_t c = 0; c < 3; ++c) {
            p.mean[3 * i + c] = mixed ? static_cast<float>((i * 11 + c * 5) % 40) : SUB[2 - c] / ALPHA;
            p.stdDev[3 * i + c] = mixed ? 1.0f + 0.25f * static_cast<float>((i + c) % 7) : DIV[2 - c] / ALPHA;
            p.permutation[3 * i + c] = perm[c];
        }
        p.mirror[i] = mixed ? static_cast<Rpp32u>(i % 3) : 0u;
        p.sizes[i] = RpptImagePatch{OUT_W, OUT_H};
        p.rois[i].xywhROI = {{static_cast<int>(i), static_cast<int>(i)}, CROP_W, CROP_H};
    }
    if (fastrpp::rppCreate(&p.handle, N, 0, stream, RppBackend::RPP_HIP_BACKEND) != rppStatusSuccess) {
        throw std::runtime_error("fastrpp::rppCreate failed");
    }
}

// RPP's two calls on the crops copied into a dense batch, with these parameters (RPP reads them on the GPU: pinned
// copies; its ROI array holds 256 * ceil(N / 256) ROIs, as its XYWH -> LTRB conversion writes that many)
template <size_t N>
std::vector<float> runRpp(const Params<N>& p, const fk::RawPtr<fk::ND::_2D, fk::uchar3>& frameView, const hipStream_t stream) {
    const size_t cropBytes = static_cast<size_t>(CROP_W) * CROP_H * 3, outCount = N * 3 * static_cast<size_t>(OUT_W) * OUT_H;
    const size_t roiCapacity = 256 * ((N + 255) / 256);
    unsigned char* crops = nullptr;
    float *mid = nullptr, *out = nullptr;
    Rpp32f *mean = nullptr, *stdDev = nullptr;
    Rpp32u *mirror = nullptr, *permutation = nullptr;
    RpptImagePatch* sizes = nullptr;
    RpptROI* rois = nullptr;
    hipCheck(hipMalloc(reinterpret_cast<void**>(&crops), cropBytes * N), "hipMalloc");
    hipCheck(hipMalloc(reinterpret_cast<void**>(&mid), sizeof(float) * outCount), "hipMalloc");
    hipCheck(hipMalloc(reinterpret_cast<void**>(&out), sizeof(float) * outCount), "hipMalloc");
    hipCheck(hipMemset(out, 0xFF, sizeof(float) * outCount), "hipMemset");
    hipCheck(hipHostMalloc(reinterpret_cast<void**>(&mean), sizeof(Rpp32f) * 3 * N), "hipHostMalloc");
    hipCheck(hipHostMalloc(reinterpret_cast<void**>(&stdDev), sizeof(Rpp32f) * 3 * N), "hipHostMalloc");
    hipCheck(hipHostMalloc(reinterpret_cast<void**>(&mirror), sizeof(Rpp32u) * N), "hipHostMalloc");
    hipCheck(hipHostMalloc(reinterpret_cast<void**>(&permutation), sizeof(Rpp32u) * 3 * N), "hipHostMalloc");
    hipCheck(hipHostMalloc(reinterpret_cast<void**>(&sizes), sizeof(RpptImagePatch) * N), "hipHostMalloc");
    hipCheck(hipHostMalloc(reinterpret_cast<void**>(&rois), sizeof(RpptROI) * roiCapacity), "hipHostMalloc");
    for (size_t i = 0; i < N; ++i) {
        std::memcpy(mean + 3 * i, p.mean + 3 * i, sizeof(Rpp32f) * 3);
        std::memcpy(stdDev + 3 * i, p.stdDev + 3 * i, sizeof(Rpp32f) * 3);
        std::memcpy(permutation + 3 * i, p.permutation + 3 * i, sizeof(Rpp32u) * 3);
        mirror[i] = p.mirror[i];
        sizes[i] = p.sizes[i];
        rois[i].xywhROI = {{0, 0}, CROP_W, CROP_H};
        hipCheck(hipMemcpy2D(crops + cropBytes * i, static_cast<size_t>(CROP_W) * 3,
                             reinterpret_cast<unsigned char*>(frameView.data) + i * frameView.dims.pitch + i * 3, frameView.dims.pitch,
                             static_cast<size_t>(CROP_W) * 3, CROP_H, hipMemcpyDeviceToDevice),
                 "hipMemcpy2D");
    }
    RpptDesc cropDesc(N, 3, CROP_H, CROP_W, RpptDataType::U8, RpptLayout::NHWC);
    RpptDesc outDesc(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
    rppHandle_t handle{};
    if (rppCreate(&handle, N, 0, stream, RppBackend::RPP_HIP_BACKEND) != rppStatusSuccess ||
        rppt_resize_mirror_normalize(crops, &cropDesc, mid, &outDesc, sizes, RpptInterpolationType::BILINEAR, mean, stdDev, mirror, rois,
                                     RpptRoiType::XYWH, handle, RppBackend::RPP_HIP_BACKEND) != RPP_SUCCESS ||
        rppt_channel_permute(mid, &outDesc, out, &outDesc, permutation, handle, RppBackend::RPP_HIP_BACKEND) != RPP_SUCCESS) {
        throw std::runtime_error("RPP failed");
    }
    hipCheck(hipStreamSynchronize(stream), "hipStreamSynchronize");
    std::vector<float> host(outCount);
    hipCheck(hipMemcpy(host.data(), out, sizeof(float) * outCount, hipMemcpyDeviceToHost), "hipMemcpy");
    static_cast<void>(rppDestroy(handle, RppBackend::RPP_HIP_BACKEND));
    hipCheck(hipFree(crops), "hipFree");
    hipCheck(hipFree(mid), "hipFree");
    hipCheck(hipFree(out), "hipFree");
    for (void* q : {static_cast<void*>(mean), static_cast<void*>(stdDev), static_cast<void*>(mirror), static_cast<void*>(permutation),
                    static_cast<void*>(sizes), static_cast<void*>(rois)}) {
        hipCheck(hipHostFree(q), "hipHostFree");
    }
    return host;
}

// RPP's clamp region for a 60x120 ROI resized to 64x128: the destination columns and rows whose source position,
// computed as RPP does, has floor >= size - 2; RPP computes a mirrored image's column x at column OUT_W - 1 - x
bool inRppClampRegion(const bool mirrored, const int x, const int y) {
    const float rx = static_cast<float>(CROP_W) / OUT_W, ry = static_cast<float>(CROP_H) / OUT_H;
    const int xs = mirrored ? OUT_W - 1 - x : x;
    return std::floor(static_cast<float>(xs) * rx + (rx - 1.0f) * 0.5f) >= static_cast<float>(CROP_W - 2) ||
           std::floor(static_cast<float>(y) * ry + (ry - 1.0f) * 0.5f) >= static_cast<float>(CROP_H - 2);
}

// max |a - b| over N NCHW images, outside or inside RPP's clamp region (or everywhere); NaN propagates
template <size_t N>
double maxDiff(const float* a, const float* b, const Rpp32u* mirror, const int region /* -1 everywhere, 0 outside, 1 inside */) {
    double m = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < 3; ++k) {
            for (int y = 0; y < OUT_H; ++y) {
                for (int x = 0; x < OUT_W; ++x) {
                    if (region >= 0 && inRppClampRegion(mirror[i] == 1u, x, y) != (region == 1)) {
                        continue;
                    }
                    const size_t idx = ((i * 3 + k) * OUT_H + static_cast<size_t>(y)) * OUT_W + static_cast<size_t>(x);
                    const double d = std::fabs(static_cast<double>(a[idx]) - static_cast<double>(b[idx]));
                    m = (std::isnan(d) || std::isnan(m)) ? std::nan("") : std::max(m, d);
                }
            }
        }
    }
    return m;
}

template <size_t N>
void runN(std::ofstream& csv, fk::Stream_<fk::ParArch::GPU_AMD>& stream, fk::Ptr2D<fk::uchar3>& frame, bool& allAgree) {
    std::cout << "N = " << N << std::endl;
    const fk::RawPtr<fk::ND::_2D, fk::uchar3> frameView = frame.ptr();
    const hipStream_t hipStream = stream.getHIPStream();
    const size_t outCount = N * 3 * static_cast<size_t>(OUT_W) * OUT_H;
    fk::Tensor<float> replicaOut(OUT_W, OUT_H, N, 3), mid(OUT_W, OUT_H, N, 3);
    fk::Tensor<float> outs[4] = {fk::Tensor<float>(OUT_W, OUT_H, N, 3), fk::Tensor<float>(OUT_W, OUT_H, N, 3),
                                 fk::Tensor<float>(OUT_W, OUT_H, N, 3), fk::Tensor<float>(OUT_W, OUT_H, N, 3)};

    // ---- The replica of fkl_fused_halfpixel
    auto buildReplica = [&]() {
        std::array<fk::Ptr2D<fk::uchar3>, N> crops;
        for (size_t i = 0; i < N; ++i) {
            crops[i] = frame.crop2D(fk::Rect(static_cast<uint>(i), static_cast<uint>(i), CROP_W, CROP_H));
        }
        const auto read = fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(crops).then(
            fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OUT_W, OUT_H)));
        return std::make_tuple(read, fk::VectorReorder<fk::float3, 2, 1, 0>::build(), fk::Mul<fk::float3>::build(fk::float3{ALPHA, ALPHA, ALPHA}),
                               fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]}),
                               fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]}),
                               fk::TensorSplit<fk::float3>::build(replicaOut.ptr()));
    };
    const auto replica = buildReplica();
    const size_t replicaArgBytes = std::apply([](const auto&... ops) { return (sizeof(ops) + ...); }, replica);

    // ---- FastRPP: [benchmark, mixed] x [UniformBranches, Selects]
    auto benchmarkParams = std::make_unique<Params<N>>();
    auto mixedParams = std::make_unique<Params<N>>();
    fillParams(*benchmarkParams, false, frameView, hipStream);
    fillParams(*mixedParams, true, frameView, hipStream);
    auto build = [&](Params<N>& p, auto impl, fk::Tensor<float>& out) {
        using Impl = decltype(impl);
        const auto rmn = fastrpp::rppt_resize_mirror_normalize<N, RpptDataType::U8, RpptLayout::NHWC, RpptDataType::F32, RpptLayout::NCHW, 3, Impl>(
            frameView.data, &p.frameDesc, mid.ptr().data, &p.outDesc, p.sizes, RpptInterpolationType::BILINEAR, p.mean, p.stdDev, p.mirror,
            p.rois, RpptRoiType::XYWH, p.handle, RppBackend::RPP_HIP_BACKEND);
        const auto cp = fastrpp::rppt_channel_permute<N, RpptDataType::F32, RpptLayout::NCHW, RpptLayout::NCHW, 3, Impl>(
            mid.ptr().data, &p.outDesc, out.ptr().data, &p.outDesc, p.permutation, p.handle, RppBackend::RPP_HIP_BACKEND);
        if (rmn.status != RPP_SUCCESS || cp.status != RPP_SUCCESS) {
            throw std::runtime_error("a FastRPP operation failed its checks");
        }
        return std::make_pair(rmn, cp);
    };
    const auto bBranches = build(*benchmarkParams, fastrpp::UniformBranches{}, outs[0]);
    const auto bSelects = build(*benchmarkParams, fastrpp::Selects{}, outs[1]);
    const auto mBranches = build(*mixedParams, fastrpp::UniformBranches{}, outs[2]);
    const auto mSelects = build(*mixedParams, fastrpp::Selects{}, outs[3]);
    const auto fbBranches = fastrpp::detail::fuse(bBranches.first, bBranches.second);
    const auto fbSelects = fastrpp::detail::fuse(bSelects.first, bSelects.second);
    const auto fmBranches = fastrpp::detail::fuse(mBranches.first, mBranches.second);
    const auto fmSelects = fastrpp::detail::fuse(mSelects.first, mSelects.second);

    auto publicFastRPP = [&](Params<N>& p, auto impl, fk::Tensor<float>& out) {
        return [&p, impl, &out, &build, &stream](double& b, double& r) {
            const auto t0 = Clock::now();
            const auto ops = build(p, impl, out);
            const auto t1 = Clock::now();
            if (fastrpp::executeOperations(p.handle, ops.first, ops.second) != RPP_SUCCESS) {
                throw std::runtime_error("executeOperations failed");
            }
            stream.sync();
            const auto t2 = Clock::now();
            b = msBetween(t0, t1);
            r = msBetween(t1, t2);
        };
    };
    std::vector<Variant> variants;
    variants.push_back({"benchmark", "fkl_fused_halfpixel replica", replicaArgBytes,
                        [&] { std::apply([&](const auto&... ops) { fk::executeOperations<DPP>(stream, ops...); }, replica); },
                        [&](double& b, double& r) {
                            const auto t0 = Clock::now();
                            const auto ops = buildReplica();
                            const auto t1 = Clock::now();
                            std::apply([&](const auto&... o) { fk::executeOperations<DPP>(stream, o...); }, ops);
                            stream.sync();
                            const auto t2 = Clock::now();
                            b = msBetween(t0, t1);
                            r = msBetween(t1, t2);
                        },
                        {}, {}, {}, {}, {}});
    variants.push_back({"benchmark", "FastRPP UniformBranches", sizeof(fbBranches.first) + sizeof(fbBranches.second),
                        [&] { fk::executeOperations<DPP>(stream, fbBranches.first, fbBranches.second); },
                        publicFastRPP(*benchmarkParams, fastrpp::UniformBranches{}, outs[0]), {}, {}, {}, {}, {}});
    variants.push_back({"benchmark", "FastRPP Selects", sizeof(fbSelects.first) + sizeof(fbSelects.second),
                        [&] { fk::executeOperations<DPP>(stream, fbSelects.first, fbSelects.second); },
                        publicFastRPP(*benchmarkParams, fastrpp::Selects{}, outs[1]), {}, {}, {}, {}, {}});
    variants.push_back({"mixed", "FastRPP UniformBranches", sizeof(fmBranches.first) + sizeof(fmBranches.second),
                        [&] { fk::executeOperations<DPP>(stream, fmBranches.first, fmBranches.second); },
                        publicFastRPP(*mixedParams, fastrpp::UniformBranches{}, outs[2]), {}, {}, {}, {}, {}});
    variants.push_back({"mixed", "FastRPP Selects", sizeof(fmSelects.first) + sizeof(fmSelects.second),
                        [&] { fk::executeOperations<DPP>(stream, fmSelects.first, fmSelects.second); },
                        publicFastRPP(*mixedParams, fastrpp::Selects{}, outs[3]), {}, {}, {}, {}, {}});

    // ---- Agreement: every output starts as NaN
    hipCheck(hipMemset(replicaOut.ptr().data, 0xFF, replicaOut.sizeInBytes()), "hipMemset");
    for (fk::Tensor<float>& t : outs) {
        hipCheck(hipMemset(t.ptr().data, 0xFF, t.sizeInBytes()), "hipMemset");
    }
    for (Variant& v : variants) {
        v.launch();
    }
    replicaOut.download(stream);
    for (fk::Tensor<float>& t : outs) {
        t.download(stream);
    }
    stream.sync();
    const std::vector<float> rppBenchmark = runRpp(*benchmarkParams, frameView, hipStream);
    const std::vector<float> rppMixed = runRpp(*mixedParams, frameView, hipStream);
    const float* replicaHost = replicaOut.ptrPinned().data;
    variants[0].vsRppOutside = maxDiff<N>(replicaHost, rppBenchmark.data(), benchmarkParams->mirror, 0);
    variants[0].vsRppInside = maxDiff<N>(replicaHost, rppBenchmark.data(), benchmarkParams->mirror, 1);
    variants[0].vsReplica = 0;
    for (size_t k = 0; k < 4; ++k) {
        Variant& v = variants[k + 1];
        const float* host = outs[k].ptrPinned().data;
        const bool mixed = k >= 2;
        const std::vector<float>& rpp = mixed ? rppMixed : rppBenchmark;
        const Rpp32u* mirror = mixed ? mixedParams->mirror : benchmarkParams->mirror;
        v.vsRppOutside = maxDiff<N>(host, rpp.data(), mirror, 0);
        v.vsRppInside = maxDiff<N>(host, rpp.data(), mirror, 1);
        if (!mixed) {
            v.vsReplica = maxDiff<N>(host, replicaHost, mirror, -1);
            allAgree &= v.vsReplica <= 1e-4;
        }
        allAgree &= v.vsRppOutside <= 1e-4;
    }
    static_cast<void>(outCount);

    // ---- GPU time and the public path: 3 rounds, rotated order
    hipEvent_t e0{}, e1{};
    hipCheck(hipEventCreate(&e0), "hipEventCreate");
    hipCheck(hipEventCreate(&e1), "hipEventCreate");
    auto eventMs = [&](const std::function<void()>& launch, const int launches) {
        hipCheck(hipEventRecord(e0, hipStream), "hipEventRecord");
        for (int l = 0; l < launches; ++l) {
            launch();
        }
        hipCheck(hipEventRecord(e1, hipStream), "hipEventRecord");
        hipCheck(hipEventSynchronize(e1), "hipEventSynchronize");
        float ms = 0.f;
        hipCheck(hipEventElapsedTime(&ms, e0, e1), "hipEventElapsedTime");
        return static_cast<double>(ms) / launches;
    };
    for (int round = 0; round < ROUNDS; ++round) {
        for (size_t k = 0; k < variants.size(); ++k) {
            Variant& v = variants[(k + static_cast<size_t>(round)) % variants.size()];
            const auto warmStart = Clock::now();
            while (msBetween(warmStart, Clock::now()) < WARMUP_MS) {
                v.launch();
                stream.sync();
            }
            std::vector<double> single, b2b, builds, runs, totals;
            for (int s = 0; s < SINGLE_SAMPLES; ++s) {
                single.push_back(eventMs(v.launch, 1));
            }
            for (int s = 0; s < B2B_SAMPLES; ++s) {
                b2b.push_back(eventMs(v.launch, B2B_LAUNCHES));
            }
            for (int s = 0; s < SINGLE_SAMPLES; ++s) {
                double b = 0, r = 0;
                v.publicPath(b, r);
                builds.push_back(b);
                runs.push_back(r);
                totals.push_back(b + r);
            }
            v.single.rounds.push_back(median(single));
            v.backToBack.rounds.push_back(median(b2b));
            v.build.rounds.push_back(median(builds));
            v.run.rounds.push_back(median(runs));
            v.total.rounds.push_back(median(totals));
        }
    }
    hipCheck(hipEventDestroy(e0), "hipEventDestroy");
    hipCheck(hipEventDestroy(e1), "hipEventDestroy");

    for (const Variant& v : variants) {
        std::cout << "  [" << v.params << "] " << v.name << ": kernel arguments " << v.argBytes << " B; GPU single " << v.single.med() << " ms ("
                  << v.single.lo() << " - " << v.single.hi() << "), back-to-back " << v.backToBack.med() << " ms (" << v.backToBack.lo()
                  << " - " << v.backToBack.hi() << "); build " << v.build.med() << " ms, run + sync " << v.run.med() << " ms, total "
                  << v.total.med() << " ms; max |diff| vs the replica " << v.vsReplica << ", vs RPP outside its clamp region "
                  << v.vsRppOutside << ", inside it " << v.vsRppInside << std::endl;
        csv << N << "," << v.params << "," << v.name << "," << v.argBytes << "," << v.single.med() << "," << v.single.lo() << "," << v.single.hi()
            << "," << v.backToBack.med() << "," << v.backToBack.lo() << "," << v.backToBack.hi() << "," << v.build.med() << "," << v.run.med()
            << "," << v.total.med() << "," << v.vsReplica << "," << v.vsRppOutside << "," << v.vsRppInside << "\n";
    }
    csv.flush();
    static_cast<void>(fastrpp::rppDestroy(benchmarkParams->handle, RppBackend::RPP_HIP_BACKEND));
    static_cast<void>(fastrpp::rppDestroy(mixedParams->handle, RppBackend::RPP_HIP_BACKEND));
}

template <size_t... Ns>
void runAll(std::ofstream& csv, fk::Stream_<fk::ParArch::GPU_AMD>& stream, fk::Ptr2D<fk::uchar3>& frame, bool& allAgree,
            std::index_sequence<Ns...>) {
    (runN<Ns>(csv, stream, frame, allAgree), ...);
}

} // namespace

int main() {
    try {
        fk::Stream_<fk::ParArch::GPU_AMD> stream;
        fk::Ptr2D<fk::uchar3> frame(SRC_W, SRC_H);
        uint32_t state = 12345u; // the benchmark's seeded noise
        for (int y = 0; y < SRC_H; ++y) {
            for (int x = 0; x < SRC_W; ++x) {
                state = state * 1664525u + 1013904223u;
                frame.at(x, y) = fk::uchar3{static_cast<unsigned char>(state >> 24), static_cast<unsigned char>(state >> 16),
                                            static_cast<unsigned char>(state >> 8)};
            }
        }
        frame.upload(stream);
        stream.sync();
        std::ofstream csv("compare_implementations.csv");
        csv << "batch,params,variant,kernel_arg_bytes,gpu_single_ms,gpu_single_round_min,gpu_single_round_max,gpu_b2b_ms,gpu_b2b_round_min,"
               "gpu_b2b_round_max,build_ms,run_sync_ms,total_ms,max_diff_vs_replica,max_diff_vs_rpp_outside_clamp,max_diff_vs_rpp_in_clamp\n";
        bool allAgree = true;
        runAll(csv, stream, frame, allAgree, std::index_sequence<10, 50, 100, 200, 300>{});
        std::cout << (allAgree ? "every required agreement holds" : "AN AGREEMENT FAILED") << std::endl;
        return allAgree ? 0 : 1;
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
