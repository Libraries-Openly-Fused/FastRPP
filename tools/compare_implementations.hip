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
// fastrpp::UniformBranches and fastrpp::Selects against a replica of fkl_fused_halfpixel (fkl-benchmarks 082ce201,
// rpp/benchmark_rpp_preprocessing.h), on pipeline (a) with the benchmark's parameters: N crops of 60x120 at (i, i) of a
// 3840x2160 frame of seeded noise, resized to 64x128, RGB to BGR, * 0.3, - {1, 4, 3.2}, / {3.2, 0.6, 11.8}, NCHW float.
// Every variant reads the same fk::Ptr2D frame (FastRPP through sharedSourceDesc and its pitch).
//   agreement     the two FastRPP implementations against the replica: max |diff|, must be <= 1e-4
//   GPU time      events around the launch of the fused kernel, the operations built beforehand; every variant warmed up
//                 for at least 0.5 s, then 300 samples; 3 rounds in a rotated order; the median of the round medians
//   public path   host wall clock of FastRPP's API: build = the two rppt_* calls; run = executeOperations + stream
//                 synchronisation; the replica: build = its crop views and operations, run = executeOperations + sync;
//                 medians of 300 runs after a warm-up of at least 0.5 s
// Not the benchmark protocol (fkl-benchmarks measures FastRPP against RPP); a tool to choose the implementation. Writes
// compare_implementations.csv in the working directory.
#include <fast_rpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

struct Variant {
    std::string name;
    std::function<void()> launch;     // the fused kernel alone, operations built beforehand
    std::function<void(double&, double&)> publicPath; // build, run (+ sync): host wall clock
    std::vector<double> gpuRoundMedians;
    double buildMedian{0}, runMedian{0};
};

template <size_t N, typename Impl>
struct FastRPPSide {
    fastrpp::rppHandle_t handle{};
    RpptDesc frameDesc, outDesc;
    Rpp32f mean[3 * N], stdDev[3 * N];
    Rpp32u mirror[N], permutation[3 * N];
    RpptImagePatch sizes[N];
    RpptROI rois[N];
};

template <size_t N>
void runN(std::ofstream& csv, fk::Stream_<fk::ParArch::GPU_AMD>& stream, fk::Ptr2D<fk::uchar3>& frame) {
    std::cout << "N = " << N << std::endl;
    const fk::RawPtr<fk::ND::_2D, fk::uchar3> frameView = frame.ptr();
    fk::Tensor<float> replicaOut(OUT_W, OUT_H, N, 3), branchesOut(OUT_W, OUT_H, N, 3), selectsOut(OUT_W, OUT_H, N, 3), mid(OUT_W, OUT_H, N, 3);

    // ---- The replica of fkl_fused_halfpixel
    auto buildReplica = [&](fk::Tensor<float>& out) {
        std::array<fk::Ptr2D<fk::uchar3>, N> crops;
        for (size_t i = 0; i < N; ++i) {
            crops[i] = frame.crop2D(fk::Rect(static_cast<uint>(i), static_cast<uint>(i), CROP_W, CROP_H));
        }
        const auto read = fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(crops).then(
            fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OUT_W, OUT_H)));
        return std::make_tuple(read, fk::VectorReorder<fk::float3, 2, 1, 0>::build(), fk::Mul<fk::float3>::build(fk::float3{ALPHA, ALPHA, ALPHA}),
                               fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]}),
                               fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]}), fk::TensorSplit<fk::float3>::build(out.ptr()));
    };
    const auto replica = buildReplica(replicaOut);

    // ---- FastRPP's arrays, as the benchmark's RPP variant fills them, on the same frame
    auto fill = [&](auto& side) {
        side.frameDesc = fastrpp::sharedSourceDesc(N, 3, SRC_H, SRC_W, RpptDataType::U8, RpptLayout::NHWC, frameView.dims.pitch);
        side.outDesc = RpptDesc(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
        for (size_t i = 0; i < N; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                side.mean[3 * i + c] = SUB[2 - c] / ALPHA;
                side.stdDev[3 * i + c] = DIV[2 - c] / ALPHA;
                side.permutation[3 * i + c] = static_cast<Rpp32u>(2 - c);
            }
            side.mirror[i] = 0;
            side.sizes[i] = RpptImagePatch{OUT_W, OUT_H};
            side.rois[i].xywhROI = {{static_cast<int>(i), static_cast<int>(i)}, CROP_W, CROP_H};
        }
        if (fastrpp::rppCreate(&side.handle, N, 0, stream.getHIPStream(), RppBackend::RPP_HIP_BACKEND) != rppStatusSuccess) {
            throw std::runtime_error("fastrpp::rppCreate failed");
        }
    };
    auto buildFastRPP = [&](auto& side, auto impl, fk::Tensor<float>& out) {
        using Impl = decltype(impl);
        const auto rmn = fastrpp::rppt_resize_mirror_normalize<N, RpptDataType::U8, RpptLayout::NHWC, RpptDataType::F32, RpptLayout::NCHW, 3, Impl>(
            frameView.data, &side.frameDesc, mid.ptr().data, &side.outDesc, side.sizes, RpptInterpolationType::BILINEAR, side.mean, side.stdDev,
            side.mirror, side.rois, RpptRoiType::XYWH, side.handle, RppBackend::RPP_HIP_BACKEND);
        const auto cp = fastrpp::rppt_channel_permute<N, RpptDataType::F32, RpptLayout::NCHW, RpptLayout::NCHW, 3, Impl>(
            mid.ptr().data, &side.outDesc, out.ptr().data, &side.outDesc, side.permutation, side.handle, RppBackend::RPP_HIP_BACKEND);
        if (rmn.status != RPP_SUCCESS || cp.status != RPP_SUCCESS) {
            throw std::runtime_error("a FastRPP operation failed its checks");
        }
        return std::make_pair(rmn, cp);
    };
    auto branches = std::make_unique<FastRPPSide<N, fastrpp::UniformBranches>>();
    auto selects = std::make_unique<FastRPPSide<N, fastrpp::Selects>>();
    fill(*branches);
    fill(*selects);
    const auto branchesOps = buildFastRPP(*branches, fastrpp::UniformBranches{}, branchesOut);
    const auto selectsOps = buildFastRPP(*selects, fastrpp::Selects{}, selectsOut);
    const auto branchesFused = fastrpp::detail::fuse(branchesOps.first, branchesOps.second);
    const auto selectsFused = fastrpp::detail::fuse(selectsOps.first, selectsOps.second);
    std::cout << "  kernel arguments: replica " << sizeof(std::get<0>(replica)) + sizeof(std::get<5>(replica)) << " B, FastRPP UniformBranches "
              << sizeof(branchesFused.first) + sizeof(branchesFused.second) << " B, FastRPP Selects "
              << sizeof(selectsFused.first) + sizeof(selectsFused.second) << " B" << std::endl;

    // ---- Agreement
    std::apply([&](const auto&... ops) { fk::executeOperations<DPP>(stream, ops...); }, replica);
    fk::executeOperations<DPP>(stream, branchesFused.first, branchesFused.second);
    fk::executeOperations<DPP>(stream, selectsFused.first, selectsFused.second);
    replicaOut.download(stream);
    branchesOut.download(stream);
    selectsOut.download(stream);
    stream.sync();
    auto maxDiff = [&](const fk::Tensor<float>& a) {
        double m = 0;
        for (size_t k = 0; k < N * 3 * OUT_W * OUT_H; ++k) {
            const double d = std::fabs(static_cast<double>(a.ptrPinned().data[k]) - static_cast<double>(replicaOut.ptrPinned().data[k]));
            m = std::isnan(d) ? d : std::max(m, d);
        }
        return m;
    };
    const double branchesDiff = maxDiff(branchesOut), selectsDiff = maxDiff(selectsOut);
    std::cout << "  agreement with the replica: UniformBranches max " << branchesDiff << ", Selects max " << selectsDiff
              << ((branchesDiff <= 1e-4 && selectsDiff <= 1e-4) ? "  ok" : "  FAIL") << std::endl;

    // ---- GPU time and the public path
    std::vector<Variant> variants{
        {"fkl_fused_halfpixel replica", [&] { std::apply([&](const auto&... ops) { fk::executeOperations<DPP>(stream, ops...); }, replica); },
         [&](double& build, double& run) {
             const auto t0 = Clock::now();
             const auto ops = buildReplica(replicaOut);
             const auto t1 = Clock::now();
             std::apply([&](const auto&... o) { fk::executeOperations<DPP>(stream, o...); }, ops);
             stream.sync();
             const auto t2 = Clock::now();
             build = msBetween(t0, t1);
             run = msBetween(t1, t2);
         },
         {}},
        {"FastRPP UniformBranches", [&] { fk::executeOperations<DPP>(stream, branchesFused.first, branchesFused.second); },
         [&](double& build, double& run) {
             const auto t0 = Clock::now();
             const auto ops = buildFastRPP(*branches, fastrpp::UniformBranches{}, branchesOut);
             const auto t1 = Clock::now();
             if (fastrpp::executeOperations(branches->handle, ops.first, ops.second) != RPP_SUCCESS) {
                 throw std::runtime_error("executeOperations failed");
             }
             stream.sync();
             const auto t2 = Clock::now();
             build = msBetween(t0, t1);
             run = msBetween(t1, t2);
         },
         {}},
        {"FastRPP Selects", [&] { fk::executeOperations<DPP>(stream, selectsFused.first, selectsFused.second); },
         [&](double& build, double& run) {
             const auto t0 = Clock::now();
             const auto ops = buildFastRPP(*selects, fastrpp::Selects{}, selectsOut);
             const auto t1 = Clock::now();
             if (fastrpp::executeOperations(selects->handle, ops.first, ops.second) != RPP_SUCCESS) {
                 throw std::runtime_error("executeOperations failed");
             }
             stream.sync();
             const auto t2 = Clock::now();
             build = msBetween(t0, t1);
             run = msBetween(t1, t2);
         },
         {}},
    };
    hipEvent_t e0{}, e1{};
    hipCheck(hipEventCreate(&e0), "hipEventCreate");
    hipCheck(hipEventCreate(&e1), "hipEventCreate");
    for (int round = 0; round < 3; ++round) {
        for (size_t k = 0; k < variants.size(); ++k) {
            Variant& v = variants[(k + static_cast<size_t>(round)) % variants.size()];
            const auto warmStart = Clock::now();
            while (msBetween(warmStart, Clock::now()) < 500.0) {
                v.launch();
                stream.sync();
            }
            std::vector<double> samples;
            for (int s = 0; s < 300; ++s) {
                hipCheck(hipEventRecord(e0, stream.getHIPStream()), "hipEventRecord");
                v.launch();
                hipCheck(hipEventRecord(e1, stream.getHIPStream()), "hipEventRecord");
                hipCheck(hipEventSynchronize(e1), "hipEventSynchronize");
                float ms = 0.f;
                hipCheck(hipEventElapsedTime(&ms, e0, e1), "hipEventElapsedTime");
                samples.push_back(static_cast<double>(ms));
            }
            v.gpuRoundMedians.push_back(median(samples));
        }
    }
    for (Variant& v : variants) {
        const auto warmStart = Clock::now();
        double build = 0, run = 0;
        while (msBetween(warmStart, Clock::now()) < 500.0) {
            v.publicPath(build, run);
        }
        std::vector<double> builds, runs;
        for (int s = 0; s < 300; ++s) {
            v.publicPath(build, run);
            builds.push_back(build);
            runs.push_back(run);
        }
        v.buildMedian = median(builds);
        v.runMedian = median(runs);
    }
    hipCheck(hipEventDestroy(e0), "hipEventDestroy");
    hipCheck(hipEventDestroy(e1), "hipEventDestroy");
    for (const Variant& v : variants) {
        const double gpu = median(v.gpuRoundMedians);
        const auto range = std::minmax_element(v.gpuRoundMedians.begin(), v.gpuRoundMedians.end());
        std::cout << "  " << v.name << ": GPU " << gpu << " ms (rounds " << *range.first << " - " << *range.second << "), build " << v.buildMedian
                  << " ms, run + sync " << v.runMedian << " ms" << std::endl;
        csv << N << "," << v.name << "," << gpu << "," << *range.first << "," << *range.second << "," << v.buildMedian << "," << v.runMedian
            << "\n";
    }
    csv.flush();
    static_cast<void>(fastrpp::rppDestroy(branches->handle, RppBackend::RPP_HIP_BACKEND));
    static_cast<void>(fastrpp::rppDestroy(selects->handle, RppBackend::RPP_HIP_BACKEND));
}

template <size_t... Ns>
void runAll(std::ofstream& csv, fk::Stream_<fk::ParArch::GPU_AMD>& stream, fk::Ptr2D<fk::uchar3>& frame, std::index_sequence<Ns...>) {
    (runN<Ns>(csv, stream, frame), ...);
}

} // namespace

int main() {
    try {
        fk::Stream_<fk::ParArch::GPU_AMD> stream;
        fk::Ptr2D<fk::uchar3> frame(SRC_W, SRC_H);
        uint32_t state = 12345u; // the benchmark's seeded noise
        for (int y = 0; y < SRC_H; ++y) {
            for (int x = 0; x < SRC_W; ++x) {
                unsigned char p[3];
                for (int c = 0; c < 3; ++c) {
                    if (c == 0) {
                        state = state * 1664525u + 1013904223u;
                    }
                    p[c] = static_cast<unsigned char>(state >> (24 - 8 * c));
                }
                frame.at(x, y) = fk::uchar3{p[0], p[1], p[2]};
            }
        }
        frame.upload(stream);
        stream.sync();
        std::ofstream csv("compare_implementations.csv");
        csv << "batch,variant,gpu_median_ms,gpu_round_min_ms,gpu_round_max_ms,build_median_ms,run_sync_median_ms\n";
        runAll(csv, stream, frame, std::index_sequence<10, 50, 100, 200, 300>{});
        return 0;
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
