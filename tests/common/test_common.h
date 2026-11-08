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

// What the tests share: RPP itself (the reference every FastRPP function is compared with), pinned and device buffers,
// seeded images, a double-precision reference of FastRPP's documented arithmetic, RPP's clamp region, the agreement
// machinery of fkl-benchmarks (082ce201, rpp/benchmark_rpp_preprocessing.h: Region, compareTensors, agreementRows), an
// FKL read of an F32 NCHW tensor, and fastrpp::Error's status.
#pragma once

#include <fast_rpp.h>
#include <rpp/rpp.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define FASTRPP_TEST_HIP(expr) fastrpp_test::hipCheck((expr), #expr, __FILE__, __LINE__)
#define FASTRPP_TEST_RPP(expr) fastrpp_test::statusCheck(static_cast<int>(expr), #expr, __FILE__, __LINE__)

namespace fastrpp_test {

// The tolerance of the required comparisons: 1e-4 absolute, as in fkl-benchmarks. The test parameters keep |output|
// below 256, where one float step is at most 2^-16 (about 1.5e-5).
constexpr double TOLERANCE = 1e-4;

inline void hipCheck(const hipError_t e, const char* expr, const char* file, const int line) {
    if (e != hipSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expr + ": " + hipGetErrorString(e));
    }
}

inline void statusCheck(const int status, const char* expr, const char* file, const int line) {
    if (status != 0) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expr + " returned " + std::to_string(status));
    }
}

// Pinned host memory: RPP's HIP back-end reads its parameter arrays on the GPU, in place
template <typename T>
class Pinned {
public:
    explicit Pinned(const size_t count) : count_(count) {
        FASTRPP_TEST_HIP(hipHostMalloc(reinterpret_cast<void**>(&data_), sizeof(T) * std::max<size_t>(count, 1)));
    }
    ~Pinned() { static_cast<void>(hipHostFree(data_)); }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    T* data() const { return data_; }
    size_t size() const { return count_; }
    T& operator[](const size_t i) { return data_[i]; }
    const T& operator[](const size_t i) const { return data_[i]; }

private:
    T* data_{nullptr};
    size_t count_;
};

template <typename T>
class Device {
public:
    explicit Device(const size_t count) : count_(count) {
        FASTRPP_TEST_HIP(hipMalloc(reinterpret_cast<void**>(&data_), sizeof(T) * std::max<size_t>(count, 1)));
    }
    // Raw bytes of the allocation (for guard bands and byte offsets)
    unsigned char* bytesPtr() const { return reinterpret_cast<unsigned char*>(data_); }
    ~Device() { static_cast<void>(hipFree(data_)); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    T* data() const { return data_; }
    size_t size() const { return count_; }
    size_t bytes() const { return sizeof(T) * count_; }
    void upload(const std::vector<T>& host) {
        FASTRPP_TEST_HIP(hipMemcpy(data_, host.data(), sizeof(T) * std::min(count_, host.size()), hipMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(count_);
        FASTRPP_TEST_HIP(hipMemcpy(host.data(), data_, bytes(), hipMemcpyDeviceToHost));
        return host;
    }
    void fillBytes(const int value) { FASTRPP_TEST_HIP(hipMemset(data_, value, bytes())); }

private:
    T* data_{nullptr};
    size_t count_;
};

// RPP 3.2.0's XYWH -> LTRB conversion kernel runs 256 * ceil(n / 256) threads with no bounds check, each rewriting one
// ROI: an ROI array given to RPP must hold that many ROIs
inline size_t rppRoiCapacity(const size_t n) { return 256 * ((n + 255) / 256); }

// ---- images: 8 bit RGB, interleaved, dense rows

// fkl-benchmarks' seeded noise: an LCG, one value per channel from bits 24, 16 and 8
inline std::vector<unsigned char> noiseImage(const int w, const int h, uint32_t seed) {
    std::vector<unsigned char> img(static_cast<size_t>(w) * h * 3);
    for (size_t k = 0; k < static_cast<size_t>(w) * h; ++k) {
        seed = seed * 1664525u + 1013904223u;
        img[k * 3 + 0] = static_cast<unsigned char>(seed >> 24);
        img[k * 3 + 1] = static_cast<unsigned char>(seed >> 16);
        img[k * 3 + 2] = static_cast<unsigned char>(seed >> 8);
    }
    return img;
}

// A smooth gradient, different per channel and per image id
inline std::vector<unsigned char> gradientImage(const int w, const int h, const int id) {
    std::vector<unsigned char> img(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double gx = static_cast<double>(x) / std::max(w - 1, 1), gy = static_cast<double>(y) / std::max(h - 1, 1);
            const double g[3] = {gx, gy, 0.5 * (gx + gy)};
            for (int c = 0; c < 3; ++c) {
                const double v = std::fmod(g[c] + 0.13 * id + 0.07 * c, 1.0);
                img[(static_cast<size_t>(y) * w + static_cast<size_t>(x)) * 3 + static_cast<size_t>(c)] =
                    static_cast<unsigned char>(std::lround(20.0 + 215.0 * v));
            }
        }
    }
    return img;
}

// ---- the double-precision reference of FastRPP's documented arithmetic

struct ImageSpec {
    const unsigned char* image; // top-left pixel of the image the ROI is in
    size_t pitch;               // bytes between rows
    int roiX, roiY, roiW, roiH;
    float mean[3], stdDev[3];
    uint32_t perm[3]; // output channel k = normalised input channel perm[k]
};

// FKL's HALF_PIXEL bilinear sample (FastRPP's sampling): src = max((x + 0.5) * f - 0.5, 0) with FKL's float factor f,
// x1 = floor(src), the second neighbour clamped to the ROI; channel c
inline double sampleHalfPixel(const ImageSpec& s, const int dstW, const int dstH, const int x, const int y, const int c) {
    const double fx = static_cast<double>(static_cast<float>(1.0 / (static_cast<double>(dstW) / s.roiW)));
    const double fy = static_cast<double>(static_cast<float>(1.0 / (static_cast<double>(dstH) / s.roiH)));
    const double sx = std::max((x + 0.5) * fx - 0.5, 0.0), sy = std::max((y + 0.5) * fy - 0.5, 0.0);
    const int x1 = static_cast<int>(std::floor(sx)), y1 = static_cast<int>(std::floor(sy));
    const int x2 = x1 + 1, y2 = y1 + 1;
    const int x2r = std::min(x2, s.roiW - 1), y2r = std::min(y2, s.roiH - 1);
    auto px = [&](const int xx, const int yy) {
        return static_cast<double>(
            s.image[static_cast<size_t>(s.roiY + yy) * s.pitch + static_cast<size_t>(s.roiX + xx) * 3 + static_cast<size_t>(c)]);
    };
    return px(x1, y1) * ((x2 - sx) * (y2 - sy)) + px(x2r, y1) * ((sx - x1) * (y2 - sy)) + px(x1, y2r) * ((x2 - sx) * (sy - y1)) +
           px(x2r, y2r) * ((sx - x1) * (sy - y1));
}

// Image i of rppt_resize_mirror_normalize, no image mirrored (then the permutation), NCHW: out[(k * dstH + y) * dstW + x]
inline void referenceImage(const ImageSpec& s, const int dstW, const int dstH, double* const out) {
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            double n[3];
            for (int c = 0; c < 3; ++c) {
                const float invStdDev = 1.0f / s.stdDev[c];
                n[c] = (sampleHalfPixel(s, dstW, dstH, x, y, c) - static_cast<double>(s.mean[c])) * static_cast<double>(invStdDev);
            }
            for (int k = 0; k < 3; ++k) {
                out[(static_cast<size_t>(k) * dstH + static_cast<size_t>(y)) * dstW + static_cast<size_t>(x)] = n[s.perm[k]];
            }
        }
    }
}

// RPP's clamp region: the destination pixels whose source position, computed as RPP 3.2.0 does (src = x * r + (r - 1) / 2
// in float, r = roiSize / dstSize), has floor >= roiSize - 2 in x or in y. RPP never reads the last column and row of the
// ROI there.
inline bool rppEdge(const ImageSpec& s, const int dstW, const int dstH, const int x, const int y) {
    const float rx = static_cast<float>(s.roiW) / static_cast<float>(dstW);
    const float ry = static_cast<float>(s.roiH) / static_cast<float>(dstH);
    const float sx = static_cast<float>(x) * rx + (rx - 1.0f) * 0.5f;
    const float sy = static_cast<float>(y) * ry + (ry - 1.0f) * 0.5f;
    return std::floor(sx) >= static_cast<float>(s.roiW - 2) || std::floor(sy) >= static_cast<float>(s.roiH - 2);
}

// ---- agreement: Region and compareTensors of fkl-benchmarks, the region given per pixel

struct Region {
    double maxAbs{0}, meanAbs{0}, fractionOver{0};
    size_t count{0};
};

// max, mean and fraction over TOLERANCE of |a - b| on the pixels of n NCHW images (3 planes of w x h) where
// inRegion(image, x, y) is true; a NaN (also inf - inf) makes the maximum NaN: it fails every tolerance
template <typename A, typename B, typename InRegion>
Region compare(const A* a, const B* b, const size_t n, const int w, const int h, InRegion inRegion) {
    Region r;
    double sum = 0;
    size_t over = 0;
    bool sawNaN = false;
    for (size_t i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (!inRegion(i, x, y)) {
                        continue;
                    }
                    const size_t idx = ((i * 3 + static_cast<size_t>(k)) * h + static_cast<size_t>(y)) * w + static_cast<size_t>(x);
                    const double d = std::fabs(static_cast<double>(a[idx]) - static_cast<double>(b[idx]));
                    if (std::isnan(d)) {
                        sawNaN = true;
                    } else if (d > r.maxAbs) {
                        r.maxAbs = d;
                    }
                    sum += d;
                    over += !(d <= TOLERANCE);
                    ++r.count;
                }
            }
        }
    }
    if (sawNaN) {
        r.maxAbs = std::numeric_limits<double>::quiet_NaN();
    }
    r.meanAbs = r.count ? sum / static_cast<double>(r.count) : 0;
    r.fractionOver = r.count ? static_cast<double>(over) / static_cast<double>(r.count) : 0;
    return r;
}

// The rows of a test, printed as fkl-benchmarks' agreementRows prints them; a required row that fails fails the test
class Report {
public:
    // A row passes when its maximum is within tolerance (TOLERANCE unless a row says otherwise)
    void row(const std::string& what, const Region& r, const bool required, const double tolerance = TOLERANCE) {
        const bool pass = r.maxAbs <= tolerance;
        allPass_ &= pass || !required;
        std::cout << "  " << what << ": max " << r.maxAbs << ", mean " << r.meanAbs << ", over 1e-4 " << r.fractionOver << " (" << r.count
                  << " values)" << (required ? (pass ? " ok" : " FAIL") : " (informational)");
        if (tolerance != TOLERANCE) {
            std::cout << " [tolerance " << tolerance << "]";
        }
        std::cout << std::endl;
    }
    void check(const std::string& what, const bool pass) {
        allPass_ &= pass;
        std::cout << "  " << what << (pass ? ": ok" : ": FAIL") << std::endl;
    }
    bool allPass() const { return allPass_; }
    int exitCode() const {
        std::cout << (allPass_ ? "all required checks passed" : "A REQUIRED CHECK FAILED") << std::endl;
        return allPass_ ? 0 : 1;
    }

private:
    bool allPass_{true};
};

// FKL's read of a whole F32 NCHW tensor, for a test whose chain starts with rppt_channel_permute: FastRPP's functions
// read only U8 sources, and an FKL IOp of the caller's can start a chain (fastrpp::executeOperations takes any FKL IOps)
inline auto readNCHW(void* const ptr, const RpptDesc& d) {
    fk::RawPtr<fk::ND::_3D, float> v;
    v.data = reinterpret_cast<float*>(static_cast<unsigned char*>(ptr) + d.offsetInBytes);
    v.dims = fk::PtrDims<fk::ND::_3D>(d.w, d.h, d.n, 3u, static_cast<uint>(d.w * sizeof(float)));
    return fk::TensorPack<fk::float3>::build(v);
}

// Whether every float of a buffer still holds the sentinel bytes it was filled with
inline bool holdsSentinel(const std::vector<float>& values, const uint32_t sentinel) {
    for (const float v : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        if (bits != sentinel) {
            return false;
        }
    }
    return true;
}

// Runs f and returns the status of the fastrpp::Error it throws, RPP_SUCCESS if it throws none; message gets its text
template <typename F>
RppStatus statusOf(F&& f, std::string& message) {
    try {
        f();
    } catch (const fastrpp::Error& e) {
        message = e.what();
        return e.status();
    }
    message = "no exception";
    return RPP_SUCCESS;
}

} // namespace fastrpp_test
