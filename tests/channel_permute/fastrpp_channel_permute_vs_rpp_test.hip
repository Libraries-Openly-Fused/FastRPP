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

// fastrpp::rppt_channel_permute against RPP's rppt_channel_permute on the same F32 NCHW tensor, for both implementations
// of FastRPP's per-image permutation (UniformBranches, Selects). Every image has its own permutation and the six
// permutations alternate within the batch; (1,2,0) and (2,0,1) are not their own inverse, so a library that applied
// out[perm[k]] = in[k] instead of out[k] = in[perm[k]] would fail. The permutation only moves values: FastRPP must be
// bitwise equal to RPP, and to the expected tensor computed on the host.
#include <common/test_common.h>

using namespace fastrpp_test;

namespace {

template <size_t N, typename Impl>
void runCase(Report& report, const int w, const int h) {
    std::cout << "channel permute " << w << "x" << h << ", N " << N << ", " << implementationName(Impl{}) << std::endl;
    const uint32_t perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    const size_t plane = static_cast<size_t>(w) * h, count = N * 3 * plane;
    std::vector<float> host(count);
    uint32_t state = 4242u;
    for (float& v : host) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(static_cast<int32_t>((state >> 8) % 40001u) - 20000) / 100.0f; // -200 to 200, two decimals
    }
    Device<float> src(count);
    src.upload(host);
    Pinned<Rpp32u> permutation(3 * N);
    std::vector<float> expected(count);
    for (size_t i = 0; i < N; ++i) {
        const uint32_t* const p = perms[(i * 5) % 6]; // 0, 5, 4, 3, 2, 1, ...: neighbours differ
        for (size_t k = 0; k < 3; ++k) {
            permutation[3 * i + k] = p[k];
            std::copy(host.begin() + static_cast<std::ptrdiff_t>((3 * i + p[k]) * plane),
                      host.begin() + static_cast<std::ptrdiff_t>((3 * i + p[k] + 1) * plane),
                      expected.begin() + static_cast<std::ptrdiff_t>((3 * i + k) * plane));
        }
    }
    RpptDesc desc(N, 3, static_cast<Rpp32u>(h), static_cast<Rpp32u>(w), RpptDataType::F32, RpptLayout::NCHW);
    hipStream_t stream{};
    FASTRPP_TEST_HIP(hipStreamCreate(&stream));

    Device<float> fastOut(count);
    fastOut.fillBytes(0xFF);
    fastrpp::rppHandle_t fastHandle{};
    FASTRPP_TEST_RPP(fastrpp::rppCreate(&fastHandle, N, 0, stream, RppBackend::RPP_HIP_BACKEND));
    const auto op = fastrpp::rppt_channel_permute<N, RpptDataType::F32, RpptLayout::NCHW, RpptLayout::NCHW, 3, Impl>(
        src.data(), &desc, fastOut.data(), &desc, permutation.data(), fastHandle, RppBackend::RPP_HIP_BACKEND);
    report.check("fastrpp::rppt_channel_permute returns RPP_SUCCESS", op.status == RPP_SUCCESS);
    FASTRPP_TEST_RPP(fastrpp::executeOperations(fastHandle, op));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> fast = fastOut.download();
    FASTRPP_TEST_RPP(fastrpp::rppDestroy(fastHandle, RppBackend::RPP_HIP_BACKEND));

    Device<float> rppOut(count);
    rppOut.fillBytes(0xFF);
    rppHandle_t rppHandle{};
    FASTRPP_TEST_RPP(rppCreate(&rppHandle, N, 0, stream, RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_RPP(rppt_channel_permute(src.data(), &desc, rppOut.data(), &desc, permutation.data(), rppHandle, RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> rpp = rppOut.download();
    FASTRPP_TEST_RPP(rppDestroy(rppHandle, RppBackend::RPP_HIP_BACKEND));
    FASTRPP_TEST_HIP(hipStreamDestroy(stream));

    report.check("FastRPP is bitwise equal to RPP", std::memcmp(fast.data(), rpp.data(), sizeof(float) * count) == 0);
    report.check("FastRPP is bitwise equal to the expected tensor (out[k] = in[perm[k]])",
                 std::memcmp(fast.data(), expected.data(), sizeof(float) * count) == 0);
    report.check("RPP is bitwise equal to the expected tensor", std::memcmp(rpp.data(), expected.data(), sizeof(float) * count) == 0);
}

template <typename Impl>
void runAll(Report& report) {
    runCase<6, Impl>(report, 40, 24);
    runCase<12, Impl>(report, 64, 128);
    runCase<300, Impl>(report, 64, 128);
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
