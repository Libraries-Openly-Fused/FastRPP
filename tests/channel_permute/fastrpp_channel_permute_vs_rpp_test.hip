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

// fastrpp::rppt_channel_permute against RPP's rppt_channel_permute on the same F32 NCHW tensor, on each execution
// back-end. FastRPP's compute IOp runs between an FKL read of the tensor (fk::TensorPack, the test's own IOp) and
// fastrpp::rppt_copy's write IOp: executeOperations(handle, backend, read, cp, copy), one GPU kernel or FKL's CPU
// executor on host memory, against RPP on the same back-end. Each of the six permutations, the same for every image of the
// batch (FastRPP rejects triples that differ between images: fastrpp_errors_test); (1,2,0) and (2,0,1) are not their own
// inverse, so a library that applied out[perm[k]] = in[k] instead of out[k] = in[perm[k]] would fail. The permutation
// only moves values: FastRPP must be bitwise equal to RPP, and to the expected tensor computed on the host.
#include <common/test_common.h>

using namespace fastrpp_test;

namespace {

template <size_t N>
void runCase(Report& report, const int w, const int h, const uint32_t (&perm)[3], const RppBackend backend) {
    std::cout << backendName(backend) << ": channel permute (" << perm[0] << ", " << perm[1] << ", " << perm[2] << ") " << w << "x" << h
              << ", N " << N << std::endl;
    const size_t plane = static_cast<size_t>(w) * h, count = N * 3 * plane;
    std::vector<float> host(count);
    uint32_t state = 4242u + perm[0] * 3u + perm[1];
    for (float& v : host) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(static_cast<int32_t>((state >> 8) % 40001u) - 20000) / 100.0f; // -200 to 200, two decimals
    }
    Buffer<float> src(count, backend);
    src.upload(host);
    Pinned<Rpp32u> permutation(3 * N);
    std::vector<float> expected(count);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < 3; ++k) {
            permutation[3 * i + k] = perm[k];
            std::copy(host.begin() + static_cast<std::ptrdiff_t>((3 * i + perm[k]) * plane),
                      host.begin() + static_cast<std::ptrdiff_t>((3 * i + perm[k] + 1) * plane),
                      expected.begin() + static_cast<std::ptrdiff_t>((3 * i + k) * plane));
        }
    }
    RpptDesc desc(N, 3, static_cast<Rpp32u>(h), static_cast<Rpp32u>(w), RpptDataType::F32, RpptLayout::NCHW);
    hipStream_t stream{};
    FASTRPP_TEST_HIP(hipStreamCreate(&stream));

    Buffer<float> fastOut(count, backend);
    fastOut.fillBytes(0xFF);
    const rppHandle_t fastHandle = createHandle(N, backend, stream);
    const auto read = readNCHW(src.data(), desc);
    const auto cp = fastrpp::rppt_channel_permute(src.data(), &desc, fastOut.data(), &desc, permutation.data());
    const auto copy = fastrpp::rppt_copy(fastOut.data(), &desc, fastOut.data(), &desc);
    fastrpp::executeOperations(fastHandle, backend, read, cp, copy);
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> fast = fastOut.download();
    FASTRPP_TEST_RPP(rppDestroy(fastHandle, backend));

    Buffer<float> rppOut(count, backend);
    rppOut.fillBytes(0xFF);
    const rppHandle_t rppHandle = createHandle(N, backend, stream);
    FASTRPP_TEST_RPP(rppt_channel_permute(src.data(), &desc, rppOut.data(), &desc, permutation.data(), rppHandle, backend));
    FASTRPP_TEST_HIP(hipStreamSynchronize(stream));
    const std::vector<float> rpp = rppOut.download();
    FASTRPP_TEST_RPP(rppDestroy(rppHandle, backend));
    FASTRPP_TEST_HIP(hipStreamDestroy(stream));

    report.check("FastRPP is bitwise equal to RPP", std::memcmp(fast.data(), rpp.data(), sizeof(float) * count) == 0);
    report.check("FastRPP is bitwise equal to the expected tensor (out[k] = in[perm[k]])",
                 std::memcmp(fast.data(), expected.data(), sizeof(float) * count) == 0);
    report.check("RPP is bitwise equal to the expected tensor", std::memcmp(rpp.data(), expected.data(), sizeof(float) * count) == 0);
}

} // namespace

int main() {
    try {
        Report report;
        const uint32_t perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        for (const RppBackend backend : BACKENDS) {
            for (const auto& perm : perms) {
                runCase<6>(report, 40, 24, perm, backend);
            }
            runCase<12>(report, 64, 128, perms[3], backend);
            runCase<300>(report, 64, 128, perms[5], backend); // the benchmark's permutation and batch
        }
        return report.exitCode();
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 2;
    }
}
