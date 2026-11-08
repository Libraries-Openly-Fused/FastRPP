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

// Every public path of fast_rpp.h instantiated, under the strict flags, for both implementations: each function with its
// template defaults and with explicit template arguments, each operation alone and the chain, and the FKL operations
// an operation exposes for fk::executeOperations. Nothing runs on the GPU (as FastNPP's basic test).
#include <fast_rpp.h>

#include <type_traits>
#include <utility>

namespace {

template <size_t N, typename Impl>
void instantiate() {
    using RMN = decltype(fastrpp::rppt_resize_mirror_normalize<N, RpptDataType::U8, RpptLayout::NHWC, RpptDataType::F32, RpptLayout::NCHW, 3, Impl>(
        std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(),
        std::declval<RpptImagePatchPtr>(), std::declval<RpptInterpolationType>(), std::declval<Rpp32f*>(), std::declval<Rpp32f*>(),
        std::declval<Rpp32u*>(), std::declval<RpptROIPtr>(), std::declval<RpptRoiType>(), std::declval<fastrpp::rppHandle_t>(),
        std::declval<RppBackend>()));
    using CP = decltype(fastrpp::rppt_channel_permute<N, RpptDataType::F32, RpptLayout::NCHW, RpptLayout::NCHW, 3, Impl>(
        std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<Rpp32u*>(),
        std::declval<fastrpp::rppHandle_t>(), std::declval<RppBackend>()));
    static_assert(std::is_same_v<RMN, fastrpp::ResizeMirrorNormalize<N, Impl>>, "rppt_resize_mirror_normalize returns an operation");
    static_assert(std::is_same_v<CP, fastrpp::ChannelPermute<N, Impl>>, "rppt_channel_permute returns an operation");
    static_assert(RMN::STARTS_CHAIN && !RMN::CONTINUES_CHAIN && CP::STARTS_CHAIN && CP::CONTINUES_CHAIN, "chain roles");
    using Alone = decltype(fastrpp::executeOperations(std::declval<fastrpp::rppHandle_t>(), std::declval<const RMN&>()));
    using PermuteAlone = decltype(fastrpp::executeOperations(std::declval<fastrpp::rppHandle_t>(), std::declval<const CP&>()));
    using Chain = decltype(fastrpp::executeOperations(std::declval<fastrpp::rppHandle_t>(), std::declval<const RMN&>(), std::declval<const CP&>()));
    using Twice = decltype(fastrpp::executeOperations(std::declval<fastrpp::rppHandle_t>(), std::declval<const CP&>(), std::declval<const CP&>()));
    static_assert(std::is_same_v<Alone, RppStatus> && std::is_same_v<PermuteAlone, RppStatus> && std::is_same_v<Chain, RppStatus> &&
                      std::is_same_v<Twice, RppStatus>,
                  "executeOperations returns an RppStatus");
    // The FKL operations of each operation, for fk::executeOperations
    using RmnRead = decltype(std::declval<const RMN&>().read());
    using RmnWrite = decltype(std::declval<const RMN&>().write());
    using CpRead = decltype(std::declval<const CP&>().read());
    using FklRun = decltype(fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(
        std::declval<fk::Stream_<fk::ParArch::GPU_AMD>&>(), std::declval<const RmnRead&>(), std::declval<const RmnWrite&>()));
    static_assert(std::is_same_v<FklRun, void> && !std::is_same_v<CpRead, void>, "the FKL operations fuse");
    // Taking their addresses instantiates the definitions, and the kernels (decltype of a call does not)
    const auto rmnFunction = &fastrpp::rppt_resize_mirror_normalize<N, RpptDataType::U8, RpptLayout::NHWC, RpptDataType::F32, RpptLayout::NCHW, 3, Impl>;
    const auto cpFunction = &fastrpp::rppt_channel_permute<N, RpptDataType::F32, RpptLayout::NCHW, RpptLayout::NCHW, 3, Impl>;
    const auto runAlone = &fastrpp::executeOperations<RMN>;
    const auto runPermuteAlone = &fastrpp::executeOperations<CP>;
    const auto runChain = &fastrpp::executeOperations<RMN, CP>;
    const auto runTwice = &fastrpp::executeOperations<CP, CP>;
    static_cast<void>(rmnFunction);
    static_cast<void>(cpFunction);
    static_cast<void>(runAlone);
    static_cast<void>(runPermuteAlone);
    static_cast<void>(runChain);
    static_cast<void>(runTwice);
}

// The template defaults: U8 NHWC -> F32 NCHW, F32 NCHW -> F32 NCHW, 3 channels, the default implementation
using DefaultRMN = decltype(fastrpp::rppt_resize_mirror_normalize<10>(
    std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RpptImagePatchPtr>(),
    std::declval<RpptInterpolationType>(), std::declval<Rpp32f*>(), std::declval<Rpp32f*>(), std::declval<Rpp32u*>(), std::declval<RpptROIPtr>(),
    std::declval<RpptRoiType>(), std::declval<fastrpp::rppHandle_t>(), std::declval<RppBackend>()));
static_assert(std::is_same_v<DefaultRMN, fastrpp::ResizeMirrorNormalize<10, fastrpp::DefaultImplementation>>, "template defaults");
using DefaultCP = decltype(fastrpp::rppt_channel_permute<10>(std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(),
                                                             std::declval<RpptDescPtr>(), std::declval<Rpp32u*>(),
                                                             std::declval<fastrpp::rppHandle_t>(), std::declval<RppBackend>()));
static_assert(std::is_same_v<DefaultCP, fastrpp::ChannelPermute<10, fastrpp::DefaultImplementation>>, "template defaults");

// The six permutation cases and the rejected triples
static_assert(std::is_same_v<decltype(fastrpp::detail::permutationCase(0u, 1u, 2u)), int>, "permutationCase");

} // namespace

int main() {
    instantiate<1, fastrpp::UniformBranches>();
    instantiate<1, fastrpp::Selects>();
    instantiate<300, fastrpp::UniformBranches>();
    instantiate<300, fastrpp::Selects>();
    // permutationCase, run on the host: the six permutations in order, and the triples that are not permutations
    const Rpp32u perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (int k = 0; k < 6; ++k) {
        if (fastrpp::detail::permutationCase(perms[k][0], perms[k][1], perms[k][2]) != k) {
            return 1;
        }
    }
    if (fastrpp::detail::permutationCase(0, 0, 1) != -1 || fastrpp::detail::permutationCase(3, 1, 0) != -1) {
        return 1;
    }
    return 0;
}
