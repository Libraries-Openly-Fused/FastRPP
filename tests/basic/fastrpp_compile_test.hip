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

// Every public path of fast_rpp.h instantiated under the strict flags; nothing runs on the GPU (as FastNPP's basic
// test). Each function, with its template defaults and with explicit template arguments, must return exactly the IOp
// written below with FKL alone: FastRPP defines no operation and only translates RPP's arguments into FKL's. Its
// parameters must be those RPP 3.2.0 declares for the function, without the handle and the back-end; executeOperations
// alone takes RPP's own rppHandle_t and RppBackend. The chains executeOperations runs are instantiated on both
// back-ends: the GPU kernels and FKL's CPU executor.
#include <fast_rpp.h>

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {

using SourceView = fk::RawPtr<fk::ND::_2D, fk::uchar3>;
using TensorView = fk::RawPtr<fk::ND::_3D, float>;

// What each function must return, written with FKL alone
using ExpectedImage = decltype(fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(std::declval<SourceView>())
                                   .then(fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>::build(std::declval<fk::Size>()))
                                   .then(fk::Sub<fk::float3>::build(std::declval<fk::float3>()))
                                   .then(fk::Mul<fk::float3>::build(std::declval<fk::float3>())));
template <size_t N>
using ExpectedRMN = decltype(fk::BatchRead<fk::PlanePolicy::PROCESS_ALL>::build(std::declval<const std::array<ExpectedImage, N>&>()));
using ExpectedCP = decltype(fk::VectorReorderRT<fk::float3>::build(std::declval<fk::int3>()));
using ExpectedCopy = decltype(fk::TensorSplit<fk::float3>::build(std::declval<TensorView>()));

// What they return, with the template defaults
template <size_t N>
using RMN = decltype(fastrpp::rppt_resize_mirror_normalize<N>(
    std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RpptImagePatchPtr>(),
    std::declval<RpptInterpolationType>(), std::declval<Rpp32f*>(), std::declval<Rpp32f*>(), std::declval<Rpp32u*>(), std::declval<RpptROIPtr>(),
    std::declval<RpptRoiType>()));
using CP = decltype(fastrpp::rppt_channel_permute(std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(),
                                                  std::declval<RpptDescPtr>(), std::declval<Rpp32u*>()));
using Copy =
    decltype(fastrpp::rppt_copy(std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(), std::declval<RpptDescPtr>()));

static_assert(std::is_same_v<RMN<1>, ExpectedRMN<1>> && std::is_same_v<RMN<300>, ExpectedRMN<300>>,
              "rppt_resize_mirror_normalize returns FKL's BatchRead of PerThreadRead -> ResizeHalfPixel -> Sub -> Mul");
static_assert(std::is_same_v<CP, ExpectedCP>, "rppt_channel_permute returns FKL's VectorReorderRT");
static_assert(std::is_same_v<Copy, ExpectedCopy>, "rppt_copy returns FKL's TensorSplit");
static_assert(fk::opIs<fk::ReadType, RMN<300>>, "rppt_resize_mirror_normalize returns a read IOp");
static_assert(fk::opIs<fk::BinaryType, CP>, "rppt_channel_permute returns a compute IOp");
static_assert(fk::opIs<fk::WriteType, Copy>, "rppt_copy returns a write IOp");

// The same types with explicit template arguments (the defaults)
static_assert(std::is_same_v<decltype(fastrpp::rppt_resize_mirror_normalize<300, RpptDataType::U8, RpptLayout::NHWC, RpptDataType::F32,
                                                                              RpptLayout::NCHW, 3>(
                                 std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(),
                                 std::declval<RpptImagePatchPtr>(), std::declval<RpptInterpolationType>(), std::declval<Rpp32f*>(),
                                 std::declval<Rpp32f*>(), std::declval<Rpp32u*>(), std::declval<RpptROIPtr>(), std::declval<RpptRoiType>())),
                             RMN<300>>,
              "explicit template arguments");
static_assert(std::is_same_v<decltype(fastrpp::rppt_channel_permute<RpptDataType::F32, RpptLayout::NCHW, RpptLayout::NCHW, 3>(
                                 std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(),
                                 std::declval<Rpp32u*>())),
                             CP>,
              "explicit template arguments");
static_assert(std::is_same_v<decltype(fastrpp::rppt_copy<RpptDataType::F32, RpptLayout::NCHW, RpptLayout::NCHW, 3>(
                                 std::declval<RppPtr_t>(), std::declval<RpptDescPtr>(), std::declval<RppPtr_t>(), std::declval<RpptDescPtr>())),
                             Copy>,
              "explicit template arguments");

// The parameters: RPP's declaration of the function (rppt_*.h, extern "C") without its rppHandle_t and its RppBackend
template <typename F>
struct ParametersOf;
template <typename R, typename... Args>
struct ParametersOf<R (*)(Args...)> {
    using type = std::tuple<Args...>;
};
template <typename T>
constexpr bool isHandleOrBackend = std::is_same_v<T, rppHandle_t> || std::is_same_v<T, RppBackend>;
template <typename F>
struct ParametersWithoutHandleAndBackend;
template <typename R, typename... Args>
struct ParametersWithoutHandleAndBackend<R (*)(Args...)> {
    using type = decltype(std::tuple_cat(std::declval<std::conditional_t<isHandleOrBackend<Args>, std::tuple<>, std::tuple<Args>>>()...));
};
template <typename FastRpp, typename Rpp>
constexpr bool rppParametersWithoutHandleAndBackend =
    std::is_same_v<typename ParametersOf<FastRpp>::type, typename ParametersWithoutHandleAndBackend<Rpp>::type> &&
    std::tuple_size_v<typename ParametersOf<FastRpp>::type> + 2 == std::tuple_size_v<typename ParametersOf<Rpp>::type>;

static_assert(rppParametersWithoutHandleAndBackend<decltype(&fastrpp::rppt_resize_mirror_normalize<300>),
                                                   decltype(&::rppt_resize_mirror_normalize)>,
              "rppt_resize_mirror_normalize takes RPP's parameters without the handle and the back-end");
static_assert(rppParametersWithoutHandleAndBackend<decltype(&fastrpp::rppt_channel_permute<>), decltype(&::rppt_channel_permute)>,
              "rppt_channel_permute takes RPP's parameters without the handle and the back-end");
static_assert(rppParametersWithoutHandleAndBackend<decltype(&fastrpp::rppt_copy<>), decltype(&::rppt_copy)>,
              "rppt_copy takes RPP's parameters without the handle and the back-end");

// The chains executeOperations runs; an FKL read of the caller's can start one (fk::TensorPack of an F32 NCHW tensor)
using PackRead = decltype(fk::TensorPack<fk::float3>::build(std::declval<TensorView>()));
static_assert(std::is_same_v<decltype(&fastrpp::executeOperations<RMN<300>, CP, Copy>),
                             void (*)(rppHandle_t, RppBackend, const RMN<300>&, const CP&, const Copy&)>,
              "executeOperations takes RPP's own rppHandle_t and RppBackend, and the IOps");
// RPP_HOST_BACKEND runs on FKL's multithreaded CPU executor when the build enables it (FASTRPP_CPU_MULTITHREAD)
#if defined(FK_CPU_MULTITHREAD_ENABLED)
static_assert(fastrpp::detail::HOST_PAR_ARCH == fk::ParArch::CPU_MULTITHREAD, "RPP_HOST_BACKEND on FKL's CPU_MULTITHREAD");
#else
static_assert(fastrpp::detail::HOST_PAR_ARCH == fk::ParArch::CPU, "RPP_HOST_BACKEND on FKL's CPU");
#endif
static_assert(std::is_same_v<decltype(fastrpp::executeOperations(std::declval<rppHandle_t>(), std::declval<RppBackend>(),
                                                                 std::declval<const RMN<300>&>(), std::declval<const CP&>(),
                                                                 std::declval<const Copy&>())),
                             void>,
              "executeOperations returns nothing: errors are exceptions");
static_assert(std::is_base_of_v<std::runtime_error, fastrpp::Error>, "fastrpp::Error is a std::runtime_error");
static_assert(std::is_same_v<decltype(std::declval<const fastrpp::Error&>().status()), RppStatus>, "fastrpp::Error carries RPP's status");

} // namespace

int main() {
    // Taking the addresses instantiates the function definitions and the kernels (decltype of a call does not)
    const auto rmn1 = &fastrpp::rppt_resize_mirror_normalize<1>;
    const auto rmn300 = &fastrpp::rppt_resize_mirror_normalize<300>;
    const auto cp = &fastrpp::rppt_channel_permute<>;
    const auto copy = &fastrpp::rppt_copy<>;
    const auto chain = &fastrpp::executeOperations<RMN<300>, CP, Copy>;
    const auto twoPermutations = &fastrpp::executeOperations<RMN<10>, CP, CP, Copy>;
    const auto resizeNormalizeOnly = &fastrpp::executeOperations<RMN<1>, Copy>;
    const auto permuteOnly = &fastrpp::executeOperations<PackRead, CP, Copy>;
    static_cast<void>(rmn1);
    static_cast<void>(rmn300);
    static_cast<void>(cp);
    static_cast<void>(copy);
    static_cast<void>(chain);
    static_cast<void>(twoPermutations);
    static_cast<void>(resizeNormalizeOnly);
    static_cast<void>(permuteOnly);
    return 0;
}
