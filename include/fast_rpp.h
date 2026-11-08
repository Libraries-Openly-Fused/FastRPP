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

#ifndef FAST_RPP_H
#define FAST_RPP_H

// FastRPP: functions of AMD's RPP tensor API (rppt_*, RPP 3.2.0) built with the Fused Kernel Library (FKL), HIP only.
//
// Each fastrpp::rppt_* function takes RPP's run-time arguments, checks them on the host and, instead of running, returns
// an operation. fastrpp::executeOperations(handle, op1, op2, ...) runs a chain of operations as ONE kernel on the
// handle's stream: the destination of each operation must be the source of the next one, and that intermediate buffer
// is never written. Template parameters give FKL what RPP's descriptors only know at run time (the batch size, the
// types and layouts); the run-time descriptors must match them.
//
// RPP's types are used as they are (rppdefs.h): FastRPP needs RPP's headers, never librpp. The handle is FastRPP's own.
// Semantics and every difference from RPP 3.2.0: README.md.

#include <rpp/rpp_backend.h>
#include <rpp/rppdefs.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fastrpp {

// ===================================================== handle =====================================================

// FastRPP's handle: the batch size and the stream. FKL needs no scratch memory, so it holds no device memory. It is a
// different type from RPP's rppHandle_t: an RPP handle cannot be passed to FastRPP by mistake.
struct rppHandle {
    size_t batchSize;
    fk::Stream_<fk::ParArch::GPU_AMD> stream; // wraps the hipStream_t given to rppCreate, which it never destroys
};
typedef rppHandle* rppHandle_t;

// RPP's rppCreate. Only RppBackend::RPP_HIP_BACKEND is implemented; numThreads is ignored (RPP uses it only for the host
// back-end); stream nullptr is the null stream.
inline rppStatus_t rppCreate(rppHandle_t* handle, size_t nBatchSize, Rpp32u numThreads = 0, void* stream = nullptr,
                             RppBackend backend = RppBackend::RPP_HOST_BACKEND) {
    static_cast<void>(numThreads);
    if (handle == nullptr || nBatchSize == 0) {
        return rppStatusBadParm;
    }
    if (backend != RppBackend::RPP_HIP_BACKEND) {
        return rppStatusNotImplemented;
    }
    try {
        *handle = new rppHandle{nBatchSize, fk::Stream_<fk::ParArch::GPU_AMD>(static_cast<hipStream_t>(stream))};
    } catch (const std::exception&) {
        return rppStatusAllocFailed;
    }
    return rppStatusSuccess;
}

// RPP's rppDestroy. As in RPP 3.2.0's GPU build, either back-end value destroys the handle (RPP_HOST_BACKEND is the
// default argument). The stream given to rppCreate is the caller's and stays alive, as in RPP.
inline rppStatus_t rppDestroy(rppHandle_t handle, RppBackend backend = RppBackend::RPP_HOST_BACKEND) {
    if (backend != RppBackend::RPP_HOST_BACKEND && backend != RppBackend::RPP_HIP_BACKEND) {
        return rppStatusNotImplemented;
    }
    if (handle == nullptr) {
        return rppStatusBadParm;
    }
    delete handle;
    return rppStatusSuccess;
}

// ============================================= shared-source batch ================================================

// FastRPP addresses image i of a source tensor as RPP does: (Rpp8u*)srcPtr + offsetInBytes + i * nStride elements.
// FastRPP also accepts nStride = 0, which RPP 3.2.0 does not document: a shared-source batch, n views of one h x w
// image, each one selected by its ROI. A batch of crops of one frame is then one descriptor (n crops, the frame's h x w)
// and n ROIs, and the crops are read in place. This builds that descriptor; rowStride is hStride, in elements.
inline RpptDesc sharedSourceDesc(const Rpp32u n, const Rpp32u c, const Rpp32u h, const Rpp32u w, const RpptDataType dataType,
                                 const RpptLayout layout, const Rpp32u rowStride, const Rpp32u offsetInBytes = 0) {
    RpptStrides strides{};
    strides.nStride = 0;
    strides.hStride = rowStride;
    if (layout == RpptLayout::NHWC) {
        strides.cStride = 1;
        strides.wStride = c;
    } else {
        strides.cStride = rowStride * h;
        strides.wStride = 1;
    }
    return RpptDesc(n, c, h, w, dataType, layout, offsetInBytes, strides);
}

// ================================================ implementations =================================================

// How the per-image mirror and channel permutation run on the GPU. Both compute RPP's result; README.md says which one
// FastRPP uses by default and why (measured).
//   UniformBranches  the mirror and the six permutations are branches on a per-image value, uniform within a block
//   Selects          the mirror selects the source column; the permutation is fk::VectorReorderRT (run-time indices)
struct UniformBranches {};
struct Selects {};
using DefaultImplementation = UniformBranches;

namespace detail {
// FKL's operation macros (DECLARE_*_PARENT) name Point, BatchUtils, BatchRead and PlanePolicy without qualification;
// MirrorX::getActiveThreads returns an ActiveThreads
using fk::ActiveThreads;
using fk::BatchRead;
using fk::BatchUtils;
using fk::PlanePolicy;
using fk::Point;

// ---- MirrorX: RPP's mirror, a conditional read

struct MirrorParams {
    Rpp32u mirror; // 1 mirrors the image horizontally; any other value does not (RPP mirrors when mirrorTensor[i] == 1)
};

// Reads its back operation at x' = width - 1 - x when the image is mirrored, else at x. After the resize this flips the
// resized image, as RPP does (it samples the source at the position of destination column width - 1 - x).
template <typename Impl, typename BackIOp_ = NullType>
struct MirrorX {
    static_assert(fk::isAnyCompleteReadType<BackIOp_>, "MirrorX reads from a complete Read or ReadBack operation");

private:
    using SelfType = MirrorX<Impl, BackIOp_>;

public:
    FK_STATIC_STRUCT(MirrorX, SelfType)
    using Parent = fk::ReadBackOperation<typename BackIOp_::Operation::OutputType, MirrorParams, BackIOp_,
                                         typename BackIOp_::Operation::OutputType, MirrorX<Impl, BackIOp_>>;
    DECLARE_READBACK_PARENT

    FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params, const BackIOp& backIOp) {
        const int lastX = static_cast<int>(fk::NumElems::x(thread, backIOp)) - 1;
        if constexpr (std::is_same_v<Impl, UniformBranches>) {
            if (params.mirror == 1u) {
                return BackIOp::Operation::exec(Point{lastX - thread.x, thread.y, thread.z}, backIOp);
            }
            return BackIOp::Operation::exec(thread, backIOp);
        } else {
            const Point source{params.mirror == 1u ? lastX - thread.x : thread.x, thread.y, thread.z};
            return BackIOp::Operation::exec(source, backIOp);
        }
    }
    FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
        return fk::NumElems::x(thread, opData.backIOp);
    }
    FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
        return fk::NumElems::y(thread, opData.backIOp);
    }
    FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
        return fk::NumElems::z(thread, opData.backIOp);
    }
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
        return {num_elems_x(Point{0, 0, 0}, opData), num_elems_y(Point{0, 0, 0}, opData), num_elems_z(Point{0, 0, 0}, opData)};
    }
};

// Before fusion: the parameters only; the fusion (.then) gives it the read it applies to
template <typename Impl>
struct MirrorX<Impl, NullType> {
private:
    using SelfType = MirrorX<Impl, NullType>;

public:
    FK_STATIC_STRUCT(MirrorX, SelfType)
    using Parent = fk::IncompleteReadBackOperation<NullType, MirrorParams, NullType, NullType, MirrorX<Impl, NullType>>;
    DECLARE_INCOMPLETEREADBACK_PARENT

    FK_HOST_FUSE InstantiableType build(const MirrorParams& params) { return InstantiableType{{params, NullType{}}}; }

    template <typename NewBackIOp>
    FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const InstantiableType& iOp) {
        return MirrorX<Impl, NewBackIOp>::build(iOp.params, backIOp);
    }
};

// ---- PermuteChannels: RPP's channel permutation as six compile-time swizzles chosen by a per-image case

// The case of a permutation (a, b, c), where output channel k is input channel (a, b, c)[k]: (0,1,2) 0, (0,2,1) 1,
// (1,0,2) 2, (1,2,0) 3, (2,0,1) 4, (2,1,0) 5; -1 if it is not a permutation of 0, 1, 2
inline int permutationCase(const Rpp32u a, const Rpp32u b, const Rpp32u c) {
    if (a > 2u || b > 2u || c > 2u || a == b || a == c || b == c) {
        return -1;
    }
    return static_cast<int>(a) * 2 + (b > c ? 1 : 0);
}

template <typename T>
struct PermuteChannels {
private:
    using SelfType = PermuteChannels<T>;

public:
    FK_STATIC_STRUCT(PermuteChannels, SelfType)
    using Parent = fk::BinaryOperation<T, int, T, PermuteChannels<T>>;
    DECLARE_BINARY_PARENT

    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const ParamsType& permutation) {
        static_assert(fk::cn<T> == 3, "PermuteChannels permutes 3 channels");
        switch (permutation) {
        case 0:
            return {input.x, input.y, input.z};
        case 1:
            return {input.x, input.z, input.y};
        case 2:
            return {input.y, input.x, input.z};
        case 3:
            return {input.y, input.z, input.x};
        case 4:
            return {input.z, input.x, input.y};
        default:
            return {input.z, input.y, input.x};
        }
    }
};

// The per-image permutation operation of an implementation, from RPP's triple
template <typename Impl>
inline auto permutationOp(const Rpp32u* const p) {
    if constexpr (std::is_same_v<Impl, UniformBranches>) {
        return PermuteChannels<fk::float3>::build(permutationCase(p[0], p[1], p[2]));
    } else {
        return fk::VectorReorderRT<fk::float3>::build(fk::int3{static_cast<int>(p[0]), static_cast<int>(p[1]), static_cast<int>(p[2])});
    }
}

// ---- views and checks

using SourceView = fk::RawPtr<fk::ND::_2D, fk::uchar3>;
using TensorView = fk::RawPtr<fk::ND::_3D, float>;

// A dense NCHW float tensor of n images, the strides RpptDesc's constructor gives: FKL's TensorSplit and TensorPack put
// channel k of a pixel k * w * h elements after channel 0, and image z at z * 3 planes
inline bool isDenseNCHW(const RpptDesc& d) {
    return d.strides.wStride == 1u && d.strides.hStride == d.w && d.strides.cStride == d.h * d.w && d.strides.nStride == d.c * d.h * d.w;
}

// FKL addresses image z of a 3D tensor with a 32-bit byte offset (PtrAccessor<ND::_3D>: plane_pitch * color_planes * z),
// so a float tensor must fit in 4 GiB, offsetInBytes included
inline bool fitsFklTensor(const RpptDesc& d) {
    return static_cast<uint64_t>(d.offsetInBytes) + static_cast<uint64_t>(d.n) * d.strides.nStride * sizeof(float) <= UINT32_MAX;
}

inline TensorView tensorView(void* const ptr, const RpptDesc& d) {
    TensorView v;
    v.data = reinterpret_cast<float*>(static_cast<Rpp8u*>(ptr) + d.offsetInBytes);
    v.dims = fk::PtrDims<fk::ND::_3D>(d.w, d.h, d.n, 3u, static_cast<uint>(d.w * sizeof(float)));
    return v;
}

inline bool sameDesc(const RpptDesc& a, const RpptDesc& b) {
    return a.numDims == b.numDims && a.offsetInBytes == b.offsetInBytes && a.dataType == b.dataType && a.n == b.n && a.c == b.c &&
           a.h == b.h && a.w == b.w && a.layout == b.layout && a.strides.nStride == b.strides.nStride &&
           a.strides.cStride == b.strides.cStride && a.strides.hStride == b.strides.hStride && a.strides.wStride == b.strides.wStride;
}

// The checks common to a source or destination descriptor; the first failure, in RPP's status codes
inline RppStatus checkDesc(const RpptDesc& d, const bool source, const RpptDataType type, const RpptLayout layout, const Rpp32u channels,
                           const size_t batch) {
    if (d.numDims != 4) {
        return source ? RPP_ERROR_INVALID_SRC_DIMS : RPP_ERROR_INVALID_DST_DIMS;
    }
    if (d.dataType != type) {
        return source ? RPP_ERROR_INVALID_SRC_DATATYPE : RPP_ERROR_INVALID_DST_DATATYPE;
    }
    if (d.layout != layout) {
        return source ? RPP_ERROR_INVALID_SRC_LAYOUT : RPP_ERROR_INVALID_DST_LAYOUT;
    }
    if (d.c != channels) {
        return source ? RPP_ERROR_INVALID_SRC_CHANNELS : RPP_ERROR_INVALID_DST_CHANNELS;
    }
    if (d.n != batch) {
        return RPP_ERROR_INVALID_ARGUMENTS;
    }
    if (d.w == 0u || d.h == 0u) {
        return RPP_ERROR_INVALID_DIM_LENGTHS;
    }
    return RPP_SUCCESS;
}

// The ROI of image i as x, y, width, height; LTRB is inclusive (RPP: right = x + width - 1)
struct Roi {
    long long x, y, width, height;
};

inline bool readRoi(const RpptROI& roi, const RpptRoiType roiType, Roi& out) {
    if (roiType == RpptRoiType::XYWH) {
        out = {roi.xywhROI.xy.x, roi.xywhROI.xy.y, roi.xywhROI.roiWidth, roi.xywhROI.roiHeight};
        return true;
    }
    if (roiType == RpptRoiType::LTRB) {
        out = {roi.ltrbROI.lt.x, roi.ltrbROI.lt.y, static_cast<long long>(roi.ltrbROI.rb.x) - roi.ltrbROI.lt.x + 1,
               static_cast<long long>(roi.ltrbROI.rb.y) - roi.ltrbROI.lt.y + 1};
        return true;
    }
    return false;
}

// ---- the per-image chain of rppt_resize_mirror_normalize

template <typename Impl>
inline auto resizeMirrorNormalizeImage(const SourceView& crop, const fk::Size& dstSize, const Rpp32u mirror, const fk::float3& mean,
                                       const fk::float3& invStdDev) {
    return fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(crop)
        .then(fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>::build(dstSize))
        .then(MirrorX<Impl>::build(MirrorParams{mirror}))
        .then(fk::Sub<fk::float3>::build(mean))
        .then(fk::Mul<fk::float3>::build(invStdDev));
}

template <typename Impl>
using ResizeMirrorNormalizeImage = decltype(resizeMirrorNormalizeImage<Impl>(std::declval<SourceView>(), std::declval<fk::Size>(), 0u,
                                                                             std::declval<fk::float3>(), std::declval<fk::float3>()));

template <typename Impl>
using PermutationOp = decltype(permutationOp<Impl>(std::declval<const Rpp32u*>()));
} // namespace detail

// ================================================== operations ====================================================

// What rppt_resize_mirror_normalize returns: one read chain per image (crop view -> HALF_PIXEL bilinear resize -> mirror
// -> (v - mean) * (1 / stdDev)), the destination, and the status of the checks. It can only start a chain.
template <size_t BATCH, typename Impl>
struct ResizeMirrorNormalize {
    static constexpr size_t BATCH_SIZE = BATCH;
    static constexpr bool STARTS_CHAIN = true;
    static constexpr bool CONTINUES_CHAIN = false;
    using Image = detail::ResizeMirrorNormalizeImage<Impl>;

    RppStatus status{RPP_SUCCESS};
    const void* srcPtr{nullptr};
    const void* dstPtr{nullptr};
    RpptDesc srcDesc{};
    RpptDesc dstDesc{};
    std::array<Image, BATCH> images{};
    detail::TensorView dstView{};

    Image image(const size_t i) const { return images[i]; }
    auto write() const { return fk::TensorSplit<fk::float3>::build(dstView); }
    // The whole operation as FKL operations, for fk::executeOperations
    auto read() const { return fk::BatchRead<fk::PlanePolicy::PROCESS_ALL>::build(images); }
};

// What rppt_channel_permute returns: the per-image permutations, the source (read when it starts a chain), the
// destination and the status of the checks.
template <size_t BATCH, typename Impl>
struct ChannelPermute {
    static constexpr size_t BATCH_SIZE = BATCH;
    static constexpr bool STARTS_CHAIN = true;
    static constexpr bool CONTINUES_CHAIN = true;
    using Permutation = detail::PermutationOp<Impl>;

    RppStatus status{RPP_SUCCESS};
    const void* srcPtr{nullptr};
    const void* dstPtr{nullptr};
    RpptDesc srcDesc{};
    RpptDesc dstDesc{};
    std::array<Permutation, BATCH> permutations{};
    detail::TensorView srcView{};
    detail::TensorView dstView{};

    auto image(const size_t i) const { return fk::TensorPack<fk::float3>::build(srcView).then(permutations[i]); }
    Permutation continuation(const size_t i) const { return permutations[i]; }
    auto write() const { return fk::TensorSplit<fk::float3>::build(dstView); }
    auto read() const {
        std::array<decltype(image(0)), BATCH> chains{};
        for (size_t i = 0; i < BATCH; ++i) {
            chains[i] = image(i);
        }
        return fk::BatchRead<fk::PlanePolicy::PROCESS_ALL>::build(chains);
    }
};

// ========================================== rppt_resize_mirror_normalize ==========================================

// RPP's rppt_resize_mirror_normalize (rppt_tensor_geometric_augmentations.h), as an operation. Implemented: BILINEAR,
// source U8 NHWC with 3 channels (packed pixels; nStride may be 0, see sharedSourceDesc), destination F32 NCHW with 3
// channels and dense strides, every dstImgSizes[i] equal to the destination's w x h. BATCH = srcDesc n = dstDesc n = the
// handle's batch size. The parameter arrays are read on the host, now, so they must be host-readable (pinned or
// pageable); they may change as soon as this returns. The ROI array is not modified.
template <size_t BATCH, RpptDataType SRC_TYPE = RpptDataType::U8, RpptLayout SRC_LAYOUT = RpptLayout::NHWC,
          RpptDataType DST_TYPE = RpptDataType::F32, RpptLayout DST_LAYOUT = RpptLayout::NCHW, Rpp32u CHANNELS = 3,
          typename Impl = DefaultImplementation>
inline ResizeMirrorNormalize<BATCH, Impl> rppt_resize_mirror_normalize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                                                       RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes,
                                                                       RpptInterpolationType interpolationType, Rpp32f* meanTensor,
                                                                       Rpp32f* stdDevTensor, Rpp32u* mirrorTensor,
                                                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                                                       rppHandle_t rppHandle, RppBackend executionBackend) {
    static_assert(BATCH > 0, "BATCH must be at least 1");
    static_assert(SRC_TYPE == RpptDataType::U8 && SRC_LAYOUT == RpptLayout::NHWC && DST_TYPE == RpptDataType::F32 &&
                      DST_LAYOUT == RpptLayout::NCHW && CHANNELS == 3,
                  "FastRPP implements rppt_resize_mirror_normalize from U8 NHWC to F32 NCHW, 3 channels");
    ResizeMirrorNormalize<BATCH, Impl> op{};
    auto fail = [&op](const RppStatus status) {
        op.status = status;
        return op;
    };
    if (executionBackend != RppBackend::RPP_HIP_BACKEND) {
        return fail(RPP_ERROR_NOT_IMPLEMENTED);
    }
    if (interpolationType != RpptInterpolationType::BILINEAR) {
        return fail(RPP_ERROR_NOT_IMPLEMENTED);
    }
    if (rppHandle == nullptr || srcPtr == nullptr || srcDescPtr == nullptr || dstPtr == nullptr || dstDescPtr == nullptr ||
        dstImgSizes == nullptr || meanTensor == nullptr || stdDevTensor == nullptr || mirrorTensor == nullptr ||
        roiTensorPtrSrc == nullptr) {
        return fail(RPP_ERROR_INVALID_ARGUMENTS);
    }
    const RpptDesc& src = *srcDescPtr;
    const RpptDesc& dst = *dstDescPtr;
    RppStatus status = detail::checkDesc(src, true, RpptDataType::U8, RpptLayout::NHWC, 3u, BATCH);
    if (status == RPP_SUCCESS) {
        status = detail::checkDesc(dst, false, RpptDataType::F32, RpptLayout::NCHW, 3u, BATCH);
    }
    if (status != RPP_SUCCESS) {
        return fail(status);
    }
    if (rppHandle->batchSize != BATCH) {
        return fail(RPP_ERROR_INVALID_ARGUMENTS);
    }
    if (src.strides.cStride != 1u || src.strides.wStride != 3u || src.strides.hStride < 3u * src.w) {
        return fail(RPP_ERROR_INVALID_SRC_LAYOUT);
    }
    if (!detail::isDenseNCHW(dst)) {
        return fail(RPP_ERROR_INVALID_DST_LAYOUT);
    }
    if (!detail::fitsFklTensor(dst)) {
        return fail(RPP_ERROR_NOT_IMPLEMENTED); // larger than 4 GiB: FKL's 32-bit tensor offsets
    }

    const fk::Size dstSize(static_cast<int>(dst.w), static_cast<int>(dst.h));
    Rpp8u* const base = static_cast<Rpp8u*>(srcPtr) + src.offsetInBytes;
    for (size_t i = 0; i < BATCH; ++i) {
        if (dstImgSizes[i].width != dst.w || dstImgSizes[i].height != dst.h) {
            return fail(RPP_ERROR_NOT_IMPLEMENTED); // per-image output sizes: FKL bounds the threads by the largest image only
        }
        detail::Roi roi{};
        if (!detail::readRoi(roiTensorPtrSrc[i], roiType, roi)) {
            return fail(RPP_ERROR_INVALID_ARGUMENTS);
        }
        if (roi.x < 0 || roi.y < 0 || roi.width < 1 || roi.height < 1 || roi.x + roi.width > static_cast<long long>(src.w) ||
            roi.y + roi.height > static_cast<long long>(src.h)) {
            return fail(RPP_ERROR_OUT_OF_BOUND_SRC_ROI);
        }
        detail::SourceView crop;
        crop.data = reinterpret_cast<fk::uchar3*>(base + i * static_cast<size_t>(src.strides.nStride) +
                                                  static_cast<size_t>(roi.y) * src.strides.hStride + static_cast<size_t>(roi.x) * 3u);
        crop.dims = fk::PtrDims<fk::ND::_2D>(static_cast<uint>(roi.width), static_cast<uint>(roi.height), src.strides.hStride);
        const Rpp32f* const mean = meanTensor + 3 * i;
        const Rpp32f* const stdDev = stdDevTensor + 3 * i;
        // RPP: (v - mean) * (1 / stdDev), 1 / stdDev in float; stdDev is not checked (0 gives inf, as in RPP)
        op.images[i] = detail::resizeMirrorNormalizeImage<Impl>(
            crop, dstSize, mirrorTensor[i] == 1u ? 1u : 0u, fk::float3{mean[0], mean[1], mean[2]},
            fk::float3{1.0f / stdDev[0], 1.0f / stdDev[1], 1.0f / stdDev[2]});
    }
    op.srcPtr = srcPtr;
    op.dstPtr = dstPtr;
    op.srcDesc = src;
    op.dstDesc = dst;
    op.dstView = detail::tensorView(dstPtr, dst);
    return op;
}

// ============================================== rppt_channel_permute ==============================================

// RPP's rppt_channel_permute (rppt_tensor_data_exchange_operations.h), as an operation: output channel k of image i is
// input channel permutationTensor[3 * i + k], as in RPP. Implemented: F32 NCHW to F32 NCHW, 3 channels, dense strides,
// the same w x h, at most 4 GiB. BATCH = srcDesc n = dstDesc n = the handle's batch size. Every triple must be one of the
// six permutations of 0, 1, 2, the function RPP documents; RPP checks nothing (a repeated index copies a channel there,
// an index above 2 is undefined). The permutations are read on the host, now.
template <size_t BATCH, RpptDataType TYPE = RpptDataType::F32, RpptLayout SRC_LAYOUT = RpptLayout::NCHW,
          RpptLayout DST_LAYOUT = RpptLayout::NCHW, Rpp32u CHANNELS = 3, typename Impl = DefaultImplementation>
inline ChannelPermute<BATCH, Impl> rppt_channel_permute(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr,
                                                        Rpp32u* permutationTensor, rppHandle_t rppHandle,
                                                        RppBackend executionBackend) {
    static_assert(BATCH > 0, "BATCH must be at least 1");
    static_assert(TYPE == RpptDataType::F32 && SRC_LAYOUT == RpptLayout::NCHW && DST_LAYOUT == RpptLayout::NCHW && CHANNELS == 3,
                  "FastRPP implements rppt_channel_permute from F32 NCHW to F32 NCHW, 3 channels");
    ChannelPermute<BATCH, Impl> op{};
    auto fail = [&op](const RppStatus status) {
        op.status = status;
        return op;
    };
    if (executionBackend != RppBackend::RPP_HIP_BACKEND) {
        return fail(RPP_ERROR_NOT_IMPLEMENTED);
    }
    if (rppHandle == nullptr || srcPtr == nullptr || srcDescPtr == nullptr || dstPtr == nullptr || dstDescPtr == nullptr ||
        permutationTensor == nullptr) {
        return fail(RPP_ERROR_INVALID_ARGUMENTS);
    }
    const RpptDesc& src = *srcDescPtr;
    const RpptDesc& dst = *dstDescPtr;
    RppStatus status = detail::checkDesc(src, true, RpptDataType::F32, RpptLayout::NCHW, 3u, BATCH);
    if (status == RPP_SUCCESS) {
        status = detail::checkDesc(dst, false, RpptDataType::F32, RpptLayout::NCHW, 3u, BATCH);
    }
    if (status != RPP_SUCCESS) {
        return fail(status);
    }
    if (rppHandle->batchSize != BATCH) {
        return fail(RPP_ERROR_INVALID_ARGUMENTS);
    }
    if (src.w != dst.w || src.h != dst.h) {
        return fail(RPP_ERROR_INVALID_DIM_LENGTHS);
    }
    if (!detail::isDenseNCHW(src)) {
        return fail(RPP_ERROR_INVALID_SRC_LAYOUT);
    }
    if (!detail::isDenseNCHW(dst)) {
        return fail(RPP_ERROR_INVALID_DST_LAYOUT);
    }
    if (!detail::fitsFklTensor(src) || !detail::fitsFklTensor(dst)) {
        return fail(RPP_ERROR_NOT_IMPLEMENTED); // larger than 4 GiB: FKL's 32-bit tensor offsets
    }
    for (size_t i = 0; i < BATCH; ++i) {
        const Rpp32u* const p = permutationTensor + 3 * i;
        if (detail::permutationCase(p[0], p[1], p[2]) < 0) {
            return fail(RPP_ERROR_INVALID_ARGUMENTS);
        }
        op.permutations[i] = detail::permutationOp<Impl>(p);
    }
    op.srcPtr = srcPtr;
    op.dstPtr = dstPtr;
    op.srcDesc = src;
    op.dstDesc = dst;
    op.srcView = detail::tensorView(srcPtr, src);
    op.dstView = detail::tensorView(dstPtr, dst);
    return op;
}

// ================================================== execution =====================================================

namespace detail {
template <typename Chain>
inline auto continueImage(const Chain& chain, size_t) {
    return chain;
}

template <typename Chain, typename Next, typename... Rest>
inline auto continueImage(const Chain& chain, const size_t i, const Next& next, const Rest&... rest) {
    return continueImage(chain.then(next.continuation(i)), i, rest...);
}

// Image i of a chain: the first operation's read chain, then the continuations of the following operations
template <typename First, typename... Rest>
inline auto composeImage(const size_t i, const First& first, const Rest&... rest) {
    return continueImage(first.image(i), i, rest...);
}

template <typename... Ops>
inline const auto& lastOf(const Ops&... ops) {
    return std::get<sizeof...(Ops) - 1>(std::tie(ops...));
}

// The status of a chain: the first operation that failed its checks, then the links (the source and descriptor of each
// operation must be the destination and descriptor of the previous one)
template <typename... Ops>
inline RppStatus chainStatus(const Ops&... ops) {
    RppStatus status = RPP_SUCCESS;
    bool first = true;
    const void* previousDst = nullptr;
    const RpptDesc* previousDesc = nullptr;
    auto check = [&](const auto& op) {
        if (status != RPP_SUCCESS) {
            return;
        }
        if (op.status != RPP_SUCCESS) {
            status = op.status;
            return;
        }
        if (!first && (op.srcPtr != previousDst || !sameDesc(op.srcDesc, *previousDesc))) {
            status = RPP_ERROR_INVALID_ARGUMENTS;
            return;
        }
        first = false;
        previousDst = op.dstPtr;
        previousDesc = &op.dstDesc;
    };
    (check(ops), ...);
    return status;
}

// A chain as the two FKL operations of its kernel: the batch of per-image read chains, and the last destination
template <typename First, typename... Rest>
inline auto fuse(const First& first, const Rest&... rest) {
    using Chain = decltype(composeImage(size_t{0}, first, rest...));
    std::array<Chain, First::BATCH_SIZE> chains{};
    for (size_t i = 0; i < First::BATCH_SIZE; ++i) {
        chains[i] = composeImage(i, first, rest...);
    }
    return std::make_pair(fk::BatchRead<fk::PlanePolicy::PROCESS_ALL>::build(chains), lastOf(first, rest...).write());
}
} // namespace detail

// Runs a chain of FastRPP operations as one kernel on the stream of the handle given here: the first operation's source,
// every operation's computation, the last operation's destination; the intermediate destinations are not written. It
// does not synchronise the stream. It returns the first failed check of an operation, RPP_ERROR_INVALID_ARGUMENTS if the
// operations do not form a chain or the handle's batch size is not BATCH (nothing is launched then), or
// RPP_ERROR_HIP_LAUNCH if a HIP error is pending after the launch (FKL checks hipGetLastError, so the error may come from
// earlier work on the device; the kernel may have been enqueued).
template <typename First, typename... Rest>
inline RppStatus executeOperations(rppHandle_t rppHandle, const First& first, const Rest&... rest) {
    static_assert(First::STARTS_CHAIN, "this operation cannot start a chain");
    static_assert((true && ... && Rest::CONTINUES_CHAIN),
                  "only rppt_channel_permute can follow another operation: rppt_resize_mirror_normalize must come first");
    static_assert((true && ... && (Rest::BATCH_SIZE == First::BATCH_SIZE)), "every operation of a chain has the same BATCH");
    constexpr size_t BATCH = First::BATCH_SIZE;
    if (rppHandle == nullptr) {
        return RPP_ERROR_INVALID_ARGUMENTS;
    }
    const RppStatus status = detail::chainStatus(first, rest...);
    if (status != RPP_SUCCESS) {
        return status;
    }
    if (rppHandle->batchSize != BATCH) {
        return RPP_ERROR_INVALID_ARGUMENTS;
    }
    const auto fused = detail::fuse(first, rest...);
    try {
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(rppHandle->stream, fused.first, fused.second);
    } catch (const std::exception&) {
        return RPP_ERROR_HIP_LAUNCH;
    }
    return RPP_SUCCESS;
}

} // namespace fastrpp

#endif // FAST_RPP_H
