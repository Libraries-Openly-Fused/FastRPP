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

// FastRPP: functions of AMD's RPP tensor API (rppt_*, RPP 3.2.0) as operations of the Fused Kernel Library (FKL), HIP
// only, in the way FastNPP offers NPP's functions.
//
// Each fastrpp::rppt_* function takes RPP's arguments, checks them on the host, translates RPP's objects (descriptors,
// ROIs, parameter arrays) into FKL's (views, sizes, vectors) and returns ONE FKL instantiable operation (IOp): the FKL
// operations the function needs, joined with .then. FastRPP defines no operation of its own, and the only ReadBack
// operation it uses is FKL's resize.
//   rppt_resize_mirror_normalize  a read IOp: the crops of the source, resized and normalised
//   rppt_channel_permute          a compute IOp: fk::VectorReorderRT (FastNPP's SwapChannels)
//   rppt_copy                     a write IOp: the destination tensor (FastNPP's CopyBatch)
// fastrpp::executeOperations(handle, iOps...) runs IOps as one kernel on the handle's stream, as fk::executeOperations
// does: a read, any number of compute IOps, a write. Only a read IOp reads memory and only a write IOp writes it: the
// other source and destination pointers keep RPP's signatures and are never accessed, so in
// executeOperations(handle, rmn, cp, copy) no intermediate tensor is read or written.
//
// A failed check throws fastrpp::Error, which carries RPP's status. RPP's types are used as they are (rppdefs.h):
// FastRPP needs RPP's headers, never librpp. The handle is FastRPP's own. Semantics and every difference from RPP 3.2.0:
// README.md.

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
#include <stdexcept>
#include <string>
#include <utility>

namespace fastrpp {

// ===================================================== errors =====================================================

// What a failed check throws: the RppStatus of the failure (RPP's own status for it, or RPP_ERROR_NOT_IMPLEMENTED for
// what FastRPP does not implement) and a message that names the function and the check
class Error : public std::runtime_error {
public:
    Error(const RppStatus status, const std::string& message) : std::runtime_error(message), status_(status) {}
    RppStatus status() const noexcept { return status_; }

private:
    RppStatus status_;
};

// ===================================================== handle =====================================================

// FastRPP's handle: the batch size and the stream. FKL needs no scratch memory, so it holds no device memory. It is a
// different type from RPP's rppHandle_t: an RPP handle cannot be passed to FastRPP by mistake.
struct rppHandle {
    size_t batchSize;
    fk::Stream_<fk::ParArch::GPU_AMD> stream; // wraps the hipStream_t given to rppCreate, which it never destroys
};
typedef rppHandle* rppHandle_t;

// RPP's rppCreate. Only RppBackend::RPP_HIP_BACKEND is implemented; numThreads is ignored (RPP uses it only for the host
// back-end); stream nullptr is the null stream. It returns a status, as RPP's does: it builds no operation.
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

// ============================================ translation and checks ==============================================

namespace detail {

[[noreturn]] inline void fail(const RppStatus status, const char* const function, const std::string& what) {
    throw Error(status, std::string("fastrpp::") + function + ": " + what);
}

inline const char* side(const bool source) { return source ? "the source" : "the destination"; }

// The FKL views of RPP's tensors: a crop of a U8 NHWC image, and a whole F32 NCHW tensor
using SourceView = fk::RawPtr<fk::ND::_2D, fk::uchar3>;
using TensorView = fk::RawPtr<fk::ND::_3D, float>;

// A dense NCHW float tensor of n images, the strides RpptDesc's constructor gives: FKL's TensorSplit puts channel k of
// a pixel k * w * h elements after channel 0, and image z at z * 3 planes
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

// The checks of every descriptor, in this order; n must be the batch size
inline void checkDesc(const RpptDesc& d, const bool source, const RpptDataType type, const RpptLayout layout, const Rpp32u channels,
                      const size_t batch, const char* const function) {
    if (d.numDims != 4) {
        fail(source ? RPP_ERROR_INVALID_SRC_DIMS : RPP_ERROR_INVALID_DST_DIMS, function, std::string(side(source)) + " numDims is not 4");
    }
    if (d.dataType != type) {
        fail(source ? RPP_ERROR_INVALID_SRC_DATATYPE : RPP_ERROR_INVALID_DST_DATATYPE, function,
             std::string(side(source)) + " data type is not the template argument's");
    }
    if (d.layout != layout) {
        fail(source ? RPP_ERROR_INVALID_SRC_LAYOUT : RPP_ERROR_INVALID_DST_LAYOUT, function,
             std::string(side(source)) + " layout is not the template argument's");
    }
    if (d.c != channels) {
        fail(source ? RPP_ERROR_INVALID_SRC_CHANNELS : RPP_ERROR_INVALID_DST_CHANNELS, function,
             std::string(side(source)) + " channel count is not the template argument's");
    }
    if (d.n != batch) {
        fail(RPP_ERROR_INVALID_ARGUMENTS, function, std::string(side(source)) + " n is not the batch size");
    }
    if (d.w == 0u || d.h == 0u) {
        fail(RPP_ERROR_INVALID_DIM_LENGTHS, function, std::string(side(source)) + " width or height is 0");
    }
}

// The checks of the functions on F32 NCHW tensors (rppt_channel_permute, rppt_copy): the handle, both descriptors, the
// same width and height
inline void checkTensorPair(const RpptDesc* const src, const RpptDesc* const dst, const rppHandle_t rppHandle, const char* const function) {
    if (rppHandle == nullptr || src == nullptr || dst == nullptr) {
        fail(RPP_ERROR_INVALID_ARGUMENTS, function, "a null handle or descriptor");
    }
    checkDesc(*src, true, RpptDataType::F32, RpptLayout::NCHW, 3u, rppHandle->batchSize, function);
    checkDesc(*dst, false, RpptDataType::F32, RpptLayout::NCHW, 3u, rppHandle->batchSize, function);
    if (src->w != dst->w || src->h != dst->h) {
        fail(RPP_ERROR_INVALID_DIM_LENGTHS, function, "the source and the destination have different widths or heights");
    }
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

inline bool isPermutation(const Rpp32u* const p) {
    return p[0] <= 2u && p[1] <= 2u && p[2] <= 2u && p[0] != p[1] && p[0] != p[2] && p[1] != p[2];
}

// Image i of rppt_resize_mirror_normalize, FKL operations only: its crop -> HALF_PIXEL bilinear resize -> - mean ->
// * (1 / stdDev)
inline auto resizeNormalizeImage(const SourceView& crop, const fk::Size& dstSize, const fk::float3& mean, const fk::float3& invStdDev) {
    return fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(crop)
        .then(fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>::build(dstSize))
        .then(fk::Sub<fk::float3>::build(mean))
        .then(fk::Mul<fk::float3>::build(invStdDev));
}

using ResizeNormalizeImage =
    decltype(resizeNormalizeImage(std::declval<SourceView>(), std::declval<fk::Size>(), std::declval<fk::float3>(), std::declval<fk::float3>()));

} // namespace detail

// ========================================== rppt_resize_mirror_normalize ==========================================

// RPP's rppt_resize_mirror_normalize (rppt_tensor_geometric_augmentations.h) as one FKL read IOp: an
// fk::BatchRead<PlanePolicy::PROCESS_ALL> of BATCH chains, image i being
//   fk::PerThreadRead(crop i).then(fk::ResizeHalfPixel(w x h)).then(fk::Sub(mean i)).then(fk::Mul(1 / stdDev i)).
// Implemented: BILINEAR; source U8 NHWC, 3 channels, packed pixels (nStride may be 0, see sharedSourceDesc);
// destination F32 NCHW, 3 channels; every dstImgSizes[i] equal to the destination's w x h; no image mirrored
// (mirrorTensor[i] == 1 throws RPP_ERROR_NOT_IMPLEMENTED: FKL has no mirror but through a ReadBack operation other than
// the resize; any other value is no mirror, as in RPP). BATCH = srcDesc n = dstDesc n = the handle's batch size.
// dstPtr is not accessed: the destination descriptor gives the output size, and the values go to the next IOp (a write,
// such as rppt_copy, puts them in memory). The parameter arrays are read on the host, now, and their values are copied
// into the IOp: they must be host-readable, and they may change as soon as this returns. The ROI array is not modified.
template <size_t BATCH, RpptDataType SRC_TYPE = RpptDataType::U8, RpptLayout SRC_LAYOUT = RpptLayout::NHWC,
          RpptDataType DST_TYPE = RpptDataType::F32, RpptLayout DST_LAYOUT = RpptLayout::NCHW, Rpp32u CHANNELS = 3>
inline auto rppt_resize_mirror_normalize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr,
                                         RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, Rpp32f* meanTensor,
                                         Rpp32f* stdDevTensor, Rpp32u* mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                         rppHandle_t rppHandle, RppBackend executionBackend) {
    static_assert(BATCH > 0, "BATCH must be at least 1");
    static_assert(SRC_TYPE == RpptDataType::U8 && SRC_LAYOUT == RpptLayout::NHWC && DST_TYPE == RpptDataType::F32 &&
                      DST_LAYOUT == RpptLayout::NCHW && CHANNELS == 3,
                  "FastRPP implements rppt_resize_mirror_normalize from U8 NHWC to F32 NCHW, 3 channels");
    constexpr const char* function = "rppt_resize_mirror_normalize";
    static_cast<void>(dstPtr); // not accessed: the IOp's values go to the next IOp
    if (executionBackend != RppBackend::RPP_HIP_BACKEND) {
        detail::fail(RPP_ERROR_NOT_IMPLEMENTED, function, "only RPP_HIP_BACKEND is implemented");
    }
    if (interpolationType != RpptInterpolationType::BILINEAR) {
        detail::fail(RPP_ERROR_NOT_IMPLEMENTED, function, "only BILINEAR interpolation is implemented");
    }
    if (rppHandle == nullptr || srcPtr == nullptr || srcDescPtr == nullptr || dstDescPtr == nullptr || dstImgSizes == nullptr ||
        meanTensor == nullptr || stdDevTensor == nullptr || mirrorTensor == nullptr || roiTensorPtrSrc == nullptr) {
        detail::fail(RPP_ERROR_INVALID_ARGUMENTS, function, "a null handle, source, descriptor or parameter array");
    }
    const RpptDesc& src = *srcDescPtr;
    const RpptDesc& dst = *dstDescPtr;
    detail::checkDesc(src, true, RpptDataType::U8, RpptLayout::NHWC, 3u, BATCH, function);
    detail::checkDesc(dst, false, RpptDataType::F32, RpptLayout::NCHW, 3u, BATCH, function);
    if (rppHandle->batchSize != BATCH) {
        detail::fail(RPP_ERROR_INVALID_ARGUMENTS, function, "the handle's batch size is not BATCH");
    }
    if (src.strides.cStride != 1u || src.strides.wStride != 3u || src.strides.hStride < 3u * src.w) {
        detail::fail(RPP_ERROR_INVALID_SRC_LAYOUT, function, "the source pixels are not packed (cStride 1, wStride 3, hStride >= 3 * w)");
    }

    const fk::Size dstSize(static_cast<int>(dst.w), static_cast<int>(dst.h));
    Rpp8u* const base = static_cast<Rpp8u*>(srcPtr) + src.offsetInBytes;
    std::array<detail::ResizeNormalizeImage, BATCH> images{};
    for (size_t i = 0; i < BATCH; ++i) {
        if (dstImgSizes[i].width != dst.w || dstImgSizes[i].height != dst.h) {
            // FKL bounds the threads of a batch by its largest image only
            detail::fail(RPP_ERROR_NOT_IMPLEMENTED, function, "per-image output sizes: every dstImgSizes[i] must be the destination's w x h");
        }
        if (mirrorTensor[i] == 1u) {
            detail::fail(RPP_ERROR_NOT_IMPLEMENTED, function,
                         "mirror (mirrorTensor[i] == 1): FKL has no mirror but through a ReadBack operation other than the resize");
        }
        detail::Roi roi{};
        if (!detail::readRoi(roiTensorPtrSrc[i], roiType, roi)) {
            detail::fail(RPP_ERROR_INVALID_ARGUMENTS, function, "the ROI type is neither XYWH nor LTRB");
        }
        if (roi.x < 0 || roi.y < 0 || roi.width < 1 || roi.height < 1 || roi.x + roi.width > static_cast<long long>(src.w) ||
            roi.y + roi.height > static_cast<long long>(src.h)) {
            detail::fail(RPP_ERROR_OUT_OF_BOUND_SRC_ROI, function, "a ROI is not inside its image, or its width or height is below 1");
        }
        detail::SourceView crop;
        crop.data = reinterpret_cast<fk::uchar3*>(base + i * static_cast<size_t>(src.strides.nStride) +
                                                  static_cast<size_t>(roi.y) * src.strides.hStride + static_cast<size_t>(roi.x) * 3u);
        crop.dims = fk::PtrDims<fk::ND::_2D>(static_cast<uint>(roi.width), static_cast<uint>(roi.height), src.strides.hStride);
        const Rpp32f* const mean = meanTensor + 3 * i;
        const Rpp32f* const stdDev = stdDevTensor + 3 * i;
        // RPP: (v - mean) * (1 / stdDev), 1 / stdDev in float; stdDev is not checked (0 gives inf, as in RPP)
        images[i] = detail::resizeNormalizeImage(crop, dstSize, fk::float3{mean[0], mean[1], mean[2]},
                                                 fk::float3{1.0f / stdDev[0], 1.0f / stdDev[1], 1.0f / stdDev[2]});
    }
    return fk::BatchRead<fk::PlanePolicy::PROCESS_ALL>::build(images);
}

// ============================================== rppt_channel_permute ==============================================

// RPP's rppt_channel_permute (rppt_tensor_data_exchange_operations.h) as one FKL compute IOp, fk::VectorReorderRT (as
// FastNPP's SwapChannels): output channel k is input channel permutationTensor[k], as in RPP. Implemented: F32 NCHW to
// F32 NCHW, 3 channels, the same width and height; n = the handle's batch size. FKL's compute operations have the same
// parameters for every image, so every image must have the same triple (RPP_ERROR_NOT_IMPLEMENTED otherwise), and it
// must be one of the six permutations of 0, 1, 2, the function RPP documents (RPP_ERROR_INVALID_ARGUMENTS otherwise;
// RPP checks nothing). srcPtr and dstPtr are not accessed: the IOp takes its input from the IOp before it and gives its
// output to the next one. The permutations are read on the host, now.
template <RpptDataType TYPE = RpptDataType::F32, RpptLayout SRC_LAYOUT = RpptLayout::NCHW, RpptLayout DST_LAYOUT = RpptLayout::NCHW,
          Rpp32u CHANNELS = 3>
inline auto rppt_channel_permute(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u* permutationTensor,
                                 rppHandle_t rppHandle, RppBackend executionBackend) {
    static_assert(TYPE == RpptDataType::F32 && SRC_LAYOUT == RpptLayout::NCHW && DST_LAYOUT == RpptLayout::NCHW && CHANNELS == 3,
                  "FastRPP implements rppt_channel_permute from F32 NCHW to F32 NCHW, 3 channels");
    constexpr const char* function = "rppt_channel_permute";
    static_cast<void>(srcPtr); // not accessed: the input comes from the IOp before it
    static_cast<void>(dstPtr); // not accessed: the output goes to the next IOp
    if (executionBackend != RppBackend::RPP_HIP_BACKEND) {
        detail::fail(RPP_ERROR_NOT_IMPLEMENTED, function, "only RPP_HIP_BACKEND is implemented");
    }
    if (permutationTensor == nullptr) {
        detail::fail(RPP_ERROR_INVALID_ARGUMENTS, function, "a null permutation array");
    }
    detail::checkTensorPair(srcDescPtr, dstDescPtr, rppHandle, function);
    const Rpp32u* const first = permutationTensor;
    for (size_t i = 0; i < rppHandle->batchSize; ++i) {
        const Rpp32u* const p = permutationTensor + 3 * i;
        if (!detail::isPermutation(p)) {
            detail::fail(RPP_ERROR_INVALID_ARGUMENTS, function, "a triple is not a permutation of 0, 1, 2");
        }
        if (p[0] != first[0] || p[1] != first[1] || p[2] != first[2]) {
            detail::fail(RPP_ERROR_NOT_IMPLEMENTED, function,
                         "per-image permutations: FKL's compute operations have the same parameters for every image");
        }
    }
    return fk::VectorReorderRT<fk::float3>::build(fk::int3{static_cast<int>(first[0]), static_cast<int>(first[1]), static_cast<int>(first[2])});
}

// =================================================== rppt_copy ====================================================

// RPP's rppt_copy (rppt_tensor_data_exchange_operations.h) as one FKL write IOp, fk::TensorSplit (as FastNPP's
// CopyBatch): the values of the IOps before it are written to the destination. Implemented: F32 NCHW to F32 NCHW,
// 3 channels, the same width and height; n = the handle's batch size; a dense destination (the strides RpptDesc's
// constructor gives) of at most 4 GiB. srcPtr is not accessed: the values come from the IOps before it.
template <RpptDataType TYPE = RpptDataType::F32, RpptLayout SRC_LAYOUT = RpptLayout::NCHW, RpptLayout DST_LAYOUT = RpptLayout::NCHW,
          Rpp32u CHANNELS = 3>
inline auto rppt_copy(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, rppHandle_t rppHandle,
                      RppBackend executionBackend) {
    static_assert(TYPE == RpptDataType::F32 && SRC_LAYOUT == RpptLayout::NCHW && DST_LAYOUT == RpptLayout::NCHW && CHANNELS == 3,
                  "FastRPP implements rppt_copy from F32 NCHW to F32 NCHW, 3 channels");
    constexpr const char* function = "rppt_copy";
    static_cast<void>(srcPtr); // not accessed: the values come from the IOps before it
    if (executionBackend != RppBackend::RPP_HIP_BACKEND) {
        detail::fail(RPP_ERROR_NOT_IMPLEMENTED, function, "only RPP_HIP_BACKEND is implemented");
    }
    if (dstPtr == nullptr) {
        detail::fail(RPP_ERROR_INVALID_ARGUMENTS, function, "a null destination");
    }
    detail::checkTensorPair(srcDescPtr, dstDescPtr, rppHandle, function);
    const RpptDesc& dst = *dstDescPtr;
    if (!detail::isDenseNCHW(dst)) {
        detail::fail(RPP_ERROR_INVALID_DST_LAYOUT, function, "the destination is not dense (the strides RpptDesc's constructor gives)");
    }
    if (!detail::fitsFklTensor(dst)) {
        detail::fail(RPP_ERROR_NOT_IMPLEMENTED, function, "the destination is larger than 4 GiB (FKL's 32-bit tensor offsets)");
    }
    return fk::TensorSplit<fk::float3>::build(detail::tensorView(dstPtr, dst));
}

// ================================================== execution =====================================================

// Runs FKL IOps as one kernel on the stream of the handle, as FastNPP's executeOperations: fk::executeOperations with
// fk::TransformDPP (AMD GPU, thread fusion off). Any FKL IOps can take part, FastRPP's or not; FKL checks at compile time
// that they form a read, compute IOps and a write. It does not synchronise. A null handle throws fastrpp::Error
// (RPP_ERROR_INVALID_ARGUMENTS); a HIP error pending after the launch throws FKL's std::runtime_error (FKL checks
// hipGetLastError, so the error may come from earlier work on the device).
template <typename... IOps>
inline void executeOperations(rppHandle_t rppHandle, const IOps&... iOps) {
    if (rppHandle == nullptr) {
        detail::fail(RPP_ERROR_INVALID_ARGUMENTS, "executeOperations", "a null handle");
    }
    fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(rppHandle->stream, iOps...);
}

} // namespace fastrpp

#endif // FAST_RPP_H
