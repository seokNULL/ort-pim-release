// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_transfer_manager.h"
#include "memcpy.h"
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

Memcpy::Memcpy(const OpKernelInfo& info)
    : OpKernel(info) {
}

// Original Code
// Status Memcpy::Compute(OpKernelContext* ctx) const {
//   const auto* X = ctx->Input<Tensor>(0);
//   Tensor* Y = ctx->Output(0, X->Shape());
//   Status retval = Info().GetDataTransferManager().CopyTensor(*X, *Y, Info().GetKernelDef().ExecQueueId());

//   if (!retval.IsOK()) {
//     LOGS(ctx->Logger(), ERROR) << MakeString(retval.ErrorMessage(),
//                                              " Copying ", Node().InputDefs()[0]->Name(),
//                                              " to ", Node().OutputDefs()[0]->Name(),
//                                              " Input shape:", X->Shape(), " Output shape:", Y->Shape(),
//                                              " X data:", X->DataRaw(), " Y data:", Y->DataRaw());
//   }

//   return retval;
// }

// Status Memcpy::Compute(OpKernelContext* ctx) const {
//   const auto* X = ctx->Input<Tensor>(0);
//   Tensor* Y = ctx->Output(0, X->Shape());
//   Status retval = Info().GetDataTransferManager().CopyTensor(*X, *Y, *ctx, Info().GetKernelDef().ExecQueueId());

//   if (!retval.IsOK()) {
//     LOGS(ctx->Logger(), ERROR) << MakeString(retval.ErrorMessage(),
//                                              " Copying ", Node().InputDefs()[0]->Name(),
//                                              " to ", Node().OutputDefs()[0]->Name(),
//                                              " Input shape:", X->Shape(), " Output shape:", Y->Shape(),
//                                              " X data:", X->DataRaw(), " Y data:", Y->DataRaw());
//   }

//   return retval;
// }

Status Memcpy::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  Tensor* Y;
  Status retval;

  // 1. memcpy from cpu to pim
  if (ctx->memcpy_cpu_to_pim) {
    std::vector<int64_t> tensor_shape = X->Shape().GetDims();
    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Y = ctx->Output(0, tensor_shape, true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
    ctx->tensor_alloc_dur = tensor_alloc_dur; 
    Y->SetIsPim();   
    Y->SetOriginalShape(X->Shape());
    retval = Info().GetDataTransferManager().CopyTensor(*X, *Y, *ctx, Info().GetKernelDef().ExecQueueId());

  }  // 1
  // 2. memcpy from pim to cpu 
  else if (ctx->memcpy_pim_to_cpu) {
    std::vector<int64_t> tensor_shape = X->Shape().GetDims();

    ORT_ENFORCE(!X->GetLayoutChanged());
    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Y = ctx->Output(0, X->Shape());
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
    ctx->tensor_alloc_dur = tensor_alloc_dur; 

    retval = Info().GetDataTransferManager().CopyTensor(*X, *Y, *ctx, Info().GetKernelDef().ExecQueueId());
  } // 2
  // 3. else
  else {
    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Y = ctx->Output(0, X->Shape());
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
    ctx->tensor_alloc_dur = tensor_alloc_dur; 
    retval = Info().GetDataTransferManager().CopyTensor(*X, *Y, Info().GetKernelDef().ExecQueueId());
  } // 3

  if (!retval.IsOK()) {
    LOGS(ctx->Logger(), ERROR) << MakeString(retval.ErrorMessage(),
                                             " Copying ", Node().InputDefs()[0]->Name(),
                                             " to ", Node().OutputDefs()[0]->Name(),
                                             " Input shape:", X->Shape(), " Output shape:", Y->Shape(),
                                             " X data:", X->DataRaw(), " Y data:", Y->DataRaw());
    
  }
  return retval;  
}

}  // namespace onnxruntime
