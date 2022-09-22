// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the Korea Univ, COMMIT Lab License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/pim/helper/pim_interface.h"
#include "core/providers/pim/helper/aten/convert_numeric.h"


namespace onnxruntime {
namespace pim {

template <typename T>
class Erf final : public OpKernel {
 public:
  Erf(const OpKernelInfo& info) : OpKernel(info) {
      pim_args = new PIMInterface();
  }

  Status Compute(OpKernelContext* context) const override;
  PIMInterface* pim_args;
};

template <typename T>
class Tanh final : public OpKernel {
 public:
  Tanh(const OpKernelInfo& info) : OpKernel(info) {
      pim_args = new PIMInterface();
  }

  Status Compute(OpKernelContext* context) const override;
  PIMInterface* pim_args;
};

}  // namespace pim
}  // namespace onnxruntime
