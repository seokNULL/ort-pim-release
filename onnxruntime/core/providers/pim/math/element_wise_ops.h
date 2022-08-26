// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/pim/helper/pim_interface.h"

// #include "core/providers/cpu/math/element_wise_ops.h"
// #include "core/util/math_cpuonly.h"
// #include <unsupported/Eigen/SpecialFunctions>

#include "core/providers/pim/helper/aten/convert_numeric.h"

namespace onnxruntime {
namespace pim {
  
template <typename T>
class Add final : public OpKernel {
 public:
  Add(const OpKernelInfo& info) : OpKernel(info) {
    pim_args = new PIMInterface();
  }

  Status Compute(OpKernelContext* context) const override;
  PIMInterface* pim_args;
};

template <typename T>
class Mul final : public OpKernel {
 public:
  Mul(const OpKernelInfo& info) : OpKernel(info) {
    pim_args = new PIMInterface();
  }

  Status Compute(OpKernelContext* context) const override;
  PIMInterface* pim_args;
};

template <typename T>
class Sub final : public OpKernel {
 public:
  Sub(const OpKernelInfo& info) : OpKernel(info) {
    pim_args = new PIMInterface();
  }

  Status Compute(OpKernelContext* context) const override;
  PIMInterface* pim_args;
};


}  // namespace pim
}  // namespace onnxruntime
