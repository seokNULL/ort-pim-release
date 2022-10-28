// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/pim/pim_execution_provider.h"
namespace onnxruntime {
namespace pim{

namespace FusionHelpers {
struct FusedOpProperties {
  std::string opType;
};

// Checks whether a candidate op can be fused with the specified activation and returns information about the
// combined fused op if true, null otherwise.
FusedOpProperties TryGetFusedOp(
    std::string candidateOpType,
    int candidateOpInputCount,
    std::string activationOpType);

// Returns true if the given activation operator type supports being fused with a fusable operator, false
// otherwise.
bool IsFusableActivationOperator(std::string opType, std::string domain);

// ActivationOperatorDesc TryGetFusedActivationDesc(const MLOperatorKernelCreationContext& kernelInfo);

// Produces names for attributes added to fused kernels. This effectively prepends a string to distinguish ONNX
// attributes from those added dynamically via operator fusion. For example, this function would be used to
// produce the attribute for Activation in a fused Conv+Activation kernel.
std::string GetFusedAttributeName(std::string name);

}  // namespace FusionHelpers

}  // namespace pim
}  // namespace onnxruntime

