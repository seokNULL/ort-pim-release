// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #include "precomp.h"
#include "core/providers/pim/pim_graph_transformer_helper.h"
// #include "core/providers/dml/OperatorAuthorHelper/OperatorRegistration.h"
namespace onnxruntime {
namespace pim {

namespace FusionHelpers {
struct OperatorInfo {
  std::string type;
  std::vector<std::string> successorFilter;
};

static bool operator==(const OperatorInfo& lhs, const OperatorInfo& rhs) {
  return (lhs.type == rhs.type);
}

static const OperatorInfo c_predecessorOps[] =
{
  //OperatorInfo{"MatMul", {"Add"}},
  OperatorInfo{"MatMul", {"Add", "Sub"}},
  //OperatorInfo{"Add", {"Add", "Sub"}},
  //OperatorInfo{"Mul", {"Add", "Sub"}},
  //OperatorInfo{"Sub", {"Add", "Sub"}},
};

FusedOpProperties TryGetFusedOp(std::string candidateOpType, int candidateOpInputCount, std::string successorOpType) {
  auto opIt = std::find(
      std::begin(c_predecessorOps),
      std::end(c_predecessorOps),
      OperatorInfo{candidateOpType, {}});
  if (opIt == std::end(c_predecessorOps)) {
    return FusedOpProperties{std::move("NULL")};
    // return NULL;
  }

  if (!opIt->successorFilter.empty() &&
      std::find(opIt->successorFilter.begin(), opIt->successorFilter.end(), successorOpType) == opIt->successorFilter.end()) {
    return FusedOpProperties{std::move("NULL")};
  }

  // All fused ops just have "Fused" prepended to their name (e.g. "Matmul" + "Add" -> "FusedMatmulAdd").
  std::string fusedOpType = std::string("Fused").append(candidateOpType).append(successorOpType);

  return FusedOpProperties{std::move(fusedOpType)};
}

/*static*/ std::string GetFusedAttributeName(std::string name) {
  return std::string("fused_").append(name);
}

}  // namespace FusionHelpers
}} // namespace onnxruntime::pim
