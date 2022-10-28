// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_registry.h"
#include "core/graph/graph_utils.h"
#include "core/providers/pim/pim_graph_transformer.h"
#include "core/providers/pim/pim_graph_transformer_helper.h"

#include <iostream>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime {

static std::string GetUniqueNodeName(const onnxruntime::Node* node) {
  std::stringstream ss;
  ss << '#' << node->Index();
  if (!node->Name().empty()) {
    ss << " \'" << node->Name() << '\'';
  }
  return ss.str();
}

void PIMGraphTransformer::PerformOperatorFusion(onnxruntime::Graph* graph, bool* modified) const {
  onnxruntime::KernelRegistry* registry = m_provider->GetKernelRegistry().get();

  struct NodeToAdd {
    std::string name;
    std::string description;
    std::string opType;
    onnxruntime::NodeAttributes attributes;
    std::string predecessorOpType;
    onnxruntime::NodeAttributes successorAttributes;
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;
  };

  // Defer adding new nodes to the graph until after we're done iterating over it, because we can't mutate the
  // graph while iterating over it
  // std::vector<NodeToAdd> nodesToAdd;

  for (auto& node : graph->Nodes()) {
    // std::cout << node.OpType() << std::endl;
    // We need to predict whether the nodes will be assigned to the DML transformer by Lotus,
    // which occurs in IExecutionProvider::GetCapability.

    // bool allow64BitInputThroughStrides = false;
    if (!(m_provider->IsNodeSupportedByPim(
            node,
            *registry))) {
      // Can't fuse nodes that don't belong to this execution provider
      continue;
    }

    // The number of nodes which use the result of this convolution as input
    const auto outputNodeCount = std::distance(node.OutputEdgesBegin(), node.OutputEdgesEnd());

    if (outputNodeCount != 1) {
      // Can only fuse nodes whose only output feeds into a single activation - if multiple nodes use the
      // output of this one, we can't fuse it.
      continue;
    }

    const auto& outputNode = *node.OutputNodesBegin();

    // We need to predict whether the nodes will be assigned to the DML transformer by Lotus,
    // which occurs in IExecutionProvider::GetCapability.
    if (!onnxruntime::KernelRegistry::HasImplementationOf(*registry, outputNode, onnxruntime::kPimExecutionProvider)) {
      // Can't fuse nodes that don't belong to this execution provider
      continue;
    }

    // if (outputNode.InputDefs().size() != 1) {
    //   // Can only fuse activation functions that take a single input
    //   continue;
    // }

    auto fusedOpProperties = pim::FusionHelpers::TryGetFusedOp(
        node.OpType(),
        gsl::narrow_cast<uint32_t>(node.InputDefs().size()),
        outputNode.OpType());

    if (fusedOpProperties.opType == "NULL") {
      // These operators can't be fused
      continue;
    }
    bool check_scalar = false;
    if (node.OpType() == "MatMul") {
      // Check 3rd input dimension of successor node
      check_scalar = true;
    }
    //std::cout <<" OPTYPE: " << node.OpType() << std::endl;

    auto& predecessorNode = *graph->GetNode(node.Index());
    auto& successorNode = *graph->GetNode(outputNode.Index());
    std::string nodeName = "fused op (" + GetUniqueNodeName(&predecessorNode) + ") + (" + GetUniqueNodeName(&successorNode) + ")";
    std::vector<NodeArg*> fused_input;
    std::vector<NodeArg*> fused_output;

    for (size_t i = 0; i < predecessorNode.MutableInputDefs().size(); i++) {
      fused_input.push_back(predecessorNode.MutableInputDefs()[i]);
    }
    for (size_t i = 0; i < successorNode.MutableInputDefs().size(); i++) {
      // if the input of the successor is the output of the predecessor, do not add.
      if (std::find(predecessorNode.OutputDefs().begin(), predecessorNode.OutputDefs().end(),
                    successorNode.MutableInputDefs()[i]) != predecessorNode.OutputDefs().end())
        continue;
      fused_input.push_back(successorNode.MutableInputDefs()[i]);
    }

    for (size_t i = 0; i < successorNode.MutableOutputDefs().size(); i++) {
      fused_output.push_back(successorNode.MutableOutputDefs()[i]);
    }
    //if (check_scalar) {
      for (size_t i = 0; i < fused_input.size(); i++) {
        //printf("    DIM:%d - ", fused_input[i]->Shape()->dim_size());
        for (size_t j = 0; j < fused_input[i]->Shape()->dim_size(); j++) {
          //printf("%d ", fused_input[i]->Shape()->dim()[j].dim_value());
        }
        //printf("\n");
      }
    //}      
    Node& fusedNode = graph->AddNode(graph->GenerateNodeName(nodeName),
                                     fusedOpProperties.opType,
                                     "PIM fusion",
                                     fused_input,
                                     fused_output,
                                     {},
                                     "");
    // fusedNode.SetExecutionProviderType(predecessorNode.GetExecutionProviderType());
    graph_utils::FinalizeNodeFusion(*graph, {predecessorNode, successorNode}, fusedNode);

    * modified = true;
  }

}

Status PIMGraphTransformer::Apply(Graph& graph, bool& modified, const logging::Logger& logger) const {
  // the Graph should be in a good state prior this being called, so there should be no need to call Resolve here
  // ORT_RETURN_IF_ERROR(graph.Resolve());
#if 0
  for (auto& node : graph.Nodes())
    std::cout << node.OpType() << std::endl;
  std::cout << "------------" << std::endl;
#endif

#if !defined(ORT_MINIMAL_BUILD)
  auto status = ApplyImpl(graph, modified, 0, logger);
  ORT_RETURN_IF_ERROR(status);

  // At least currently, some transformers (InsertCastTransformer and MemcpyTransformer) need this to be called
  // after they complete to put the graph back into a valid state for the next transformer.
  if (modified) {
    status = graph.Resolve();
  }
#else
  ORT_UNUSED_PARAMETER(graph);
  ORT_UNUSED_PARAMETER(modified);
  ORT_UNUSED_PARAMETER(logger);
  Status status(ONNXRUNTIME, FAIL, "Transformers are not supported in this build");
#endif
  return status;
}

Status PIMGraphTransformer::ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level, const logging::Logger&) const {
  // int graph_level, const logging::Logger&)  {
  modified = false;

  // Perform fusion
  {
    bool transformModifiedGraph = false;
    PerformOperatorFusion(&graph, &transformModifiedGraph);
    modified |= transformModifiedGraph;

    if (modified) {
      ORT_RETURN_IF_ERROR(graph.Resolve());
    }
  }

  return Status::OK();
}

}  // namespace pim
