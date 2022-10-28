// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// Lotus framework headers for onnxruntime::IExecutionProvider (not part of the operator ABI).
#include "core/common/logging/logging.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/graph_transformer.h"

#include "core/providers/pim/pim_execution_provider.h"
// #include "core/providers/pim/PIMGraphTransformer_helper.h"

namespace onnxruntime
{

    // Applies transforms to a Lotus graph. The graph transformer is responsible for setting the execution provider
    // on the graph nodes which DML supports.
    class PIMGraphTransformer
    {
    public:
     PIMGraphTransformer(
         const std::string& name,
         const onnxruntime::IExecutionProvider* provider)
         : name_(name),
           m_provider((PIMExecutionProvider*)(provider)){

    }
    ~PIMGraphTransformer() {}
    bool ShouldOnlyApplyOnce() { return false; };
    Status Apply(Graph& graph, bool& modified, const logging::Logger& logger) const;

   private:
    void PerformOperatorFusion(onnxruntime::Graph* graph, bool* modified) const;

    Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level, const logging::Logger&) const;
    const std::string name_;
    // std::shared_ptr<onnxruntime::KernelRegistry> m_registry;
    PIMExecutionProvider* m_provider = nullptr;
    // uint32_t m_supportedDataTypeMask = 0;
    // const ExecutionProviderImpl* m_providerImpl = nullptr;
    };

} // namespace pim