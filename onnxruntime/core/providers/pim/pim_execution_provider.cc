// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "pim_execution_provider.h"

#include "core/framework/compute_capability.h"
#include "core/framework/fallback_cpu_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/provider_options_utils.h"
#include "core/framework/memcpy.h"
#include "core/graph/graph_utils.h"
#include "core/providers/pim/pim_allocator.h"
#include "core/providers/pim/pim_data_transfer.h"

#include "core/framework/data_types_internal.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"

#include "core/util/math.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include "core/providers/pim/math/lut_ops.h"
// #include "core/providers/pim/math/lut_helper.h"

#include "core/providers/pim/helper/pim_init.h"

using namespace onnxruntime::common;

namespace lut_env_vars {
static const std::string kLutPath = "ORT_PIM_LUT_PATH";
}  // namespace tensorrt_env_vars

// #define MANUAL

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  Status st;
};
}  // namespace
namespace onnxruntime {

PIMExecutionProvider::PIMExecutionProvider(const PIMExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kPimExecutionProvider} {

  AllocatorCreationInfo default_memory_info(
      [](OrtDevice::DeviceId id) {
        return onnxruntime::make_unique<PIMAllocator>(id, PIM);
      },
      info.device_id,
      true,
      {info.hip_mem_limit,
       static_cast<int>(info.arena_extend_strategy),
       -1, -1});

  allocator_ = CreateAllocator(default_memory_info);
  InsertAllocator(allocator_);

  AllocatorCreationInfo cpu_memory_info(
      [](int device_id) {
        return onnxruntime::make_unique<CPUAllocator>(
            OrtMemoryInfo("CPU_PIM", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), device_id,
                          OrtMemTypeCPUOutput));
      },
      CPU_ALLOCATOR_DEVICE_ID);

  InsertAllocator(CreateAllocator(cpu_memory_info));
  SetPimDevice();
  RegisterLut();
}

PIMExecutionProvider::~PIMExecutionProvider() {
  FreePimDevice();
}

AllocatorPtr PIMExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  // Pinned memory allocator is shared between threads, but CUDA memory allocator is per-thread or it may cause result changes
  // A hypothesis is that arena allocator is not aligned with CUDA output cache, and data from different kernel writes may
  // cause cacheline to contain dirty data.
  if (mem_type == OrtMemTypeDefault) {
    return allocator_;
  } else {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
PIMExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph, const std::vector<const KernelRegistry*>& kernel_registries) const {
  // Find inputs, initializers and outputs for each supported subgraph

  std::vector<NodeIndex> candidates;

  if (!graph.is_partitioned) {
    for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
      const auto* p_node = graph.GetNode(node_index);
      if (p_node == nullptr)
        continue;

      const auto& node = *p_node;
      const KernelCreateInfo* pim_kernel_def = nullptr;
      if (!node.GetExecutionProviderType().empty()) {
        continue;
      }

      for (auto registry : kernel_registries) {
        auto st = registry->TryFindKernel(node, Type(), &pim_kernel_def);

        // at least one registry has a CUDA kernel for this node
        if (st.IsOK())
          break;
      }

      // none of the provided registries has a CUDA kernel for this node
      if (pim_kernel_def == nullptr) {
        LOGS_DEFAULT(WARNING) << "PIM kernel not found in registries for Op type: " << node.OpType() << " node name: " << node.Name();
        continue;
      }

      bool not_supported = false;

      //////////////////////////////////////
      //////// Supported Operation
      /////////////////////////////////////

      // std::vector<std::pair<int,int>> mm_vec = {};
      std::vector<std::pair<int,int>> mm_vec = {{3,2}};

      // std::vector<std::pair<int,int>> add_vec = {};
      std::vector<std::pair<int,int>> add_vec = {{3,3}, {3,1}, {1,3}, {3,0}};
      // std::vector<std::pair<int,int>> mul_vec = {};
      std::vector<std::pair<int,int>> mul_vec = {{3,3}, {3,1}, {1,3}, {3,0}};

       std::vector<std::pair<int,int>> sub_vec = {};
      //std::vector<std::pair<int,int>> sub_vec = {{3,3}};
      // std::vector<std::pair<int,int>> sub_vec = {{3,1}, {1,3}, {3,0}};
      // std::vector<std::pair<int,int>> sub_vec = {{3,3}, {3,1}, {1,3}, {3,0}};
      // std::vector<std::pair<int,int>> sub_vec = {{3,3}, {3,1}, {1,3}, {3,0}};

      // Operation information
      std::vector<std::tuple<int,int,int>> gemm_vec;
      gemm_vec.push_back(std::make_tuple(2, 2, 1));


#ifdef MANUAL
        std::vector<std::string> node_list = {"Sub_37"}; 

        auto it = find(node_list.begin(), node_list.end(), node.Name());
        if (it == node_list.end()) {
          not_supported = true;
        } 

#else
        // std::string op_name = node.Name().substr(0, node.Name().find('_'));
        std::string op_name(node.OpType());
        // std::cout << "Node: " << node.Name() << " type: " << node.OpType() << std::endl;
        // Case 1.
        if (op_name == "MatMul" || op_name == "Add" || op_name == "Mul" || op_name == "Sub") {
          // Shape information
          std::vector<std::pair<std::string, int>> inputs = {{"mat_A", 0}, {"mat_B", 0}};


          for (int i = 0; i < (int) node.InputArgCount().size(); i++) {
            auto def = node.InputDefs().at(i);
            auto shape = def->Shape();
            if (shape == nullptr) {
              ORT_NOT_IMPLEMENTED("Shape is a nullptr. @ pim_execution_provider");
              continue;
            }
            auto input_dims = shape->dim_size();
            inputs[i].second = input_dims;
          }
          std::pair<int, int> input_shape = std::make_pair(inputs[0].second, inputs[1].second);

          if (op_name == "MatMul") {
            auto mm_it = find(mm_vec.begin(), mm_vec.end(), input_shape);
            if (mm_it == mm_vec.end()) {
              not_supported = true;
            }
          }
          else if (op_name == "Add") {
            auto add_it = find(add_vec.begin(), add_vec.end(), input_shape);
            if (add_it == add_vec.end()) {
              not_supported = true;
            }
          }
          else if (op_name == "Mul") {
            auto mul_it = find(mul_vec.begin(), mul_vec.end(), input_shape);
            if (mul_it == mul_vec.end()) {
              not_supported = true;
            }
          }
          else if (op_name == "Sub") {
            auto sub_it = find(sub_vec.begin(), sub_vec.end(), input_shape);
            if (sub_it == sub_vec.end()) {
              not_supported = true;
            }
          }
        }
        // Case 2.
       else if (op_name == "Gemm") {
         // Shape information
         std::vector<std::pair<std::string, int>> inputs = {{"mat_A", 0}, {"mat_B", 0}, {"mat_C", 0}};

         for (int i = 0; i < (int) node.InputArgCount().size(); i++) {
           auto def = node.InputDefs().at(i);
           auto shape = def->Shape();
           if (shape == nullptr) {
             ORT_NOT_IMPLEMENTED("Shape is a nullptr. Need to include pim-helper funcion @ pim_execution_provider");
             continue;
           }
           auto input_dims = shape->dim_size();
           inputs[i].second = input_dims;
         }
         std::tuple<int, int, int> input_shape = std::make_tuple(inputs[0].second, inputs[1].second, inputs[2].second);

         if (op_name == "Gemm") {
           auto gemm_it = find(gemm_vec.begin(), gemm_vec.end(), input_shape);
           if (gemm_it == gemm_vec.end()) {
             not_supported = true;
           }
         }
       }
      #endif

      if (not_supported) {
        // std::cout << "Node: " << node.Name() << " shape: (" << input_shape.first << ", " << input_shape.second << ")  not supported" << std::endl;
        LOGS_DEFAULT(WARNING) << "PIM kernel not supported. Fallback to CPU execution provider for Op type: " << node.OpType() << " node name: " << node.Name();
      } else {
        candidates.push_back(node.Index());
      }
    }
  } else {
    for (auto& node : graph.partition_nodes_) {
      candidates.push_back(node->Index());
    }
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {

    std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node_index);
    result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  }  

  return result;
}

namespace pim {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kPimExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kPimExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 9, 12, float, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, MatMul);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 12, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Add);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 12, float, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Mul);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 12, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sub);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 8, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 9, 10, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 11, 12, float, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Gemm);


//LUT operation extended: Tanh, Erf, Sqrt, Div, Pow, Relu, Sigmoid, Neg, Abs, Log
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Tanh);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 9, 12, float, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Erf);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sqrt);

// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 12, float, Div);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Div);

// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 13, float, Pow);
// // class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 12, float, Pow);

// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 11, Pow);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 12, 12, Pow);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, Pow);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Relu);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sigmoid);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Neg);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Abs);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Log);

// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sigmoid);

// // class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 7, float, Sum);
// // class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 8, 12, float, Sum);
// // class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sum);

// // class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax);
// // class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 11, 12, float, ArgMax);
// // class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, ArgMax);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

static Status ModifyLutPimKernels(KernelRegistry& kernel_registry) {
  // Need to be implemented with dynamic LUT generator compilation. --> Future works
  static const BuildKernelCreateInfoFn function_table[] = {};

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}


static Status RegisterPimKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {

    //Below operator has library without table, can be added to the registry without checking.
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 1, 8, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 9, 12, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 12, float, Add)>,      
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 12, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Mul)>,  
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 12, float, Sub)>,      
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 8, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 9, 10, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 11, 12, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Gemm)>,  
      //14

//Otherwise, LUT operators need to checking process for the dynamic LUT table generation.
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 9, 12, float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Erf)>,
      
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sqrt)>,
      

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 13, float, Pow)>,
      


      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Relu)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sigmoid)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Neg)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Abs)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Log)>,

      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 7, 12, float, Div)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Div)>,


      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 12, float, Sigmoid)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sigmoid)>,
      // // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 6, 7, float, Sum)>,
      // // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 8, 12, float, Sum)>,  
      // // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, Sum)>,
      // // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax)>, 
      // // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 11, 12, float, ArgMax)>, 
      // // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kPimExecutionProvider, kOnnxDomain, 13, float, ArgMax)>,                
  };
  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

KernelRegistryAndStatus GetPimKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterPimKernels(*ret.kernel_registry);
  // ret.st = ModifyLutPimKernels(*ret.kernel_registry);

  return ret;
}

} // namespace pim

bool PIMExecutionProvider::FileExistanceCheck(const std::string& funct, std::vector<std::string>& check_tables) {
// Furture works
    auto check_func_it = find(check_tables.begin(), check_tables.end(), funct);
    if(check_func_it == check_tables.end()){
        return false;
    }
    else {
        return true;
    }
}

PathString PIMExecutionProvider::MakeLutFileName(const std::string& funct) {
  auto make_valid_name = [](std::string name) {
    std::replace_if(
        name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
    return name;
  };
  return path_utils::MakePathString(make_valid_name(funct),".dat");
}

void PIMExecutionProvider::ReadFile(const char * fname, Bfloat16* array, size_t length){
      std::ifstream fin(fname, std::ios::binary);
      fin.seekg(0, std::ios_base::end);
      unsigned file_len = fin.tellg();
      if (file_len != length * sizeof(short))
      {
          std::cout << "Error: file length: " << file_len 
                    << "  expected: " << length * sizeof(short) << std::endl;
          return; 
      }
      
      fin.seekg(0, std::ios_base::beg);
      fin.read( (char *) array, file_len);
      fin.close();
}



std::shared_ptr<KernelRegistry>
PIMExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = onnxruntime::pim::GetPimKernelRegistry();

  //LUT table create & initialize in PIM memory
  // throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  
  return k.kernel_registry;
}

Status PIMExecutionProvider::RegisterLut() {
    Status st;
    // LutHelper helper(LUT_OPS_NUM, lut_ptr_arr);
//Pre-existed file list
/*Abs(), Div(), Erf(), Log(), Neg(), Pow2(), Relu(), Sigmoid(), Sqrt(), Tanh() */
    std::vector<std::string> lut_tables = {"Abs","Erf","Log","Neg","Relu","Sigmoid","Sqrt","Tanh"};
    //                                        0,     1,     2,    3,    4,       5,      6,      7

    Path lut_base_dir;
    Path lut_func_file;
    std::string function_name;
    lut_base_dir = Path::Parse(ToPathString(Env::Default().GetEnvironmentVar(lut_env_vars::kLutPath)));

  for(size_t i=0; i<=LUT_OPS_NUM; i++){
    function_name = lut_tables[i];
    lut_func_file = lut_base_dir / Path::Parse(MakeLutFileName(function_name));

    if (lut_base_dir.IsEmpty()) {
      ORT_NOT_IMPLEMENTED("[PIM Execution Provider] ORT_PIM_LUT_PATH isn't configured! Please use ORT_PIM_LUT_PATH to specify pre-existed lut data path");
    }
    //For future uses
    if (!FileExistanceCheck(function_name, lut_tables)){
      st = common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Target function's LUT is not generated");
    }
    else{
      const PathString file_path_str = lut_func_file.ToPathString();
      Bfloat16* lut_ptr = (Bfloat16*)(pim_malloc(LUT_SIZE*sizeof(Bfloat16)));
      if (lut_ptr == NULL) {
        perror("PIM Memory Allocation Failed");
        exit(1);
      } 
      ReadFile(file_path_str.c_str(), lut_ptr, LUT_SIZE);
      // for(size_t i=0; i<LUT_SIZE; i++){
      //   std::cout<<"16'h"<<std::hex<<lut_ptr[i]<<std::endl;
      // }
      LOGS_DEFAULT(WARNING) << "PIM LUT tensor is allocated and initialized! Op type:" <<function_name;
      lut_ptr_arr[i] = lut_ptr;
    }
  }
    // st = helper.State();
    return st;
}

Bfloat16* PIMExecutionProvider::ReturnLut(int funct_id) const {
    return lut_ptr_arr[funct_id];
}

std::unique_ptr<onnxruntime::IDataTransfer> PIMExecutionProvider::GetDataTransfer() const {
  return onnxruntime::make_unique<onnxruntime::PIMDataTransfer>();
}




}  // namespace onnxruntime