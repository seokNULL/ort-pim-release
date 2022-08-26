// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "simple_tensor_allocator.h"
#include "tensorprotoutils.h"

namespace onnxruntime {
common::Status SimpleTensorAllocator::Trace(int id, const ONNX_NAMESPACE::TensorProto* value) {
  values_[id] = value;
  return Status::OK();
}

// common::Status SimpleTensorAllocator::GetPreallocatedBuffer(int ort_value_index, const char* name,
//                                                             std::unique_ptr<MemBuffer>& out) {
//   auto iter = values_.find(ort_value_index);
//   if (iter == values_.end()) {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "invalid ort_value_index:", ort_value_index);
//   }

//   size_t len = 0;
//   ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<kAllocAlignment>(*iter->second, &len));
//   const struct OrtMemoryInfo& location = seq_plan_.GetLocation(ort_value_index);
//   if (len == 0) {
//     out = onnxruntime::make_unique<MemBuffer>(nullptr, 0, location);
//     return Status::OK();
//   }
//   auto alloc = GetAllocator(location);
//   if (!alloc)
//     return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get allocator for initializer '", name,
//                            "', location: ", location.ToString());
//   void* buffer = alloc->Alloc(len);
//   weights_buffers_.push_back(BufferUniquePtr(buffer, alloc));
//   out = onnxruntime::make_unique<MemBuffer>(buffer, len, location);
//   return Status::OK();
// }

common::Status SimpleTensorAllocator::GetPreallocatedBuffer(int ort_value_index, const char* name,
                                                            std::unique_ptr<MemBuffer>& out, 
                                                            std::map<std::string, std::vector<std::string>>& align_map) {
  auto iter = values_.find(ort_value_index);
  if (iter == values_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "invalid ort_value_index:", ort_value_index);
  }

  size_t len = 0;

  const struct OrtMemoryInfo& location = seq_plan_.GetLocation(ort_value_index);

  std::vector<size_t> tensor_shape;
  bool is_pim = strcmp(location.name, "Pim") == 0 ? 1 : 0;
  if (is_pim) {
    ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProtoToPim<kAllocAlignment>(*iter->second, &len, tensor_shape));
    size_t dims = tensor_shape.size();
    if (dims != 0) {


      size_t last_dim = tensor_shape[dims - 1];
      len = len / sizeof(float) * sizeof(short);

      // Need align
      bool need_align = false;
      std::string s(name);

      for (auto& it: align_map) {
        if (it.first == "mm_b") {
          if (std::find(it.second.begin(), it.second.end(), s) != it.second.end())
            {
              need_align = true;
            }
        } else {
          if (std::find(it.second.begin(), it.second.end(), s) != it.second.end())
            {
              need_align = true;
            }
        }
      }

      if (need_align) {
        if (last_dim % 512 != 0) {
          size_t N_pad = 512 * (last_dim / 512 + 1);
          len = (len / last_dim) * N_pad;
        }
      }

    } else {
      len = len / sizeof(float) * sizeof(short);
    }
  } else {
    ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<kAllocAlignment>(*iter->second, &len));
  }

  if (len == 0) {
    out = onnxruntime::make_unique<MemBuffer>(nullptr, 0, location);
    return Status::OK();
  }
  auto alloc = GetAllocator(location);
  if (!alloc)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get allocator for initializer '", name,
                           "', location: ", location.ToString());
  void* buffer = alloc->Alloc(len);
  weights_buffers_.push_back(BufferUniquePtr(buffer, alloc));
  out = onnxruntime::make_unique<MemBuffer>(buffer, len, location);
  return Status::OK();
}

}  // namespace onnxruntime
