// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pim_allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "pim_data_transfer.h"
#include <sys/mman.h>

namespace onnxruntime {

// static const PIMDataTransfer* GetPIMDataTransfer(const SessionState* session_state) {
//   OrtDevice pim_device(OrtDevice::PIM, OrtDevice::MemType::DEFAULT, 0);
//   OrtDevice cpu_device;
//   return static_cast<const PIMDataTransfer*>(session_state->GetDataTransferMgr().GetDataTransfer(pim_device, cpu_device));
// }

void* __attribute__((optimize("O0"))) PIMAllocator::Alloc(size_t size) {

  short *p;  
  // p = (short *)(mmap(0x0, size, PROT_WRITE|PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_PCIE , -1, 0));
  // p = (short *)(mmap(0x0, size, PROT_WRITE|PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS , -1, 0));
  p = (short *)(pim_malloc(size));

  if (p == NULL) {
    perror("PL src mmap() FAILED");
    exit(1);
  } 
  size_table_[p] = size;
  for (size_t i = 0; i < size/sizeof(Bfloat16); i+=2048) {
    ((short *) p)[i] = 0;
  }
  return (void *) p;
}

// void*  __attribute__((optimize("O0"))) PIMAllocator::PimAlloc(size_t size) {
//   void *p;
//   p = malloc(size);
//   return p;
// }

void PIMAllocator::Free(void* p) {
   
  //  auto it = size_table_.find(p);
  //  if (it != size_table_.end()) {
  //     // munmap(p, it->second);
  //     if (munmap(p, it->second) == -1) {
  //     	perror("PL src munmap() FAILED");
  //     }
  //     size_table_.erase(it);
  //  } else {
  //   ORT_NOT_IMPLEMENTED("FREE ERROR @ pim_allocator");
  //  }

   pim_free(p);
}

// FencePtr PIMAllocator::CreateFence(const SessionState* session_state) {
//   return std::make_shared<PIMFence>(GetPIMDataTransfer(session_state));
// }

}  // namespace onnxruntime
