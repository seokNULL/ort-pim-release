#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/pim/pim_execution_provider_info.h"
#include "core/graph/graph_utils.h"
#include <set>
#include <vector>

#include "core/graph/constants.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/execution_provider.h"

#include "core/providers/pim/helper/pim_init.h"
#include <unistd.h>     // write(), close()
#include <fcntl.h>      // O_WRONLY
#include <stdarg.h>     // va_args


namespace onnxruntime {

const int CPU_ALLOCATOR_DEVICE_ID = 0;

class PIMExecutionProvider : public IExecutionProvider {
 public:
  PIMExecutionProvider(const PIMExecutionProviderInfo& info);
  virtual ~PIMExecutionProvider();

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;
  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  ProviderOptions GetProviderOptions() const override {
    return PIMExecutionProviderInfo::ToProviderOptions(info_);
  }
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  std::vector<NodeIndex> pim_nodes;

  void SetFileDescriptor() {
    // if ((pim::pl_dma_fd=open(PL_DMA_DRV, O_RDWR|O_SYNC)) < 0) {
    //     perror("pl_dma open fail");
    //     exit(-1);
    // }    
    std::cout << "DEBUG SetFileDescriptor" << std::endl; 
  }

 private:
  PIMExecutionProviderInfo info_;  
  AllocatorPtr allocator_;
};

}  // namespace onnxruntime


