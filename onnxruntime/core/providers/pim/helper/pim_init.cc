#include <iostream>
#include "pim_init.h"

//#define FPGA_ENV

namespace onnxruntime {
  int pl_dma_fd;
  ioctl_info* set_info;
  
  void SetPimDevice() {
    int _pl_dma_fd = 0;
    ioctl_info* _set_info = nullptr;

#if defined (FPGA_ENV)
      if ((_pl_dma_fd =open(PL_DMA_DRV, O_RDWR|O_SYNC)) < 0) {
        perror("pl_dma open fail");
        exit(-1);
      }
    pl_dma_fd = _pl_dma_fd;
#else
    pl_dma_fd = _pl_dma_fd;
#endif
    // _set_info = (ioctl_info *)(mmap(NULL, sizeof(ioctl_info), PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    _set_info = (ioctl_info *)(malloc(sizeof(ioctl_info)));
    set_info = _set_info;
  }

  int GetPimDevice() {
    return pl_dma_fd;
  }

  void FreePimDevice() {
#if defined (FPGA_ENV)
    close(pl_dma_fd);
#else
    return;
#endif
  }

  ioctl_info* GetSetInfo() {
    return set_info;
  }


} // namespace onnxruntime