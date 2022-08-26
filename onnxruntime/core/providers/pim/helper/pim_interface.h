// Changed TensorImpl to Tensor
#pragma once

#include <string>
#include <type_traits>
#include <utility>
#include <stdlib.h>

// #include "pim_lookup.h"
// #include "pim_init.h"

#include <sys/syscall.h>
#include <unistd.h>     // write(), close()
#include <fcntl.h>      // O_WRONLY
#include <stdarg.h>     // va_args

#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/ioctl.h> // PIM!!

#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"

#include <pim.h>


namespace onnxruntime{ 
    
extern ioctl_info* set_info;
extern int pl_dma_fd;

class PIMInterface {
 public:
  explicit PIMInterface() : pl_dma_fd_(pl_dma_fd), set_info_(set_info) {}  
    void Release();
    int GetFileDescriptor() {return pl_dma_fd_;};
    ioctl_info* GetSetInfo() {return set_info_;};

    int pl_dma_fd_;
    ioctl_info* set_info_;

};

} // onnxruntime