// // pytorch_aim/c10/aim/AIMInit.h
// #pragma once

// // #include "pim_macros.h"
#pragma once
// #include <vector>
// #include <fcntl.h>      // O_WRONLY
// #include <unistd.h>     // write(), close()
// #include "core/providers/pim/dma_driver/dma_lib.h"
// #include <stdarg.h>     // va_args
// #include <sys/mman.h>
// #include <sys/syscall.h>
// #include <stdint.h>
// #include <stdio.h>
// #include <cstdlib> 

// namespace pim {
// typedef uint16_t Bfloat16;

// static int __pl_dma_fd;
// static int __fd_ps_dma;
// static int __fd_conf;

// static Bfloat16* __ones;
// static Bfloat16* __point_fives;


// void aim_reg_init();
// void init_bias();
// uint64_t *get_pl_conf_reg();
// int get_fd_pl_dma();
// int get_fd_ps_dma();
// Bfloat16* get_bias_ones_2048();
// Bfloat16* get_bias_point_fives_2048();


// } // pim
#include <vector>
#include <fcntl.h>      // O_WRONLY
#include <unistd.h>     // write(), close()
#include <stdarg.h>     // va_args
#include <sys/mman.h>
#include <sys/syscall.h>
#include <stdint.h>
#include <stdio.h>
#include <cstdlib> 


#include <pim.h>

namespace onnxruntime {

  // static ioctl_info* set_info = (ioctl_info *)(mmap(NULL, sizeof(ioctl_info), PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));

  void SetPimDevice();
  void FreePimDevice();
  int GetPimDevice();
  ioctl_info* GetSetInfo();

  // void SetPimDevice() {
  //   int _pl_dma_fd = 0;
  //   if ((_pl_dma_fd =open(PL_DMA_DRV, O_RDWR|O_SYNC)) < 0) {
  //     perror("pl_dma open fail");
  //     exit(-1);
  //   }    
  //   pl_dma_fd = _pl_dma_fd;
  // }

  // int GetPimDevice() {
  //   return pl_dma_fd;
  // }
  
} // namespace onnxruntime
