// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/pim/pim_data_transfer.h"
#include "core/providers/pim/helper/aten/convert_numeric.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#include <sys/types.h>

#if defined(__x86_64__)
  #define ioctl_ps_to_pl 0
  #define ioctl_pl_to_ps 0
#endif

namespace onnxruntime {

bool PIMDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::PIM || dst_device.Type() == OrtDevice::PIM;
}

common::Status PIMDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int /*exec_queue_id*/) const {

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;


  int pl_dma_fd = pim_args->GetFileDescriptor();
  auto num_of_elements = src.Shape().Size();

  // 1. Source : CPU -> Destination: PIM
  if ((dst_device.Type() == OrtDevice::PIM) && (src_device.Type() == OrtDevice::CPU)) { 
    ORT_ENFORCE(!src.GetIsPim());
    ORT_ENFORCE(dst.GetIsPim());

    const void* src_data = src.DataRaw();
    const float* src_p = (const float*)(src_data);
    uint16_t* dst_p = dst.MutableData<uint16_t>();

    bool decoupled_weight = dst.GetIsMatrixBDecooupled();
    if(decoupled_weight) {
      printf("Decoupled Weight Layout\n");
        int weight_row_size = src.Shape().GetDims()[0];
        int weight_col_size = src.Shape().GetDims()[1];
        // Bfloat16* layouted_p = (Bfloat16 *)(mmap(NULL, num_of_elements * sizeof(Bfloat16), PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        decoupled_PIM_weight_layout(src_p, dst_p, weight_row_size, weight_col_size);
    }

    bool layout_changed = dst.GetLayoutChanged();
    Bfloat16 *bf_buf = (Bfloat16 *)(mmap(NULL, num_of_elements * sizeof(Bfloat16), PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    Bfloat16 *pad_bf_buf = (Bfloat16 *)(mmap(NULL, dst.Shape().Size() * sizeof(Bfloat16), PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));

    if (!layout_changed && !decoupled_weight) {
      if (dst.GetIsMatrixB()) {
        auto dims = src.Shape().GetDims().size();
        // int64_t dims = tensor_shape.size();
        int64_t O = 1, X = 1, M = 1, N = 1;
        int64_t N_pad = dst.Shape().GetDims()[dims - 1];
        // int64_t N = src.Shape().GetDims()[dims - 1];

        if (dims == 4) {
          O = src.Shape().GetDims()[0];
          X = src.Shape().GetDims()[1];
          M = src.Shape().GetDims()[2];
          N = src.Shape().GetDims()[3];
        } else if (dims == 3) {
          X = src.Shape().GetDims()[0];
          M = src.Shape().GetDims()[1];
          N = src.Shape().GetDims()[2];
        } else if (dims == 2) {
          M = src.Shape().GetDims()[0];
          N = src.Shape().GetDims()[1];
        } else if (dims == 1) {
          N = src.Shape().GetDims()[0];
        } else {
          ORT_NOT_IMPLEMENTED("NOT IMPLEMENTED DATA TRANSFER SIZE WITH ZERO");
        }

        // # 1. Type conversion
        for(int i = 0; i < num_of_elements; i++) {
          bf_buf[i] = pim::float_to_short(src_p[i]);
        }
        // # 2. Align
        int CHUNK = 512;
        int row_dim = O * X * M;
        int col_dim = N;
        int col_chunk_num = col_dim / CHUNK;
        int dest_idx = 0;

        if (col_dim > CHUNK) {

            for (int i = 0; i < col_chunk_num; i++) {
                for (int j = 0; j < row_dim; j++) {
                    for (int k = 0; k < CHUNK; k++) {
                        dst_p[dest_idx] = bf_buf[(i*CHUNK)+(j*col_dim)+k];
                        // std::cout << "dest_idx: "<< dest_idx << "\tsrc_index: " << (i*CHUNK)+(j*col_dim)+k << std::endl;
                        dest_idx++;
                    }
                }
            }

            // std::cout << "DEBUG" << std::endl;

        } else if (col_dim == 512) {
          for(int i = 0; i < num_of_elements; i++) {
            dst_p[i] = bf_buf[i];
          }          
        } else {
            printf("NOT SUPPORTED, NEED ALIGNMENT");
        }
      } else {
        for(int i = 0; i < num_of_elements; i++) {
          bf_buf[i] = pim::float_to_short(src_p[i]);
        } 
    
        memcpy(dst_p, bf_buf, num_of_elements * sizeof(short));
      }
    } else if (layout_changed && !decoupled_weight){
      int64_t dst_size = dst.Shape().Size() * sizeof(uint16_t);
      std::vector<int64_t> tensor_shape = dst.Shape().GetDims();
      int64_t dims = tensor_shape.size();
      int64_t O = 1, X = 1, M = 1, N = 1;
      int64_t N_pad = tensor_shape[dims - 1];

      if (dims == 4) {
        O = src.Shape().GetDims()[0];
        X = src.Shape().GetDims()[1];
        M = src.Shape().GetDims()[2];
        N = src.Shape().GetDims()[3];
      } else if (dims == 3) {
        X = src.Shape().GetDims()[0];
        M = src.Shape().GetDims()[1];
        N = src.Shape().GetDims()[2];
      } else if (dims == 2) {
        M = src.Shape().GetDims()[0];
        N = src.Shape().GetDims()[1];
      } else if (dims == 1) {
        N = src.Shape().GetDims()[0];
      } else {
        ORT_NOT_IMPLEMENTED("NOT IMPLEMENTED DATA TRANSFER SIZE WITH ZERO");
      }

      if (dst.GetIsMatrixB()) {
        // # 1. Type conversion
        for(int i = 0; i < num_of_elements; i++) {
          bf_buf[i] = pim::float_to_short(src_p[i]);
        }

        // # 2. Pad
        for (int64_t o = 0; o < O; o++) {
          for (int64_t x = 0; x < X; x++) {
            for (int64_t m = 0; m < M; m++) {
              for (int64_t n = 0; n < N; n++) {
                pad_bf_buf[(X*M*N_pad)*o + (M*N_pad)*x + N_pad*m + n] = bf_buf[(X*M*N)*o + (M*N)*x + N*m + n];
              } // n
            } // m
          } // x
        } // o 

        // # 3. Align
        ORT_ENFORCE(N % 512 != 0);
        int CHUNK = 512;
        int row_dim = O * X * M;
        // int col_dim = N;
        // int col_chunk_num = 512 * (col_dim / 512 + 1) / CHUNK;
        int col_dim = N_pad;
        int col_chunk_num = col_dim / CHUNK;
        int dest_idx = 0;

        if (col_dim > CHUNK) {
          for (int i = 0; i < col_chunk_num; i++) {
              for (int j = 0; j < row_dim; j++) {
                  for (int k = 0; k < CHUNK; k++) {
                      dst_p[dest_idx] = pad_bf_buf[(i*CHUNK)+(j*col_dim)+k];
                      // std::cout << "dest_idx: "<< dest_idx << "\tsrc_index: " << (i*CHUNK)+(j*col_dim)+k << std::endl;
                      dest_idx++;
                  }
              }
          }

        } else {
            printf("NOT SUPPORTED, NEED ALIGNMENT");
        }
       
      } else {
        // # 1. Type conversion
        for(int i = 0; i < num_of_elements; i++) {
          bf_buf[i] = pim::float_to_short(src_p[i]);
        }
        // # 2. Data move to padded dst
        for (int64_t o = 0; o < O; o++) {
          for (int64_t x = 0; x < X; x++) {
            for (int64_t m = 0; m < M; m++) {
              for (int64_t n = 0; n < N; n++) {
                dst_p[(X*M*N_pad)*o + (M*N_pad)*x + N_pad*m + n] = bf_buf[(X*M*N)*o + (M*N)*x + N*m + n];
              } // n
            } // m
          } // x
        } // o

      } // dst.GetISMatrixB() ?
    } // layout_changed ?

  } // case 1.
  // 2. Source : PIM -> Destination: CPU
  else if ((src_device.Type() == OrtDevice::PIM) && (dst_device.Type() == OrtDevice::CPU)) {
    ORT_ENFORCE(src.GetIsPim());

    ORT_ENFORCE(!dst.GetIsPim());

    const void* src_data = src.DataRaw();
    const uint16_t* src_p = (const uint16_t*)(src_data);
    float* dst_p = dst.MutableData<float>();

    bool layout_changed = src.GetLayoutChanged();

    int64_t src_size = src.Shape().Size() * sizeof(uint16_t);

    uint16_t *bf_buf = (uint16_t *)(mmap(NULL, src_size, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    memcpy(bf_buf, src_p, num_of_elements * sizeof(short));


    if (!layout_changed) {
      // auto elem_size = dst.Shape().Size();
      for(int i = 0; i < num_of_elements; i++) {
        dst_p[i] = pim::short_to_float(bf_buf[i]);
      }        
    } else {
      int64_t src_size = src.Shape().Size() * sizeof(uint16_t);
      std::vector<int64_t> tensor_shape = src.Shape().GetDims();
      int64_t dims = tensor_shape.size();
      int64_t O = 1, X = 1, M = 1;
      int64_t N_pad = tensor_shape[dims - 1];
      int64_t N = dst.Shape().GetDims()[dims - 1];

      if (dims == 4) {
        O = tensor_shape[0];
        X = tensor_shape[1];
        M = tensor_shape[2];
      } else if (dims == 3) {
        X = tensor_shape[0];
        M = tensor_shape[1];
      } else if (dims == 2) {
        M = tensor_shape[0];
      } else {
        ORT_NOT_IMPLEMENTED("unexpected shape");
      }

      for (int64_t o = 0; o < O; o++) {
        for (int64_t x = 0; x < X; x++) {
          for (int64_t m = 0; m < M; m++) {
            for (int64_t n = 0; n < N; n++) {
              dst_p[(X*M*N)*o + (M*N)*x + N*m + n] = pim::short_to_float(bf_buf[(X*M*N_pad)*o + (M*N_pad)*x + N_pad*m + n]);
            } // n
          } // m
        } // x
      } // o
      // std::cout << "DEBUG" << std::endl;

    // munmap(bf_buf, src_size);
    } // layout_changed

  } // case 2.
  else {
    ORT_NOT_IMPLEMENTED("Not implemented copy transfer.");
  } // case 3.

  return Status::OK();
}

common::Status PIMDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, OpKernelContext& ctx, int /*exec_queue_id*/) const {

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

 
  ioctl_info* set_info = pim_args->GetSetInfo();
  int pl_dma_fd = pim_args->GetFileDescriptor();

  int64_t dma_tx = 0;
  void   *dma_tx_ptr;
  dma_tx_ptr = &dma_tx;

  auto num_of_elements = src.Shape().Size();

  // 1. Source : CPU -> Destination: PIM
  if ((dst_device.Type() == OrtDevice::PIM) && (src_device.Type() == OrtDevice::CPU)) { 
    ORT_ENFORCE(!src.GetIsPim());
    ORT_ENFORCE(dst.GetIsPim());

    const void* src_data = src.DataRaw();
    const float* src_p = (const float*)(src_data);
    uint16_t* dst_p = dst.MutableData<uint16_t>();

    bool layout_changed = dst.GetLayoutChanged();

    short *bf_buf = (short *)(mmap(NULL, num_of_elements * sizeof(short), PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));

    for(int i = 0; i < num_of_elements; i++) {
      bf_buf[i] = pim::float_to_short(src_p[i]);
    }  
#ifdef ioctl_ps_to_pl
    // std::cout << "MEMCPY PS --> PL" << std::endl; 
    memcpy(dst_p, bf_buf, num_of_elements * sizeof(short));
#else
    set_info->srcA_ptr      = &bf_buf[0]; 
    set_info->srcB_ptr      = NULL;
    set_info->dstC_ptr      = &dst_p[0];
    set_info->srcA_va       = (uint64_t) &bf_buf[0];
    set_info->srcB_va       = 0x0;
    set_info->dstC_va       = (uint64_t) &dst_p[0];      
    set_info->srcA_size = num_of_elements * sizeof(short);
    set_info->srcB_size = 0x0;
    set_info->dstC_size = num_of_elements * sizeof(short);
    set_info->p_size    = 0x0;
    set_info->q_size    = 0x0;
    set_info->r_size    = 0x0;

    set_info->dma_tx     = dma_tx;
    set_info->dma_tx_ptr = dma_tx_ptr;

    // ioctl(pl_dma_fd, MEM_CPY_PL_DST, set_info);
    auto begin_exe = std::chrono::high_resolution_clock::now();
    if (ioctl(pl_dma_fd, MEMCPY_PS2PL, set_info) < 0) {
      printf("PS --> PL ERROR DMA \n");
      exit(-1);
    }
    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx.exe_dur = exe_dur;
    ctx.dma_tx = dma_tx;
#endif

  } // case 1.
  // 2. Source : PIM -> Destination: CPU
  else if ((src_device.Type() == OrtDevice::PIM) && (dst_device.Type() == OrtDevice::CPU)) {
    ORT_ENFORCE(src.GetIsPim());
    ORT_ENFORCE(!dst.GetIsPim());

    const void* src_data = src.DataRaw();
    const uint16_t* src_p = (const uint16_t*)(src_data);
    float* dst_p = dst.MutableData<float>();

    bool layout_changed = src.GetLayoutChanged();

    int64_t src_size = src.Shape().Size() * sizeof(uint16_t);
    // uint16_t *bf_buf = (uint16_t *) malloc(src_size);
    uint16_t *bf_buf = (uint16_t *)(mmap(NULL, src_size, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));

    // auto elem_size = src.Shape().Size();
#ifdef ioctl_pl_to_ps
    // std::cout << "MEMCPY PL --> PS" << std::endl; 
    memcpy(bf_buf, src_p, num_of_elements * sizeof(short));

    // for (int i = 0; i < num_of_elements; i++) {
    //   if (pim::short_to_float(src_p[i]) - pim::short_to_float(bf_buf[i]) > 0.002) {
    //     std::cout << "MEMCPY: Error in pl to ps memcpy" << std::endl;
    //     std::cout << "i: " << i << "\t" << "src_p: " << pim::short_to_float(src_p[i]) << "\t" << pim::short_to_float(bf_buf[i]) << std::endl;
    //   }
    // }

#else
    set_info->srcA_ptr      = &src_p[0]; 
    set_info->srcB_ptr      = NULL;
    set_info->dstC_ptr      = &bf_buf[0];
    set_info->srcA_va       = (uint64_t) &src_p[0];
    set_info->srcB_va       = 0x0;
    set_info->dstC_va       = (uint64_t) &bf_buf[0];      
    set_info->srcA_size = src_size;
    set_info->srcB_size = 0x0;
    set_info->dstC_size = src_size; 
    set_info->p_size    = 0x0;
    set_info->q_size    = 0x0;
    set_info->r_size    = 0x0;

    set_info->dma_tx     = dma_tx;
    set_info->dma_tx_ptr = dma_tx_ptr;

    // ioctl(pl_dma_fd, MEM_CPY_PL_SRC, set_info);  
    auto begin_exe = std::chrono::high_resolution_clock::now();
    if (ioctl(pl_dma_fd, MEMCPY_PL2PS, set_info) < 0) {
      printf("PL --> PS ERROR DMA \n");
      exit(-1);
    }
    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx.exe_dur = exe_dur;
    ctx.dma_tx = dma_tx;
#endif


    for(int i = 0; i < num_of_elements; i++) {
      dst_p[i] = pim::short_to_float(bf_buf[i]);
    }        
  
  } // case 2.
  else {
    ORT_NOT_IMPLEMENTED("Not implemented copy transfer.");
  } // case 3.

  return Status::OK();
}

void decoupled_PIM_weight_layout (const float* src, uint16_t* dst, int row_size, int col_size){
    // const void* src_data = src.DataRaw();
    // const float* src_p = (const float*)(src_data);
    // uint16_t* dst_p = dst.MutableData<uint16_t>();  
    
    // int A_ROW = p_size;
    int A_COL = row_size;
    int B_COL = col_size;
    int REG_SIZE = 32;
    int COMPUTE_WAY = 8;
    int NUM_BANK = 16;
    int A_COL_PAD, B_COL_PAD;    
      A_COL_PAD = (A_COL + REG_SIZE - 1) / REG_SIZE * REG_SIZE;
      B_COL_PAD = (B_COL + NUM_BANK - 1) / NUM_BANK * NUM_BANK;

    float cnt = 0;
    for (int jo = 0; jo < B_COL_PAD; jo += 16) {
        for (int ko = 0; ko < A_COL_PAD; ko += 32) {
            for (int ji = 0; ji < 16; ji++) {
                for (int ki = 0; ki < 32; ki++) {
                    if (ko + ki < A_COL && jo + ji < B_COL) {
                        float tmp0 = src[(ko + ki) * B_COL + (jo + ji)];
                        short tmp  = pim::float_to_short(tmp0);
                        dst[int(cnt)] = tmp;
                        // PL_srcB_buf[(int)cnt] = tmp0;
                        // src_B_DRAM[(ko + ki) * B_COL + (jo + ji)] = pim::short_to_float(tmp0);
                    }
                    else {
                        dst[int(cnt)] = 0;
                        // PL_srcB_buf[(int)cnt] = 0;
                    }
                    cnt += 1;
                }
            }
        }
    }   
}

};  // namespace onnxruntime
