// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/pim/math/matmul.h"
// #include "core/providers/pim_arm/math/matmul_helper.h"
#include "core/util/math.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
namespace pim {

#define REGISTER_KERNEL_TYPED(T)                                                      \
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                              \
      MatMul,                                                                         \
      kOnnxDomain,                                                                    \
      1,                                                                              \
      8,                                                                              \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      MatMul<T>);                                                                     \
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                              \
      MatMul,                                                                         \
      kOnnxDomain,                                                                    \
      9,                                                                              \
      12,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      MatMul<T>);                                                                     \
ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      MatMul,                                                                         \
      kOnnxDomain,                                                                    \
      13,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      MatMul<T>);                                                                     

REGISTER_KERNEL_TYPED(float)

template <typename T>
Status MatMul<T>::Compute(OpKernelContext* ctx) const {

  int pl_dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* set_info = pim_args->GetSetInfo();

  const auto A = ctx->Input<Tensor>(0);
  const auto B = ctx->Input<Tensor>(1);

  bool layout_changed = B->GetLayoutChanged();

  auto a_dim = A->Shape().NumDimensions();
  auto b_dim = B->Shape().NumDimensions();  
  
  const Bfloat16* A_PL_ptr = A->Data<Bfloat16>();
  const Bfloat16* B_PL_ptr = B->Data<Bfloat16>();

  int64_t dma_tx = 0;
  void   *dma_tx_ptr;
  dma_tx_ptr = &dma_tx;

  if (a_dim == 3 && b_dim == 2) {
    int64_t X = A->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 FOR NOW");
    int64_t M = A->Shape()[1];
    int64_t K = B->Shape()[0];
    int64_t N = B->Shape()[1];
    int64_t N_org = B->GetOriginalShape()[1];
    
    bool is_decoupled_compute = (M >=32)? 1: 0;
    // bool is_decoupled_compute = 0;

  if(!is_decoupled_compute){
    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({1, M, N_org}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
    ctx->tensor_alloc_dur = tensor_alloc_dur;

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({1, M, N_org}));

    Bfloat16* aligned_Y_PL_ptr = nullptr;
    if (layout_changed) {
      aligned_Y_PL_ptr = (Bfloat16 *)pim_malloc(X * M * N * sizeof(Bfloat16));
    } 

    auto out_ptr = layout_changed ? aligned_Y_PL_ptr : Y_PL_ptr;

    set_info->srcA_ptr      = &A_PL_ptr[0];
    set_info->srcB_ptr      = &B_PL_ptr[0];
    set_info->dstC_ptr      = &out_ptr[0];
    set_info->srcA_va   = (uint64_t) &A_PL_ptr[0];
    set_info->srcB_va   = (uint64_t) &B_PL_ptr[0];
    set_info->dstC_va   = (uint64_t) &out_ptr[0];
    set_info->srcA_size = X * M * K * sizeof(Bfloat16);
    set_info->srcB_size = K * N * sizeof(Bfloat16);
    set_info->dstC_size = X * M * N * sizeof(Bfloat16);
    set_info->p_size    = X * M;
    set_info->q_size    = K;
    set_info->r_size    = N;

    set_info->dma_tx     = dma_tx;
    set_info->dma_tx_ptr = dma_tx_ptr;

    auto begin_exe = std::chrono::high_resolution_clock::now();
    if (ioctl(pl_dma_fd, MATMUL, set_info) < 0) {
        printf("ERROR DMA \n");
        exit(-1);
    }   
    // end
    if (layout_changed) {
      for (int x = 0; x < X; x++) {
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N_org; n++) {
            Y_PL_ptr[(M*N_org)*x + N_org*m + n] = aligned_Y_PL_ptr[(M*N)*x + N*m + n];
            // continue;
          }
        }
      }
    }
    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur = exe_dur;
    ctx->dma_tx = dma_tx;
    if (layout_changed) {
        pim_free(aligned_Y_PL_ptr);
    }
  } // Silent PIM Compute END

  else if(is_decoupled_compute){
    bool supported_size_condition = ((K%32==0)&&(N%16==0))? 1: 0;
    ORT_ENFORCE(supported_size_condition, "Matrix multiplication size (pxq)x(qxr) condition: q (multiple of 32), r(multiple of 16)");

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({1, M, N}), true);

    Bfloat16* layouted_in = nullptr;
    layouted_in = (Bfloat16 *)pim_malloc(X * M * K * sizeof(Bfloat16));

    int a_row_pad;
    a_row_pad = decoupled_PIM_input_layout (A_PL_ptr, layouted_in, M, K);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
    ctx->tensor_alloc_dur = tensor_alloc_dur;

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    
    Bfloat16* result_Y_PL_ptr = nullptr;
    result_Y_PL_ptr = (Bfloat16 *)pim_malloc(X * M * N * sizeof(Bfloat16));

    
    int srcA_size = M * K;
    int srcB_size = K * N;
    int dstC_size = M * N;
    // Y->SetOriginalShape(TensorShape({1, M, N_org}));

    set_info->srcA_ptr      = &layouted_in[0];
    set_info->srcB_ptr      = &B_PL_ptr[0];
    set_info->dstC_ptr      = &result_Y_PL_ptr[0];
    // set_info->dstC      = &Y_PL_ptr[0];

    set_info->srcA_va   = (uint64_t) &layouted_in[0];
    set_info->srcB_va   = (uint64_t) &B_PL_ptr[0];
    set_info->dstC_va   = (uint64_t) &result_Y_PL_ptr[0];
    // set_info->dstC_pa   = (uint64_t) &Y_PL_ptr[0];

    set_info->srcA_size = srcA_size*sizeof(Bfloat16);
    set_info->srcB_size = srcB_size*sizeof(Bfloat16);
    set_info->dstC_size = dstC_size*sizeof(Bfloat16);
    set_info->p_size    = a_row_pad;
    set_info->q_size    = K;
    set_info->r_size    = N;

    set_info->dma_tx     = dma_tx;
    set_info->dma_tx_ptr = dma_tx_ptr;

    auto begin_exe = std::chrono::high_resolution_clock::now();
    if (ioctl(pl_dma_fd, MATMUL_DECOUPLED, set_info) < 0) {
        printf("ERROR DMA \n");
        exit(-1);
    }   

    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur = exe_dur;
    ctx->dma_tx = dma_tx;
    decoupled_PIM_result_layout(result_Y_PL_ptr, Y_PL_ptr, M, N);
    

    pim_free(layouted_in);
    pim_free(result_Y_PL_ptr);
  }
    return Status::OK();

  } // shape 3D
  else {
    ORT_NOT_IMPLEMENTED("NOT IMPLEMENTED SHAPE");
  }

} // MatMul::Compute

}  // namespace pim

int decoupled_PIM_input_layout (const Bfloat16* src,  Bfloat16* dst, int row_size, int col_size){
    int A_ROW = row_size;
    int A_COL = col_size;
    
    int REG_SIZE = 32;
    int COMPUTE_WAY = 8;
    int NUM_BANK = 16;
    int A_COL_PAD, A_ROW_PAD;
    int idx = 0;
      A_COL_PAD = (A_COL + REG_SIZE - 1) / REG_SIZE * REG_SIZE;
      
      if (row_size%32 == 0)
        A_ROW_PAD = (A_ROW + REG_SIZE - 1) / REG_SIZE * REG_SIZE;
      else
        A_ROW_PAD = (A_ROW + COMPUTE_WAY - 1) / COMPUTE_WAY * COMPUTE_WAY;


          for (int io = 0; io < A_ROW_PAD; io += REG_SIZE) {
              for (int k = 0; k < A_COL_PAD; k++) {
                  for (int ii = 0; ii < REG_SIZE; ii++) {
                      // float tmp = generate_random_255();
                      // short tmp0 = float_to_short(tmp);
                      if (io + ii < A_ROW && k < A_COL) {
                        short tmp0 = src[(io + ii) * A_COL + k];
                        dst[idx]   = tmp0;
                        // PL_srcA_buf[idx] = tmp0;
                        // src_A_DRAM[(io + ii) * A_COL + k] = short_to_float(tmp0);
                      }
                      else {
                        dst[idx] = 0;
                        // PL_srcA_buf[idx] = 0;
                      }
                      // cnt += 1;
                      idx += 1;
                  }
              }
          }
  return A_ROW_PAD;
}

void decoupled_PIM_result_layout (Bfloat16* src, Bfloat16* dst, int row_size, int col_size){
  int NUM_BANK = 16;
  // int A_ROW = row_size;
  int B_COL = col_size;
  int REG_SIZE = 32;
  int COMPUTE_WAY = 8;
  int A_ROW_PAD = row_size;
  int cnt;
        cnt = 0;
        ////// result layout
        for (int io = 0; io < A_ROW_PAD; io += REG_SIZE) {
            for (int j = 0; j < B_COL; j++) {
                for (int ii = 0; ii < REG_SIZE; ii++) {
                    short tmp = src[(int)cnt];
                    dst[(io + ii) * B_COL + j] = tmp;
                    // a.f_val = dst_C_DRAM[(io + ii) * B_COL + j];
                    // printf("%0.f idx[%d] 0x%x || 0x%x\n", cnt, (int)((io + ii) * B_COL + j), a.u_val,  PL_dstC_buf[(int)cnt]);
                    cnt += 1;
                }
            }
        }
}
}  // namespace onnxruntime
