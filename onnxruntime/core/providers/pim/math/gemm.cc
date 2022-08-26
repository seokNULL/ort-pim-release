// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/pim/math/gemm.h"
// #include "core/providers/cpu/math/gemm_matmul_common.h"
// #include "core/util/math_cpuonly.h"
// #include "gemm_helper.h"
// #include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace pim {

#define REGISTER_KERNEL_TYPED(T)                                                      \
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                              \
      Gemm,                                                                         \
      kOnnxDomain,                                                                    \
      7,                                                                              \
      8,                                                                              \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Gemm<T>);                                                                     \
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                              \
      Gemm,                                                                         \
      kOnnxDomain,                                                                    \
      9,                                                                              \
      10,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Gemm<T>);                                                                     \
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                              \
      Gemm,                                                                         \
      kOnnxDomain,                                                                    \
      11,                                                                              \
      12,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Gemm<T>);                                                                     \
ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      Gemm,                                                                         \
      kOnnxDomain,                                                                    \
      13,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      Gemm<T>);                                                                     

REGISTER_KERNEL_TYPED(float)

template <typename T>
Status Gemm<T>::Compute(OpKernelContext* ctx) const {

  int pl_dma_fd = pim_args->GetFileDescriptor();
  // ioctl_info* set_info = (ioctl_info *)(mmap(NULL, sizeof(ioctl_info), PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ioctl_info* set_info = pim_args->GetSetInfo();

  const auto* A = ctx->Input<Tensor>(0);
  const auto* B = ctx->Input<Tensor>(1);
  const auto* C = ctx->Input<Tensor>(2);

  bool layout_changed = B->GetLayoutChanged() && C->GetLayoutChanged();

  auto a_dim = A->Shape().NumDimensions();
  auto b_dim = B->Shape().NumDimensions(); 
  auto c_dim = C->Shape().NumDimensions();  

  ORT_ENFORCE(alpha_ == 1, "Gemm alpha 1 for now");
  ORT_ENFORCE(beta_ == 1, "Gemm beta 1 for now");
  ORT_ENFORCE(is_trans_A_ == 0, "Gemm Matrix A no transpose for now");

  const Bfloat16* A_PL_ptr = A->Data<Bfloat16>();
  const Bfloat16* B_PL_ptr = B->Data<Bfloat16>();
  const Bfloat16* C_PL_ptr = C->Data<Bfloat16>();

  int64_t dma_tx = 0;
  void   *dma_tx_ptr;
  dma_tx_ptr = &dma_tx;

  if (a_dim == 2 && b_dim == 2 && c_dim == 1) {
    // A (M, K) x B (K, N) + C (N) = Y (M, N)
    int64_t M = A->Shape()[0];
    int64_t K = A->Shape()[1];
    int64_t N = C->Shape()[0];
    int64_t N_org = C->GetOriginalShape()[0];
    // int64_t N_org = C->GetOriginalShape()[1];
    // ORT_ENFORCE(N == C->Shape()[0], "Gemm C shape should be same as B.");

    // Allocate Tensor Y. Remove tensor allocation time @ sequential executor.cc

    if (!is_trans_B_) {    
        auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
        Tensor* Y = ctx->Output(0, TensorShape({M, N_org}), true);
        long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
        ctx->tensor_alloc_dur = tensor_alloc_dur;

        Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
        Y->SetIsPim();
        Y->SetOriginalShape(TensorShape({M, N_org}));

        Bfloat16* aligned_Y_PL_ptr = nullptr;
        if (layout_changed) {
        //   aligned_Y_PL_ptr = (Bfloat16 *)(mmap(0x0, M * N * sizeof(Bfloat16), PROT_WRITE|PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_PCIE , -1, 0));
          aligned_Y_PL_ptr = (Bfloat16 *)pim_malloc(M * N * sizeof(Bfloat16));

            // for(int i=0; i < M * N * sizeof(Bfloat16); i++){
            //     aligned_Y_PL_ptr[i] = 0;
            // }          
        }

        // for (int m = 0; m < M; m++) {
        //     for (int n = 0; n < N_org; n++) {
        //         Y_PL_ptr[N_org * m + n] = 0;
        //     }
        // }

        auto out_ptr = layout_changed ? aligned_Y_PL_ptr : Y_PL_ptr;

        set_info->srcA_ptr      = A_PL_ptr;
        set_info->srcB_ptr      = B_PL_ptr;
        set_info->srcbias_ptr   = C_PL_ptr;
        set_info->dstC_ptr      = out_ptr;

        set_info->srcA_va   = (uint64_t)&A_PL_ptr[0];
        set_info->srcB_va   = (uint64_t)&B_PL_ptr[0];
        set_info->bias_va   = (uint64_t)&C_PL_ptr[0];
        set_info->dstC_va   = (uint64_t)&out_ptr[0];

        set_info->srcA_size    = M * K * sizeof(Bfloat16);
        set_info->srcB_size    = K * N * sizeof(Bfloat16);
        set_info->dstC_size    = M * N * sizeof(Bfloat16);

        set_info->p_size    = M;
        set_info->q_size    = K;
        set_info->r_size    = N;

        set_info->dma_tx     = dma_tx;
        set_info->dma_tx_ptr = dma_tx_ptr;

        auto begin_exe = std::chrono::high_resolution_clock::now();
        if (ioctl(pl_dma_fd, GEMM, set_info) < 0) {
            printf("ERROR DMA \n");
            exit(-1);
        }   

        if (layout_changed) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N_org; n++) {
                    Y_PL_ptr[N_org * m + n] = aligned_Y_PL_ptr[N * m + n];
                    // Y_PL_ptr[N_org * m + n] = out_ptr[N * m + n];
                }
            }
        }
        long long exe_dur = TimeDiffMicroSeconds(begin_exe);
        ctx->exe_dur = exe_dur;
        ctx->dma_tx = dma_tx;

        if (layout_changed) {
          pim_free(aligned_Y_PL_ptr);
            //if (munmap(aligned_Y_PL_ptr, M * N * sizeof(Bfloat16)) == -1) {
            //    perror("aligned_Y_PL_ptr munmap() FAILED");
            //}                     
        }
        return Status::OK();
    } else {
        bool need_pad = N % 512 != 0;
        // ORT_ENFORCE(need_pad == layout_changed);
        int64_t N_pad = need_pad ? 512 * (N / 512 + 1) : N;
        int64_t K_pad = K % 32 != 0 ? 32 * (K / 32 + 1) : K;

        auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
        Tensor* Y = ctx->Output(0, TensorShape({M, N}), true);
        long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
        ctx->tensor_alloc_dur = tensor_alloc_dur;

        Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
        Y->SetIsPim();
        Y->SetOriginalShape(TensorShape({M, N_org}));

        Bfloat16* padded_A_PL_ptr = nullptr;
        if (K % 32 != 0) {
            padded_A_PL_ptr = (Bfloat16 *)pim_malloc(M * K_pad * sizeof(Bfloat16));
            for (int m = 0; m < M; m++) {
                for (int k = 0; k < K; k++) {
                    padded_A_PL_ptr[m*K_pad+k] = A_PL_ptr[m*K+k];
                }
            }
        }

        // A (M, K) x B (N, K) + C (N) = Y (M, N)
        // int64_t N = B->Shape()[0];
        Bfloat16* transB_PL_ptr = (Bfloat16 *)pim_malloc(K_pad * N * sizeof(Bfloat16));
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                // Bfloat16 tmp = B_PL_ptr[K * n + k];
                transB_PL_ptr[N * k + n] = B_PL_ptr[K * n + k];
            }
        }

        Bfloat16* align_transB_PL_ptr = nullptr;
        if (need_pad) {
            // ORT_NOT_IMPLEMENTED("SHOULD NOT BE IN HERE FOR NOW.");
            Bfloat16 *padded_transB_PL_ptr = (Bfloat16 *)pim_malloc(K_pad * N_pad * sizeof(Bfloat16));
            align_transB_PL_ptr = (Bfloat16 *)pim_malloc(K_pad * N_pad * sizeof(Bfloat16));

            // 1. pad
            for (int k = 0; k < K; k++) {
                for (int n = 0; n < N; n++) {
                    padded_transB_PL_ptr[k*N_pad+n] = transB_PL_ptr[k*N+n];
                }
            }
            ORT_ENFORCE(N % 512 != 0);
            int CHUNK = 512;
            int row_dim = K_pad;
            int col_dim = N_pad;
            int col_chunk_num = col_dim / CHUNK;
            int dest_idx = 0;
            // 2. align
            for (int i = 0; i < col_chunk_num; i++) {
              for (int j = 0; j < row_dim; j++) {
                  for (int k = 0; k < CHUNK; k++) {
                      align_transB_PL_ptr[dest_idx] = padded_transB_PL_ptr[(i*CHUNK)+(j*col_dim)+k];
                      // std::cout << "dest_idx: "<< dest_idx << "\tsrc_index: " << (i*CHUNK)+(j*col_dim)+k << std::endl;
                      dest_idx++;
                  }
              }
            }       
        }

        Bfloat16* paddedC_PL_ptr = (Bfloat16 *)pim_malloc(M * N_pad * sizeof(Bfloat16));
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                paddedC_PL_ptr[m*N_pad+n] = C_PL_ptr[n];
            }
        }
        Bfloat16* aligned_Y_PL_ptr = nullptr;
        if (need_pad) {
          aligned_Y_PL_ptr = (Bfloat16 *)pim_malloc(M * N_pad * sizeof(Bfloat16));
        } 
    
        auto inp_ptr = K % 32 != 0 ? padded_A_PL_ptr : A_PL_ptr;
        auto wgt_ptr = need_pad ? align_transB_PL_ptr : transB_PL_ptr;
        auto bias_ptr = paddedC_PL_ptr;
        auto out_ptr = need_pad ? aligned_Y_PL_ptr : Y_PL_ptr;

        set_info->srcA_ptr      = inp_ptr;
        set_info->srcB_ptr      = wgt_ptr;
        set_info->srcbias_ptr   = bias_ptr;
        set_info->dstC_ptr      = out_ptr;

        set_info->srcA_va       = (uint64_t)&inp_ptr[0];
        set_info->srcB_va       = (uint64_t)&wgt_ptr[0];
        set_info->bias_va       = (uint64_t)&bias_ptr[0];
        set_info->dstC_va       = (uint64_t)&out_ptr[0];

        set_info->srcA_size     = M * K_pad * sizeof(Bfloat16);
        set_info->srcB_size     = K_pad * N_pad * sizeof(Bfloat16);
        set_info->dstC_size     = M * N_pad * sizeof(Bfloat16);

        set_info->p_size        = M;
        set_info->q_size        = K_pad;
        set_info->r_size        = N_pad;    

        set_info->dma_tx        = dma_tx;
        set_info->dma_tx_ptr    = dma_tx_ptr;

        auto begin_exe = std::chrono::high_resolution_clock::now();
        if (ioctl(pl_dma_fd, GEMM, set_info) < 0) {
            printf("ERROR DMA \n");
            exit(-1);
        }   

        if (need_pad) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    Y_PL_ptr[N * m + n] = aligned_Y_PL_ptr[N_pad * m + n];
                }
            }
        }

        long long exe_dur = TimeDiffMicroSeconds(begin_exe);
        ctx->exe_dur = exe_dur;
        ctx->dma_tx = dma_tx;

        pim_free(transB_PL_ptr);
        if (need_pad) {
            pim_free(align_transB_PL_ptr);         
        }

        pim_free(paddedC_PL_ptr);
        if (layout_changed) {
          pim_free(aligned_Y_PL_ptr);                   
        }
        return Status::OK();        
    }
  }
  else {
    ORT_NOT_IMPLEMENTED("NOT IMPLEMENTED SHAPE");
  }
}

}  // namespace pim
}  // namespace onnxruntime
