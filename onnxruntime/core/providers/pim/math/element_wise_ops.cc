// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types_internal.h"
#include "core/providers/pim/math/element_wise_ops.h"
#include "core/util/math.h"

#include <cmath>

#if defined(__x86_64__)
#define TARGET_X86
#endif
#if defined(__aarch64__)
#define TARGET_ARM
#endif

namespace onnxruntime {
namespace pim {

#define ONNX_PIM_OPERATOR_TYPED_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kOnnxDomain, ver, type, kPimExecutionProvider, builder, __VA_ARGS__)
#define ONNX_PIM_OPERATOR_VERSIONED_TYPED_KERNEL(name, startver, endver, type, builder, ...) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kOnnxDomain, startver, endver, type, kPimExecutionProvider, builder, __VA_ARGS__)
#define ONNX_PIM_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, ver, kPimExecutionProvider, builder, __VA_ARGS__)
#define ONNX_PIM_OPERATOR_VERSIONED_KERNEL(name, startver, endver, builder, ...) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, kOnnxDomain, startver, endver, kPimExecutionProvider, builder, __VA_ARGS__)

#define REG_ELEMENTWISE_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS)         \
  ONNX_PIM_OPERATOR_TYPED_KERNEL(                                                  \
      OP_TYPE,                                                                     \
      VERSION,                                                                     \
      TYPE,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      KERNEL_CLASS<TYPE>);

#define REG_ELEMENTWISE_LOGICALOP_TYPED_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS) \
  ONNX_PIM_OPERATOR_TYPED_KERNEL(                                                    \
      OP_TYPE,                                                                       \
      VERSION,                                                                       \
      TYPE,                                                                          \
      KernelDefBuilder()                                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),                \
      KERNEL_CLASS<TYPE>);

#define REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_PIM_OPERATOR_VERSIONED_TYPED_KERNEL(                                                           \
      OP_TYPE,                                                                                        \
      VERSION_FROM, VERSION_TO,                                                                       \
      TYPE,                                                                                           \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()),                    \
      KERNEL_CLASS<TYPE>);

#define REG_ELEMENTWISE_LOGICALOP_VERSIONED_TYPED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, TYPE, KERNEL_CLASS) \
  ONNX_PIM_OPERATOR_VERSIONED_TYPED_KERNEL(                                                                     \
      OP_TYPE,                                                                                                  \
      VERSION_FROM, VERSION_TO,                                                                                 \
      TYPE,                                                                                                     \
      KernelDefBuilder()                                                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())                                             \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),                                           \
      KERNEL_CLASS<TYPE>);

// var args are type constraints for T and T1
#define REG_ELEMENTWISE_KERNEL_NONT(OP_TYPE, VERSION, KERNEL_CLASS, ...)   \
  ONNX_PIM_OPERATOR_KERNEL(                                                \
      OP_TYPE,                                                             \
      VERSION,                                                             \
      KernelDefBuilder()                                                   \
          .TypeConstraint("T", BuildKernelDefConstraints<__VA_ARGS__>())   \
          .TypeConstraint("T1", BuildKernelDefConstraints<__VA_ARGS__>()), \
      KERNEL_CLASS);

// var args are type constraints for T and T1
#define REG_ELEMENTWISE_VERSIONED_KERNEL_NONT(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, ...) \
  ONNX_PIM_OPERATOR_VERSIONED_KERNEL(                                                               \
      OP_TYPE,                                                                                      \
      VERSION_FROM,                                                                                 \
      VERSION_TO,                                                                                   \
      KernelDefBuilder()                                                                            \
          .TypeConstraint("T", BuildKernelDefConstraints<__VA_ARGS__>())                            \
          .TypeConstraint("T1", BuildKernelDefConstraints<__VA_ARGS__>()),                          \
      KERNEL_CLASS);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Add, 7, 12, float, Add);
REG_ELEMENTWISE_TYPED_KERNEL(Add, 13, float, Add);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sub, 7, 12, float, Sub);
REG_ELEMENTWISE_TYPED_KERNEL(Sub, 13, float, Sub);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Mul, 7, 12, float, Mul);
REG_ELEMENTWISE_TYPED_KERNEL(Mul, 13, float, Mul);

// Add start
template <typename T>
Status Add<T>::Compute(OpKernelContext* ctx) const {

  int pl_dma_fd = pim_args->GetFileDescriptor();

  ioctl_info* set_info = pim_args->GetSetInfo();
  const auto A = ctx->Input<Tensor>(0);
  const auto B = ctx->Input<Tensor>(1);

  auto a_dim = A->Shape().NumDimensions();
  auto b_dim = B->Shape().NumDimensions();  

  const Bfloat16* A_PL_ptr = A->Data<Bfloat16>();
  const Bfloat16* B_PL_ptr = B->Data<Bfloat16>();

  int64_t dma_tx = 0;
  void   *dma_tx_ptr;
  dma_tx_ptr = &dma_tx;

  if (a_dim == 3 && b_dim == 3) {
    int64_t X = A->Shape()[0];
    // ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = A->Shape()[1];
    int64_t N = A->Shape()[2];
    int64_t N_org = A->GetOriginalShape()[2];

    bool same_shape = A->Shape()[2] == B->Shape()[2] ? true : false;

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc);
    ctx->tensor_alloc_dur += tensor_alloc_dur;

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X, M, N_org}));

    if (same_shape) {

      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &B_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx     = dma_tx;
      set_info->dma_tx_ptr = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, ELEWISE_ADD, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur += exe_dur;
      ctx->dma_tx += dma_tx;  

    } else {
      ORT_ENFORCE(B->Shape()[2] == 1, "ELEWISE B SHAPE 1");
      int64_t bias_size = X * M * N * sizeof(Bfloat16);
      //Bfloat16* bias_ptr = (Bfloat16 *)(mmap(0x0, bias_size, PROT_WRITE|PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_PCIE , -1, 0));
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* bias_ptr = (Bfloat16 *)pim_malloc(bias_size);
      for (int x = 0; x < X; x++) {
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            bias_ptr[(M*N)*x + N * m + n] = B_PL_ptr[M*x + m];
          } // N
        } // M
      } // X
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;
      // Case 1.
      for (int x = 0; x < X; x++) {
        set_info->srcA_ptr      = &A_PL_ptr[M * N * x];
        set_info->srcB_ptr      = &bias_ptr[M * N * x];
        set_info->dstC_ptr      = &Y_PL_ptr[M * N * x];
        set_info->srcA_va       = (uint64_t) &A_PL_ptr[M * N * x];
        set_info->srcB_va       = (uint64_t) &bias_ptr[M * N * x];
        set_info->dstC_va       = (uint64_t) &Y_PL_ptr[M * N * x];
        set_info->srcA_size     = M * N * sizeof(Bfloat16);
        set_info->srcB_size     = M * N * sizeof(Bfloat16);
        set_info->dstC_size     = M * N * sizeof(Bfloat16);
        set_info->p_size        = M;
        set_info->q_size        = N;
        set_info->r_size        = N;

        set_info->dma_tx        = dma_tx;
        set_info->dma_tx_ptr    = dma_tx_ptr;

        auto begin_exe = std::chrono::high_resolution_clock::now();
        if (ioctl(pl_dma_fd, ELEWISE_ADD, set_info) < 0) {
            printf("ERROR DMA \n");
            exit(-1);
        }   
        long long exe_dur = TimeDiffMicroSeconds(begin_exe);
        ctx->exe_dur += exe_dur;
        ctx->dma_tx += dma_tx;       
      }
      pim_free(bias_ptr);
    } // same shape

    return Status::OK();
  } 
  else if (a_dim == 3 && b_dim == 1) {
    int64_t X = A->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = A->Shape()[1];
    int64_t N = A->Shape()[2];
    int64_t N_org = A->GetOriginalShape()[2];

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X, M, N_org}));

    bool need_pad = N % 512 != 0;
    // bool need_pad = true;

    auto begin_exe = std::chrono::high_resolution_clock::now();
    
    if (need_pad) {
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_bias_ptr = (Bfloat16 *)pim_malloc(X * M * N * sizeof(Bfloat16));
#if defined (TARGET_X86)
      for (int i = 0; i < X * M; i++) {
        for (int j = 0; j < N; j++) {
          padded_bias_ptr[i * N + j] = B_PL_ptr[j];
        }
      }

#elif defined (TARGET_ARM)
      set_info->srcA_ptr      = &B_PL_ptr[0]; 
      set_info->srcB_ptr      = NULL;
      set_info->srcA_va       = (uint64_t) &B_PL_ptr[0];
      set_info->srcB_va       = 0x0;
      set_info->srcA_size     = N * sizeof(short);
      set_info->srcB_size     = 0x0;
      set_info->dstC_size     = N * sizeof(short);
      set_info->p_size        = 0x0;
      set_info->q_size        = 0x0;
      set_info->r_size        = 0x0;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      // ioctl(pl_dma_fd, MEM_CPY_PL_DST, set_info);
      for (int i = 0; i < X * M; i++) {
        set_info->dstC_ptr   = &padded_bias_ptr[i * N];
        set_info->dstC_va    = (uint64_t) &padded_bias_ptr[i * N]; 
        if (ioctl(pl_dma_fd, MEMCPY_PL2PL, set_info) < 0) {
          printf("PL --> PL ERROR DMA \n");
          exit(-1);
        }
      }
#endif
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &padded_bias_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &padded_bias_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      if (ioctl(pl_dma_fd, ELEWISE_ADD, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }
      pim_free(padded_bias_ptr);

    } else {
      set_info->srcA_ptr      = &B_PL_ptr[0];
      set_info->srcB_ptr      = &A_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &B_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &A_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;
      
      if (ioctl(pl_dma_fd, BIAS_ADD, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   
    }

    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur += exe_dur;
    ctx->dma_tx += dma_tx;

    return Status::OK();
  }
  else if (a_dim == 1 && b_dim == 3) {
    int64_t X = B->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = B->Shape()[1];
    int64_t N = B->Shape()[2];

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
    ctx->tensor_alloc_dur = tensor_alloc_dur;    

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X, M, N}));

    bool need_pad = N % 512 != 0;
    // bool need_pad = true;

    auto begin_exe = std::chrono::high_resolution_clock::now();

    if (need_pad) {
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_bias_ptr = (Bfloat16 *)pim_malloc(X * M * N * sizeof(Bfloat16));
#if defined (TARGET_X86)
      for (int i = 0; i < X * M; i++) {
        for (int j = 0; j < N; j++) {
          padded_bias_ptr[i * N + j] = A_PL_ptr[j];
        }
      }
#elif defined (TARGET_ARM)
      set_info->srcA_ptr      = &A_PL_ptr[0]; 
      set_info->srcB_ptr      = NULL;
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = 0x0;
      set_info->srcA_size     = N * sizeof(short);
      set_info->srcB_size     = 0x0;
      set_info->dstC_size     = N * sizeof(short);
      set_info->p_size        = 0x0;
      set_info->q_size        = 0x0;
      set_info->r_size        = 0x0;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      // ioctl(pl_dma_fd, MEM_CPY_PL_DST, set_info);
      for (int i = 0; i < X * M; i++) {
        set_info->dstC_ptr   = &padded_bias_ptr[i * N];
        set_info->dstC_va    = (uint64_t) &padded_bias_ptr[i * N]; 
        if (ioctl(pl_dma_fd, MEMCPY_PL2PL, set_info) < 0) {
          printf("PL --> PL ERROR DMA \n");
          exit(-1);
        }
      }
#endif
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      set_info->srcA_ptr      = &padded_bias_ptr[0];
      set_info->srcB_ptr      = &B_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &padded_bias_ptr[0];
      set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      if (ioctl(pl_dma_fd, ELEWISE_ADD, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }

      pim_free(padded_bias_ptr);      

    } else {
      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &B_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      if (ioctl(pl_dma_fd, BIAS_ADD, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      } 
    }

    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur += exe_dur;
    ctx->dma_tx += dma_tx;

    return Status::OK();
  }  
  else if (a_dim == 3 && b_dim == 0) {
    int64_t X = A->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = A->Shape()[1];
    int64_t N = A->Shape()[2];
    int64_t N_org = A->GetOriginalShape()[2];

    if (N == 1) {
      bool need_pad = M % 512 != 0;
      // Ignore N.
      int M_pad = need_pad ? 512 * (M / 512 + 1) : M;
      Bfloat16* padded_A_PL_ptr = nullptr;
      if (need_pad) {
        padded_A_PL_ptr = (Bfloat16 *)pim_malloc(X * M_pad * sizeof(Bfloat16));
        for (int m = 0; m < M; m++) {
          padded_A_PL_ptr[m] = A_PL_ptr[m];
        }
      } 
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_B_PL_ptr = (Bfloat16 *)pim_malloc(1 * M_pad * sizeof(Bfloat16));
      for (int m = 0; m < M; m++) {
        padded_B_PL_ptr[m] = B_PL_ptr[0];
      }
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      Bfloat16* padded_Y_PL_ptr = nullptr;
      if (need_pad) {
        padded_Y_PL_ptr = (Bfloat16 *)pim_malloc(X * M_pad * sizeof(Bfloat16));
      }

      auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
      Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
      long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
      ctx->tensor_alloc_dur += tensor_alloc_dur;

      Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
      Y->SetIsPim();
      Y->SetOriginalShape(TensorShape({X, M, N_org}));

      set_info->srcA_ptr      = need_pad ? padded_A_PL_ptr : A_PL_ptr;
      set_info->srcB_ptr      = padded_B_PL_ptr;
      set_info->dstC_ptr      = need_pad ? padded_Y_PL_ptr : Y_PL_ptr;
      set_info->srcA_va       = need_pad ? (uint64_t) &padded_A_PL_ptr[0] : (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &padded_B_PL_ptr[0];
      set_info->dstC_va       = need_pad ? (uint64_t)&padded_Y_PL_ptr[0] : (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * sizeof(Bfloat16);
      set_info->srcB_size     = M_pad * sizeof(Bfloat16);
      set_info->dstC_size     = M_pad * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = 1;
      set_info->r_size        = M_pad;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, BIAS_ADD, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   

      if (need_pad) {
        for (int m = 0; m < M; m++) {
          Y_PL_ptr[m] = padded_Y_PL_ptr[m];
        }
      }
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur += exe_dur;
      ctx->dma_tx += dma_tx;

      pim_free(padded_B_PL_ptr);
      if (need_pad) {
        pim_free(padded_A_PL_ptr);
        pim_free(padded_Y_PL_ptr);
      }      

    } else {
      bool need_pad = N % 512 != 0;
      int N_pad = need_pad ? 512 * (N / 512 + 1) : N;

      Bfloat16* padded_A_PL_ptr = nullptr;
      if (need_pad) {
        padded_A_PL_ptr = (Bfloat16 *)pim_malloc(X * M * N_pad * sizeof(Bfloat16));
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            padded_A_PL_ptr[m*N_pad+n] = A_PL_ptr[m*N+n];
          }
        }
      } 

      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_B_PL_ptr = (Bfloat16 *)pim_malloc(1 * N_pad * sizeof(Bfloat16));
      for (int n = 0; n < N_pad; n++) {
        padded_B_PL_ptr[n] = B_PL_ptr[0];
      }
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      Bfloat16* padded_Y_PL_ptr = nullptr;
      if (need_pad) {
        padded_Y_PL_ptr = (Bfloat16 *)pim_malloc(X * M * N_pad * sizeof(Bfloat16));
      }
      auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
      Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
      long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 

      Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
      Y->SetIsPim();
      Y->SetOriginalShape(TensorShape({X, M, N_org}));

      set_info->srcA_ptr      = padded_B_PL_ptr;
      set_info->srcB_ptr      = need_pad ? padded_A_PL_ptr : A_PL_ptr;
      set_info->dstC_ptr      = need_pad ? padded_Y_PL_ptr : Y_PL_ptr;
      set_info->srcA_va       = (uint64_t) &padded_B_PL_ptr[0];
      set_info->srcB_va       = need_pad ? (uint64_t) &padded_A_PL_ptr[0] : (uint64_t) &A_PL_ptr[0];
      set_info->dstC_va       = need_pad ? (uint64_t)&padded_Y_PL_ptr[0] : (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N_pad * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N_pad * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N_pad * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N_pad;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, BIAS_ADD, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   
      if (need_pad) {
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            Y_PL_ptr[m*N+n] = padded_Y_PL_ptr[m*N_pad+n];
          }
        }
      }
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur += exe_dur;
      ctx->dma_tx += dma_tx;

      pim_free(padded_B_PL_ptr);
      if (need_pad) {
        pim_free(padded_A_PL_ptr);
        pim_free(padded_Y_PL_ptr);
      }

    }
    return Status::OK();
  }  
  else {
    ORT_NOT_IMPLEMENTED("NOT IMPLEMENTED SHAPE");
  } // not implemented
} // Add finished

// Mul start
template <typename T>
Status Mul<T>::Compute(OpKernelContext* ctx) const {

  int pl_dma_fd = pim_args->GetFileDescriptor();

  ioctl_info* set_info = pim_args->GetSetInfo();
  const auto A = ctx->Input<Tensor>(0);
  const auto B = ctx->Input<Tensor>(1);

  auto a_dim = A->Shape().NumDimensions();
  auto b_dim = B->Shape().NumDimensions();  

  const Bfloat16* A_PL_ptr = A->Data<Bfloat16>();
  const Bfloat16* B_PL_ptr = B->Data<Bfloat16>();

  int64_t dma_tx = 0;
  void   *dma_tx_ptr;
  dma_tx_ptr = &dma_tx;

  if (a_dim == 3 && b_dim == 3) {
    int64_t X = A->Shape()[0];
    // ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = A->Shape()[1];
    int64_t N = A->Shape()[2];
    int64_t N_org = A->GetOriginalShape()[2];

    bool same_shape = A->Shape()[2] == B->Shape()[2] ? true : false;

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc);
    ctx->tensor_alloc_dur = tensor_alloc_dur;

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X, M, N_org}));

    if (same_shape) {

      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &B_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx     = dma_tx;
      set_info->dma_tx_ptr = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, ELEWISE_MUL, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur += exe_dur;
      ctx->dma_tx += dma_tx;  

    } else {
      ORT_ENFORCE(B->Shape()[2] == 1, "ELEWISE B SHAPE 1");
      int64_t bias_size = X * M * N * sizeof(Bfloat16);
      //Bfloat16* bias_ptr = (Bfloat16 *)(mmap(0x0, bias_size, PROT_WRITE|PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_PCIE , -1, 0));
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* bias_ptr = (Bfloat16 *)pim_malloc(bias_size);
      for (int x = 0; x < X; x++) {
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            bias_ptr[(M*N)*x + N * m + n] = B_PL_ptr[M*x + m];
          } // N
        } // M
      } // X
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;
      // Case 1.
      for (int x = 0; x < X; x++) {
        set_info->srcA_ptr      = &A_PL_ptr[M * N * x];
        set_info->srcB_ptr      = &bias_ptr[M * N * x];
        set_info->dstC_ptr      = &Y_PL_ptr[M * N * x];
        set_info->srcA_va       = (uint64_t) &A_PL_ptr[M * N * x];
        set_info->srcB_va       = (uint64_t) &bias_ptr[M * N * x];
        set_info->dstC_va       = (uint64_t) &Y_PL_ptr[M * N * x];
        set_info->srcA_size     = M * N * sizeof(Bfloat16);
        set_info->srcB_size     = M * N * sizeof(Bfloat16);
        set_info->dstC_size     = M * N * sizeof(Bfloat16);
        set_info->p_size        = M;
        set_info->q_size        = N;
        set_info->r_size        = N;

        set_info->dma_tx        = dma_tx;
        set_info->dma_tx_ptr    = dma_tx_ptr;

        auto begin_exe = std::chrono::high_resolution_clock::now();
        if (ioctl(pl_dma_fd, ELEWISE_MUL, set_info) < 0) {
            printf("ERROR DMA \n");
            exit(-1);
        }   
        long long exe_dur = TimeDiffMicroSeconds(begin_exe);
        ctx->exe_dur += exe_dur;
        ctx->dma_tx += dma_tx;       
      }
      pim_free(bias_ptr);
    } // same shape

    return Status::OK();
  } 
  else if (a_dim == 3 && b_dim == 1) {
    int64_t X = A->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = A->Shape()[1];
    int64_t N = A->Shape()[2];
    int64_t N_org = A->GetOriginalShape()[2];

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X, M, N_org}));

    bool need_pad = N % 512 != 0;
    // bool need_pad = true;

    auto begin_exe = std::chrono::high_resolution_clock::now();
    
    if (need_pad) {
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_bias_ptr = (Bfloat16 *)pim_malloc(X * M * N * sizeof(Bfloat16));
#if defined (TARGET_X86)
      for (int i = 0; i < X * M; i++) {
        for (int j = 0; j < N; j++) {
          padded_bias_ptr[i * N + j] = B_PL_ptr[j];
        }
      }

#elif defined (TARGET_ARM)
      set_info->srcA_ptr      = &B_PL_ptr[0]; 
      set_info->srcB_ptr      = NULL;
      set_info->srcA_va       = (uint64_t) &B_PL_ptr[0];
      set_info->srcB_va       = 0x0;
      set_info->srcA_size     = N * sizeof(short);
      set_info->srcB_size     = 0x0;
      set_info->dstC_size     = N * sizeof(short);
      set_info->p_size        = 0x0;
      set_info->q_size        = 0x0;
      set_info->r_size        = 0x0;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      // ioctl(pl_dma_fd, MEM_CPY_PL_DST, set_info);
      for (int i = 0; i < X * M; i++) {
        set_info->dstC_ptr   = &padded_bias_ptr[i * N];
        set_info->dstC_va    = (uint64_t) &padded_bias_ptr[i * N]; 
        if (ioctl(pl_dma_fd, MEMCPY_PL2PL, set_info) < 0) {
          printf("PL --> PL ERROR DMA \n");
          exit(-1);
        }
      }
#endif
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &padded_bias_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &padded_bias_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      if (ioctl(pl_dma_fd, ELEWISE_MUL, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }
      pim_free(padded_bias_ptr);
    } else {
      set_info->srcA_ptr      = &B_PL_ptr[0];
      set_info->srcB_ptr      = &A_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &B_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &A_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;
      
      if (ioctl(pl_dma_fd, BIAS_MUL, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   
    }

    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur += exe_dur;
    ctx->dma_tx += dma_tx;

    return Status::OK();
  }
  else if (a_dim == 1 && b_dim == 3) {
    int64_t X = B->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = B->Shape()[1];
    int64_t N = B->Shape()[2];

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
    ctx->tensor_alloc_dur = tensor_alloc_dur;    

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X, M, N}));

    bool need_pad = N % 512 != 0;
    // bool need_pad = true;

    auto begin_exe = std::chrono::high_resolution_clock::now();

    if (need_pad) {
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_bias_ptr = (Bfloat16 *)pim_malloc(X * M * N * sizeof(Bfloat16));
#if defined (TARGET_X86)
      for (int i = 0; i < X * M; i++) {
        for (int j = 0; j < N; j++) {
          padded_bias_ptr[i * N + j] = A_PL_ptr[j];
        }
      }
#elif defined (TARGET_ARM)
      set_info->srcA_ptr      = &A_PL_ptr[0]; 
      set_info->srcB_ptr      = NULL;
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = 0x0;
      set_info->srcA_size     = N * sizeof(short);
      set_info->srcB_size     = 0x0;
      set_info->dstC_size     = N * sizeof(short);
      set_info->p_size        = 0x0;
      set_info->q_size        = 0x0;
      set_info->r_size        = 0x0;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      // ioctl(pl_dma_fd, MEM_CPY_PL_DST, set_info);
      for (int i = 0; i < X * M; i++) {
        set_info->dstC_ptr   = &padded_bias_ptr[i * N];
        set_info->dstC_va    = (uint64_t) &padded_bias_ptr[i * N]; 
        if (ioctl(pl_dma_fd, MEMCPY_PL2PL, set_info) < 0) {
          printf("PL --> PL ERROR DMA \n");
          exit(-1);
        }
      }
#endif
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      set_info->srcA_ptr      = &padded_bias_ptr[0];
      set_info->srcB_ptr      = &B_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &padded_bias_ptr[0];
      set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      if (ioctl(pl_dma_fd, ELEWISE_MUL, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }

      pim_free(padded_bias_ptr);      

    } else {
      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &B_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      if (ioctl(pl_dma_fd, BIAS_MUL, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      } 
    }

    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur += exe_dur;
    ctx->dma_tx += dma_tx;

    return Status::OK();
  }  
  else if (a_dim == 3 && b_dim == 0) {
    int64_t X = A->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = A->Shape()[1];
    int64_t N = A->Shape()[2];
    int64_t N_org = A->GetOriginalShape()[2];

    if (N == 1) {
      bool need_pad = M % 512 != 0;
      // Ignore N.
      int M_pad = need_pad ? 512 * (M / 512 + 1) : M;
      Bfloat16* padded_A_PL_ptr = nullptr;
      if (need_pad) {
        padded_A_PL_ptr = (Bfloat16 *)pim_malloc(X * M_pad * sizeof(Bfloat16));
        for (int m = 0; m < M; m++) {
          padded_A_PL_ptr[m] = A_PL_ptr[m];
        }
      } 
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_B_PL_ptr = (Bfloat16 *)pim_malloc(1 * M_pad * sizeof(Bfloat16));
      for (int m = 0; m < M; m++) {
        padded_B_PL_ptr[m] = B_PL_ptr[0];
      }
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      Bfloat16* padded_Y_PL_ptr = nullptr;
      if (need_pad) {
        padded_Y_PL_ptr = (Bfloat16 *)pim_malloc(X * M_pad * sizeof(Bfloat16));
      }

      auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
      Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
      long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 

      Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
      Y->SetIsPim();
      Y->SetOriginalShape(TensorShape({X, M, N_org}));

      set_info->srcA_ptr      = need_pad ? padded_A_PL_ptr : A_PL_ptr;
      set_info->srcB_ptr      = padded_B_PL_ptr;
      set_info->dstC_ptr      = need_pad ? padded_Y_PL_ptr : Y_PL_ptr;
      set_info->srcA_va       = need_pad ? (uint64_t) &padded_A_PL_ptr[0] : (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &padded_B_PL_ptr[0];
      set_info->dstC_va       = need_pad ? (uint64_t)&padded_Y_PL_ptr[0] : (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * sizeof(Bfloat16);
      set_info->srcB_size     = M_pad * sizeof(Bfloat16);
      set_info->dstC_size     = M_pad * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = 1;
      set_info->r_size        = M_pad;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, BIAS_MUL, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   

      if (need_pad) {
        for (int m = 0; m < M; m++) {
          Y_PL_ptr[m] = padded_Y_PL_ptr[m];
        }
      }
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur += exe_dur;
      ctx->dma_tx += dma_tx;

      pim_free(padded_B_PL_ptr);
      if (need_pad) {
        pim_free(padded_A_PL_ptr);
        pim_free(padded_Y_PL_ptr);
      }      

    } else {
      bool need_pad = N % 512 != 0;
      int N_pad = need_pad ? 512 * (N / 512 + 1) : N;

      Bfloat16* padded_A_PL_ptr = nullptr;
      if (need_pad) {
        padded_A_PL_ptr = (Bfloat16 *)pim_malloc(X * M * N_pad * sizeof(Bfloat16));
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            padded_A_PL_ptr[m*N_pad+n] = A_PL_ptr[m*N+n];
          }
        }
      } 

      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_B_PL_ptr = (Bfloat16 *)pim_malloc(1 * N_pad * sizeof(Bfloat16));
      for (int n = 0; n < N_pad; n++) {
        padded_B_PL_ptr[n] = B_PL_ptr[0];
      }
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      Bfloat16* padded_Y_PL_ptr = nullptr;
      if (need_pad) {
        padded_Y_PL_ptr = (Bfloat16 *)pim_malloc(X * M * N_pad * sizeof(Bfloat16));
      }
      auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
      Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
      long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 

      Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
      Y->SetIsPim();
      Y->SetOriginalShape(TensorShape({X, M, N_org}));

      set_info->srcA_ptr      = padded_B_PL_ptr;
      set_info->srcB_ptr      = need_pad ? padded_A_PL_ptr : A_PL_ptr;
      set_info->dstC_ptr      = need_pad ? padded_Y_PL_ptr : Y_PL_ptr;
      set_info->srcA_va       = (uint64_t) &padded_B_PL_ptr[0];
      set_info->srcB_va       = need_pad ? (uint64_t) &padded_A_PL_ptr[0] : (uint64_t) &A_PL_ptr[0];
      set_info->dstC_va       = need_pad ? (uint64_t)&padded_Y_PL_ptr[0] : (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N_pad * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N_pad * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N_pad * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N_pad;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, BIAS_MUL, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   
      if (need_pad) {
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            Y_PL_ptr[m*N+n] = padded_Y_PL_ptr[m*N_pad+n];
          }
        }
      }
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur += exe_dur;
      ctx->dma_tx += dma_tx;

      pim_free(padded_B_PL_ptr);
      if (need_pad) {
        pim_free(padded_A_PL_ptr);
        pim_free(padded_Y_PL_ptr);
      }

    }
    return Status::OK();
  }  
  else {
    ORT_NOT_IMPLEMENTED("NOT IMPLEMENTED SHAPE");
  } // not implemented
} // Mul finished

// Sub start
template <typename T>
Status Sub<T>::Compute(OpKernelContext* ctx) const {

  int pl_dma_fd = pim_args->GetFileDescriptor();

  ioctl_info* set_info = pim_args->GetSetInfo();
  const auto A = ctx->Input<Tensor>(0);
  const auto B = ctx->Input<Tensor>(1);

  auto a_dim = A->Shape().NumDimensions();
  auto b_dim = B->Shape().NumDimensions();  

  const Bfloat16* A_PL_ptr = A->Data<Bfloat16>();
  const Bfloat16* B_PL_ptr = B->Data<Bfloat16>();

  int64_t dma_tx = 0;
  void   *dma_tx_ptr;
  dma_tx_ptr = &dma_tx;

  if (a_dim == 3 && b_dim == 3) {
    int64_t X = A->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = A->Shape()[1];
    int64_t N = A->Shape()[2];
    int64_t N_org = A->GetOriginalShape()[2];

    bool same_shape = A->Shape()[2] == B->Shape()[2] ? true : false;

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc);
    ctx->tensor_alloc_dur = tensor_alloc_dur;

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X, M, N_org}));

    if (same_shape) {

      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &B_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx     = dma_tx;
      set_info->dma_tx_ptr = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, ELEWISE_SUB, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur += exe_dur;
      ctx->dma_tx += dma_tx;  

    } else {
      ORT_ENFORCE(B->Shape()[2] == 1, "ELEWISE B SHAPE 1");
      ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");

      // auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* bias_ptr = (Bfloat16 *)pim_malloc(M * N * sizeof(Bfloat16));
      for (int m = 0; m < M; m++) {
        // std::cout << " " << "m: " << m << std::endl;
        for (int n = 0; n < N; n++) {
          bias_ptr[m * N + n] = B_PL_ptr[m];
          // std::cout << "  " << "n: " << n << " " << bias_ptr[m * N + n] << " " << B_PL_ptr[m] << std::endl;
        } // N
      } // M
      // long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &bias_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &bias_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = M * N * sizeof(Bfloat16);
      set_info->srcB_size     = M * N * sizeof(Bfloat16);
      set_info->dstC_size     = M * N * sizeof(Bfloat16);
      set_info->p_size        = M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, ELEWISE_SUB, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
        }   
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur = exe_dur;
      ctx->dma_tx = dma_tx;       
      
      pim_free(bias_ptr);
    } // same shape

    return Status::OK();
  } 
  else if (a_dim == 3 && b_dim == 1) {
    int64_t X = A->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = A->Shape()[1];
    int64_t N = A->Shape()[2];
    int64_t N_org = A->GetOriginalShape()[2];

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X, M, N_org}));

    bool need_pad = N % 512 != 0;
    // bool need_pad = true;

    auto begin_exe = std::chrono::high_resolution_clock::now();
    
    if (need_pad) {
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_bias_ptr = (Bfloat16 *)pim_malloc(X * M * N * sizeof(Bfloat16));
#if defined (TARGET_X86)
      for (int i = 0; i < X * M; i++) {
        for (int j = 0; j < N; j++) {
          padded_bias_ptr[i * N + j] = B_PL_ptr[j];
        }
      }

#elif defined (TARGET_ARM)
      set_info->srcA_ptr      = &B_PL_ptr[0]; 
      set_info->srcB_ptr      = NULL;
      set_info->srcA_va       = (uint64_t) &B_PL_ptr[0];
      set_info->srcB_va       = 0x0;
      set_info->srcA_size     = N * sizeof(short);
      set_info->srcB_size     = 0x0;
      set_info->dstC_size     = N * sizeof(short);
      set_info->p_size        = 0x0;
      set_info->q_size        = 0x0;
      set_info->r_size        = 0x0;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      // ioctl(pl_dma_fd, MEM_CPY_PL_DST, set_info);
      for (int i = 0; i < X * M; i++) {
        set_info->dstC_ptr   = &padded_bias_ptr[i * N];
        set_info->dstC_va    = (uint64_t) &padded_bias_ptr[i * N]; 
        if (ioctl(pl_dma_fd, MEMCPY_PL2PL, set_info) < 0) {
          printf("PL --> PL ERROR DMA \n");
          exit(-1);
        }
      }
#endif
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &padded_bias_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &padded_bias_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      if (ioctl(pl_dma_fd, ELEWISE_SUB, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }
      pim_free(padded_bias_ptr);

    } else {
      set_info->srcA_ptr      = &B_PL_ptr[0];
      set_info->srcB_ptr      = &A_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &B_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &A_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;
      
      if (ioctl(pl_dma_fd, BIAS_SUB, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   
    }

    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur += exe_dur;
    ctx->dma_tx += dma_tx;

    return Status::OK();
  }
  else if (a_dim == 1 && b_dim == 3) {
    int64_t X = B->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = B->Shape()[1];
    int64_t N = B->Shape()[2];

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
    ctx->tensor_alloc_dur = tensor_alloc_dur;    

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X, M, N}));

    bool need_pad = N % 512 != 0;
    // bool need_pad = true;

    auto begin_exe = std::chrono::high_resolution_clock::now();

    if (need_pad) {
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_bias_ptr = (Bfloat16 *)pim_malloc(X * M * N * sizeof(Bfloat16));
#if defined (TARGET_X86)
      for (int i = 0; i < X * M; i++) {
        for (int j = 0; j < N; j++) {
          padded_bias_ptr[i * N + j] = A_PL_ptr[j];
        }
      }
#elif defined (TARGET_ARM)
      set_info->srcA_ptr      = &A_PL_ptr[0]; 
      set_info->srcB_ptr      = NULL;
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = 0x0;
      set_info->srcA_size     = N * sizeof(short);
      set_info->srcB_size     = 0x0;
      set_info->dstC_size     = N * sizeof(short);
      set_info->p_size        = 0x0;
      set_info->q_size        = 0x0;
      set_info->r_size        = 0x0;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      // ioctl(pl_dma_fd, MEM_CPY_PL_DST, set_info);
      for (int i = 0; i < X * M; i++) {
        set_info->dstC_ptr   = &padded_bias_ptr[i * N];
        set_info->dstC_va    = (uint64_t) &padded_bias_ptr[i * N]; 
        if (ioctl(pl_dma_fd, MEMCPY_PL2PL, set_info) < 0) {
          printf("PL --> PL ERROR DMA \n");
          exit(-1);
        }
      }
#endif
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      set_info->srcA_ptr      = &padded_bias_ptr[0];
      set_info->srcB_ptr      = &B_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &padded_bias_ptr[0];
      set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      if (ioctl(pl_dma_fd, ELEWISE_SUB, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }

      pim_free(padded_bias_ptr);      

    } else {
      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &B_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      if (ioctl(pl_dma_fd, BIAS_SUB, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      } 
    }

    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur += exe_dur;
    ctx->dma_tx += dma_tx;

    return Status::OK();
  }  
  else if (a_dim == 3 && b_dim == 0) {
    int64_t X = A->Shape()[0];
    ORT_ENFORCE(X == 1, "ONLY BATCH 1 NOW");
    int64_t M = A->Shape()[1];
    int64_t N = A->Shape()[2];
    int64_t N_org = A->GetOriginalShape()[2];

    if (N == 1) {
      bool need_pad = M % 512 != 0;
      // Ignore N.
      int M_pad = need_pad ? 512 * (M / 512 + 1) : M;
      Bfloat16* padded_A_PL_ptr = nullptr;
      if (need_pad) {
        padded_A_PL_ptr = (Bfloat16 *)pim_malloc(X * M_pad * sizeof(Bfloat16));
        for (int m = 0; m < M; m++) {
          padded_A_PL_ptr[m] = A_PL_ptr[m];
        }
      } 
      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_B_PL_ptr = (Bfloat16 *)pim_malloc(1 * M_pad * sizeof(Bfloat16));
      for (int m = 0; m < M; m++) {
        padded_B_PL_ptr[m] = B_PL_ptr[0];
      }
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      Bfloat16* padded_Y_PL_ptr = nullptr;
      if (need_pad) {
        padded_Y_PL_ptr = (Bfloat16 *)pim_malloc(X * M_pad * sizeof(Bfloat16));
      }

      auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
      Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
      long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 

      Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
      Y->SetIsPim();
      Y->SetOriginalShape(TensorShape({X, M, N_org}));

      set_info->srcA_ptr      = need_pad ? padded_A_PL_ptr : A_PL_ptr;
      set_info->srcB_ptr      = padded_B_PL_ptr;
      set_info->dstC_ptr      = need_pad ? padded_Y_PL_ptr : Y_PL_ptr;
      set_info->srcA_va       = need_pad ? (uint64_t) &padded_A_PL_ptr[0] : (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &padded_B_PL_ptr[0];
      set_info->dstC_va       = need_pad ? (uint64_t)&padded_Y_PL_ptr[0] : (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * sizeof(Bfloat16);
      set_info->srcB_size     = M_pad * sizeof(Bfloat16);
      set_info->dstC_size     = M_pad * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = 1;
      set_info->r_size        = M_pad;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, BIAS_SUB, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   

      if (need_pad) {
        for (int m = 0; m < M; m++) {
          Y_PL_ptr[m] = padded_Y_PL_ptr[m];
        }
      }
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur += exe_dur;
      ctx->dma_tx += dma_tx;

      pim_free(padded_B_PL_ptr);
      if (need_pad) {
        pim_free(padded_A_PL_ptr);
        pim_free(padded_Y_PL_ptr);
      }      

    } else {
      bool need_pad = N % 512 != 0;
      int N_pad = need_pad ? 512 * (N / 512 + 1) : N;

      Bfloat16* padded_A_PL_ptr = nullptr;
      if (need_pad) {
        padded_A_PL_ptr = (Bfloat16 *)pim_malloc(X * M * N_pad * sizeof(Bfloat16));
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            padded_A_PL_ptr[m*N_pad+n] = A_PL_ptr[m*N+n];
          }
        }
      } 

      auto bias_pad_overhead = std::chrono::high_resolution_clock::now();
      Bfloat16* padded_B_PL_ptr = (Bfloat16 *)pim_malloc(1 * N_pad * sizeof(Bfloat16));
      for (int n = 0; n < N_pad; n++) {
        padded_B_PL_ptr[n] = B_PL_ptr[0];
      }
      long long bias_pad_overhead_dur = TimeDiffMicroSeconds(bias_pad_overhead);
      // ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      Bfloat16* padded_Y_PL_ptr = nullptr;
      if (need_pad) {
        padded_Y_PL_ptr = (Bfloat16 *)pim_malloc(X * M * N_pad * sizeof(Bfloat16));
      }
      auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
      Tensor* Y = ctx->Output(0, TensorShape({X, M, N}), true);
      long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 

      Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
      Y->SetIsPim();
      Y->SetOriginalShape(TensorShape({X, M, N_org}));

      set_info->srcA_ptr      = padded_B_PL_ptr;
      set_info->srcB_ptr      = need_pad ? padded_A_PL_ptr : A_PL_ptr;
      set_info->dstC_ptr      = need_pad ? padded_Y_PL_ptr : Y_PL_ptr;
      set_info->srcA_va       = (uint64_t) &padded_B_PL_ptr[0];
      set_info->srcB_va       = need_pad ? (uint64_t) &padded_A_PL_ptr[0] : (uint64_t) &A_PL_ptr[0];
      set_info->dstC_va       = need_pad ? (uint64_t)&padded_Y_PL_ptr[0] : (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N_pad * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N_pad * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N_pad * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N_pad;

      set_info->dma_tx        = dma_tx;
      set_info->dma_tx_ptr    = dma_tx_ptr;

      auto begin_exe = std::chrono::high_resolution_clock::now();
      if (ioctl(pl_dma_fd, BIAS_SUB, set_info) < 0) {
          printf("ERROR DMA \n");
          exit(-1);
      }   
      if (need_pad) {
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            Y_PL_ptr[m*N+n] = padded_Y_PL_ptr[m*N_pad+n];
          }
        }
      }
      long long exe_dur = TimeDiffMicroSeconds(begin_exe);
      ctx->exe_dur += exe_dur;
      ctx->dma_tx += dma_tx;

      pim_free(padded_B_PL_ptr);
      if (need_pad) {
        pim_free(padded_A_PL_ptr);
        pim_free(padded_Y_PL_ptr);
      }

    }
    return Status::OK();
  }  
  else {
    ORT_NOT_IMPLEMENTED("NOT IMPLEMENTED SHAPE");
  } // not implemented
} // Sub finished

}  // namespace pim
}  // namespace onnxruntime