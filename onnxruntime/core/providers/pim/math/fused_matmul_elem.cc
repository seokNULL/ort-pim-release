// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/pim/math/fused_matmul_elem.h"
#include "core/providers/pim/math/matmul.h"
// #include "core/providers/cpu/math/gemm_matmul_common.h"
// #include "core/util/math_cpuonly.h"
// #include "gemm_helper.h"
// #include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace pim {

#define REGISTER_KERNEL_TYPED(T)                                                      \
ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      FusedMatMulAdd,                                                                 \
      kOnnxDomain,                                                                    \
      9,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      FusedMatMul<T>);                                                             \
ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      FusedMatMulMul,                                                                 \
      kOnnxDomain,                                                                    \
      9,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      FusedMatMul<T>);                                                             \
ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      FusedMatMulSub,                                                                 \
      kOnnxDomain,                                                                    \
      9,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      FusedMatMul<T>);                                                             \
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                        \
      FusedAddAdd,                                                                    \
      kOnnxDomain,                                                                    \
      1,                                                                              \
      8,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      FusedEle<T>);                                                                   \
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                        \
      FusedAddAdd,                                                                    \
      kOnnxDomain,                                                                    \
      9,                                                                              \
      12,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      FusedEle<T>);                                                                   \
ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      FusedAddAdd,                                                                    \
      kOnnxDomain,                                                                    \
      13,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      FusedEle<T>);                                                                   \
ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      FusedAddMul,                                                                    \
      kOnnxDomain,                                                                    \
      9,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      FusedEle<T>);                                                                   \
ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      FusedMulAdd,                                                                    \
      kOnnxDomain,                                                                    \
      9,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      FusedEle<T>);                                                                   \
ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      FusedMulMul,                                                                    \
      kOnnxDomain,                                                                    \
      9,                                                                             \
      T,                                                                              \
      kPimExecutionProvider,                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),       \
      FusedEle<T>);                                                                   \

REGISTER_KERNEL_TYPED(float)

template <typename T>
Status FusedEle<T>::Compute(OpKernelContext* ctx) const {
  //printf("FusedAddAdd\n");
  auto op_name = ctx->GetNodeName();
  std::istringstream tmp(op_name);
  std::string stringBuffer;
  std::vector<std::string> x;
  x.clear();
  while (getline(tmp, stringBuffer, '\''))
      x.push_back(stringBuffer);

  std::cout<<"FusionNode: "<< x[1] << " + " << x[3] << " (" << op_name << ")" << std::endl;
  if (x[1].find("Add") != std::string::npos)
    printf("\t First: Add\n");
  else if (x[1].find("Mul") != std::string::npos)
    printf("\t First: Mul\n");
  else if (x[1].find("Sub") != std::string::npos)
    printf("\t First: Sub\n");

  if (x[3].find("Add") != std::string::npos)
    printf("\t Second: Add\n");
  else if (x[3].find("Mul") != std::string::npos)
    printf("\t Second: Mul\n");
  else if (x[3].find("Sub") != std::string::npos)
    printf("\t Second: Sub\n");

  int X1, M1, N1, X2, M2, N2, X3, M3, N3;
  Tensor* Y;

  const auto A = ctx->Input<Tensor>(0);
  const auto B = ctx->Input<Tensor>(1);
  const auto C = ctx->Input<Tensor>(2);

  const Bfloat16* A_PL_ptr = A->Data<Bfloat16>();
  const Bfloat16* B_PL_ptr = B->Data<Bfloat16>();
  const Bfloat16* C_PL_ptr = C->Data<Bfloat16>();

  auto a_dim = A->Shape().NumDimensions();
  auto b_dim = B->Shape().NumDimensions();    
  auto c_dim = C->Shape().NumDimensions();

  X1 = A->Shape()[0];
  M1 = A->Shape()[1];
  N1 = A->Shape()[2];
  int N1_orig = A->GetOriginalShape()[2];  
  /* 
   * For scalar tensor 
   * TODO: Determining as a scalar tensor using only dimension can cause bugs.
   */
  if (b_dim != 0) {
    X2 = B->Shape()[0];
    M2 = B->Shape()[1];
    N2 = B->Shape()[2];
  } else {
    X2 = 0;
    M2 = 0;
    N2 = 0;
  }
  X3 = C->Shape()[0];
  M3 = C->Shape()[1];
  N3 = C->Shape()[2];

  printf("\t Shape A[%lu] = (%d x %d x %d) \n", a_dim, X1, M1, N1);
  printf("\t Shape B[%lu] = (%d x %d x %d) \n", b_dim, X2, M2, N2);
  printf("\t Shape C[%lu] = (%d x %d x %d) \n", c_dim, X3, M3, N3);

  if (a_dim == 3 && b_dim == 3) {

    auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
    Tensor* Y = ctx->Output(0, TensorShape({X1, M1, N1_orig}), true);
    long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc);
    ctx->tensor_alloc_dur = tensor_alloc_dur;

    Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
    Y->SetOriginalShape(TensorShape({X1, M1, N1_orig}));

    auto y_dim = Y->Shape().NumDimensions();
    int X0 = Y->Shape()[0];
    int M0 = Y->Shape()[1];
    int N0 = Y->Shape()[2];
    printf("\t Shape Y[%lu] = (%d x %d x %d) \n", y_dim, X0, M0, N0);

    set_info->srcA_ptr      = &A_PL_ptr[0];
    set_info->srcB_ptr      = &B_PL_ptr[0];
    set_info->src2_ptr      = &C_PL_ptr[0];
    set_info->dstC_ptr      = &Y_PL_ptr[0];
    set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
    set_info->srcB_va       = (uint64_t) &B_PL_ptr[0];
    set_info->bias_va       = (uint64_t) &C_PL_ptr[0];
    set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
    set_info->srcA_size     = X1 * M1 * N1 * sizeof(Bfloat16);
    set_info->srcB_size     = X2 * M2 * N2 * sizeof(Bfloat16);
    set_info->src2_size     = X3 * M3 * N3 * sizeof(Bfloat16);
    set_info->dstC_size     = X0 * M0 * N0 * sizeof(Bfloat16);

    set_info->p_size        = X0 * M0;
    set_info->q_size        = N0;
    set_info->r_size        = N0;

    //printf("\t srcA : %lx - %d \n", set_info->srcA_va, set_info->srcA_size);
    //printf("\t srcB : %lx - %d \n", set_info->srcB_va, set_info->srcB_size);
    //printf("\t src2 : %lx - %d \n", set_info->bias_va, set_info->src2_size);
    //printf("\t dstC : %lx - %d \n", set_info->dstC_va, set_info->dstC_size);
    //printf("\t p_size = %d \n", set_info->p_size);
    //printf("\t q_size = %d \n", set_info->q_size);
    //printf("\t r_size = %d \n", set_info->r_size);
    //if (ioctl(pl_dma_fd, ELEWISE_ADD, set_info) < 0) {
    //    printf("ERROR DMA \n");
    //    exit(-1);
    //}   
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

    auto y_dim = Y->Shape().NumDimensions();
    int X0 = Y->Shape()[0];
    int M0 = Y->Shape()[1];
    int N0 = Y->Shape()[2];
    printf("\t Shape Y[%lu] = (%d x %d x %d) \n", y_dim, X0, M0, N0);

    bool need_pad = N % 512 != 0;
    // bool need_pad = true;    
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
      ctx->tensor_alloc_dur += bias_pad_overhead_dur;

      set_info->srcA_ptr      = &A_PL_ptr[0];
      set_info->srcB_ptr      = &padded_bias_ptr[0];
      set_info->src2_ptr      = &C_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &A_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &padded_bias_ptr[0];
      set_info->bias_va       = (uint64_t) &C_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = X * M * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->src2_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = X * M;
      set_info->q_size        = N;
      set_info->r_size        = N;
      //printf("\t srcA : %lx - %d \n", set_info->srcA_va, set_info->srcA_size);
      //printf("\t srcB : %lx - %d \n", set_info->srcB_va, set_info->srcB_size);
      //printf("\t src2 : %lx - %d \n", set_info->bias_va, set_info->src2_size);
      //printf("\t dstC : %lx - %d \n", set_info->dstC_va, set_info->dstC_size);
      //printf("\t p_size = %d \n", set_info->p_size);
      //printf("\t q_size = %d \n", set_info->q_size);
      //printf("\t r_size = %d \n", set_info->r_size);
      //if (ioctl(pl_dma_fd, ELEWISE_ADD, set_info) < 0) {
      //    printf("ERROR DMA \n");
      //    exit(-1);
      //}
      pim_free(padded_bias_ptr);
    } // need_pad
    else {
      set_info->srcA_ptr      = &B_PL_ptr[0];
      set_info->srcB_ptr      = &A_PL_ptr[0];
      set_info->src2_ptr      = &C_PL_ptr[0];
      set_info->dstC_ptr      = &Y_PL_ptr[0];
      set_info->srcA_va       = (uint64_t) &B_PL_ptr[0];
      set_info->srcB_va       = (uint64_t) &A_PL_ptr[0];
      set_info->bias_va       = (uint64_t) &C_PL_ptr[0];
      set_info->dstC_va       = (uint64_t) &Y_PL_ptr[0];
      set_info->srcA_size     = 1 * N * sizeof(Bfloat16);
      set_info->srcB_size     = X * M * N * sizeof(Bfloat16);
      set_info->src2_size     = X * M * N * sizeof(Bfloat16);
      set_info->dstC_size     = X * M * N * sizeof(Bfloat16);
      set_info->p_size        = 1;
      set_info->q_size        = M;
      set_info->r_size        = N;

      printf("\t srcA : %lx - %d \n", set_info->srcA_va, set_info->srcA_size);
      printf("\t srcB : %lx - %d \n", set_info->srcB_va, set_info->srcB_size);
      printf("\t src2 : %lx - %d \n", set_info->bias_va, set_info->src2_size);
      printf("\t dstC : %lx - %d \n", set_info->dstC_va, set_info->dstC_size);
      printf("\t p_size = %d \n", set_info->p_size);
      printf("\t q_size = %d \n", set_info->q_size);
      printf("\t r_size = %d \n", set_info->r_size);
      //if (ioctl(pl_dma_fd, BIAS_ADD, set_info) < 0) {
      //    printf("ERROR DMA \n");
      //    exit(-1);
      //}
    }
  } // a_dim==3 and b_dim==1
  else {
    printf("A_DIM:%d, B_DIM:%d \n", a_dim, b_dim);
    ORT_NOT_IMPLEMENTED("NOT IMPLEMENTED dimension");
  }
  return Status::OK();
}

// __TODO
template <typename T>
Status FusedMatMul<T>::Compute(OpKernelContext* ctx) const {
  printf("FusedMatMulAdd\n");
  auto op_name = ctx->GetNodeName();
  int fused_op = 0;
  std::istringstream tmp(op_name);
  std::string stringBuffer;
  std::vector<std::string> x;
  x.clear();
  while (getline(tmp, stringBuffer, '\''))
      x.push_back(stringBuffer);    
  //std::cout<<"FusionNode: "<< x[1] << " + " << x[3] << " (" << op_name << ")" << std::endl;
  if (x[3].find("Add") != std::string::npos) {
    //printf("\t Second: Add\n");
    fused_op = ELEWISE_ADD;
  }
  else if (x[3].find("Sub") != std::string::npos) {
    //printf("\t Second: Sub\n");
    fused_op = ELEWISE_SUB;
  }

  const auto A = ctx->Input<Tensor>(0);
  const auto B = ctx->Input<Tensor>(1);
  const auto C = ctx->Input<Tensor>(2);

  auto a_dim = A->Shape().NumDimensions();
  auto b_dim = B->Shape().NumDimensions();    
  auto c_dim = C->Shape().NumDimensions();

  const Bfloat16* A_PL_ptr = A->Data<Bfloat16>();
  const Bfloat16* B_PL_ptr = B->Data<Bfloat16>();
  const Bfloat16* C_PL_ptr = C->Data<Bfloat16>();

  int X0 = A->Shape()[0];
  int M0 = A->Shape()[1];
  int N0 = A->Shape()[2];

  int X1 = B->Shape()[0];
  int M1 = B->Shape()[1];
  int M1_org = B->GetOriginalShape()[1];
  int N1 = B->Shape()[2];

  int X2 = C->Shape()[0];
  int M2 = C->Shape()[1];
  int N2 = C->Shape()[2];

  auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();  
  Tensor* Y = ctx->Output(0, TensorShape({1, M0, M1_org}), true);
  long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc);
  Bfloat16* Y_PL_ptr = Y->MutableData<Bfloat16>();
  Y->SetIsPim();

  Y->SetOriginalShape(TensorShape({1, M0, M1_org}));
  auto y_dim = Y->Shape().NumDimensions();
  bool layout_changed = B->GetLayoutChanged();
  Bfloat16* aligned_Y_PL_ptr = nullptr;
  if (layout_changed) {
    aligned_Y_PL_ptr = (Bfloat16 *)pim_malloc(X0 * M0 * M1 * sizeof(Bfloat16));
  }

  //printf("FusionNodeCnt (cmd:%d) \n", DECOUPLED_FUSION);
  //printf("\t Shape A[%lu] = (%d x %d x %d) \n", a_dim, X0, M0, N0);
  //printf("\t Shape B[%lu] = (%d x %d (%d) x %d) \n", b_dim, X1, M1, M1_org, N1);
  //printf("\t Shape C[%lu] = (%d x %d x %d) \n", c_dim, X2, M2, N2);
  //printf("\t Shape Y[%lu] = (%d x %d x %d) (%s)\n", y_dim, 1, M0, M1_org, aligned_Y_PL_ptr ? "T":"F");

  // First Matmul operation dimension
  // TODO: Implementation of mat. mul for various dimension
  if ((a_dim != 3) || (b_dim != 2)) {
    ORT_NOT_IMPLEMENTED("NOT IMPLEMENTED SHAPE");      
  }
  // Second Element-wise operation dimension
  if (c_dim == 1) {
    int64_t X = Y->Shape()[0];
    int64_t M = M0;
    int64_t N = M1;
    int64_t N_org = Y->GetOriginalShape()[2];
    bool need_pad = N % 512 != 0;
    //printf("\t Shape Y[%lu] = (%d x %d x %d) (%d) %s, %s\n", y_dim, X, M, N, N_org, need_pad ? "T":"F", layout_changed ? "T":"F");
    // bool need_pad = true;
    if (need_pad) {
      ORT_NOT_IMPLEMENTED("Matrix B must be aligned by 512 !");
    }
    else {
      //bool is_decoupled_compute = (M >= 32)? 1 : 0;
      bool is_decoupled_compute = false;
      //printf("decoupled (%d): %s\n", M, is_decoupled_compute ? "T":"F");

      if(!is_decoupled_compute) {
        // Silent-PIM
        auto out_ptr = layout_changed ? aligned_Y_PL_ptr : Y_PL_ptr;
        set_info->srcA_va   = (uint64_t) &A_PL_ptr[0];
        set_info->srcB_va   = (uint64_t) &B_PL_ptr[0];
        set_info->bias_va   = (uint64_t) &C_PL_ptr[0];
        set_info->dstC_va   = (uint64_t) &out_ptr[0];
        set_info->p_size    = X0 * M0;
        set_info->q_size    = X1;
        set_info->r_size    = M1;
        set_info->fused_op  = fused_op;
        //printf("\t srcA : %lx - %d \n", set_info->srcA_va, set_info->srcA_size);
        //printf("\t srcB : %lx - %d \n", set_info->srcB_va, set_info->srcB_size);
        //printf("\t src2 : %lx - %d \n", set_info->bias_va, set_info->src2_size);
        //printf("\t dstC : %lx - %d \n", set_info->dstC_va, set_info->dstC_size);
        //printf("\t p_size = %d \n", set_info->p_size);
        //printf("\t q_size = %d \n", set_info->q_size);
        //printf("\t r_size = %d \n", set_info->r_size);
        if (ioctl(pl_dma_fd, MATMUL_FUSION, set_info) < 0) {
            printf("ERROR DMA \n");
            exit(-1);
        }
      } else {
        // Decoupled-PIM
        auto begin_tensor_alloc = std::chrono::high_resolution_clock::now();
        Bfloat16* layouted_in = nullptr;
        layouted_in = (Bfloat16 *)pim_malloc(X0 * M0 * X1 * sizeof(Bfloat16));
        int a_row_pad;
        a_row_pad = decoupled_PIM_input_layout (A_PL_ptr, layouted_in, M0, X1);

        Bfloat16* result_Y_PL_ptr = nullptr;
        result_Y_PL_ptr = (Bfloat16 *)pim_malloc(X0 * M0 * M1 * sizeof(Bfloat16));

        long long tensor_alloc_dur = TimeDiffMicroSeconds(begin_tensor_alloc); 
        ctx->tensor_alloc_dur += tensor_alloc_dur;

        //printf("\t DCP_A[%lu] = (%d x %d x %d) (%d) \n", a_dim, X0, M0, X1, a_row_pad);
        //printf("\t DCP_Y[%lu] = (%d x %d x %d) \n", y_dim, X0, M0, M1);
        set_info->srcA_va   = (uint64_t) &layouted_in[0];
        set_info->srcB_va   = (uint64_t) &B_PL_ptr[0];
        set_info->bias_va   = (uint64_t) &C_PL_ptr[0];
        set_info->dstC_va   = (uint64_t) &result_Y_PL_ptr[0];
        set_info->p_size    = X0 * M0;
        set_info->q_size    = X1;
        set_info->r_size    = M1;
        set_info->fused_op  = fused_op;
        //printf("\t srcA : %lx - %d \n", set_info->srcA_va, set_info->srcA_size);
        //printf("\t srcB : %lx - %d \n", set_info->srcB_va, set_info->srcB_size);
        //printf("\t src2 : %lx - %d \n", set_info->bias_va, set_info->src2_size);
        //printf("\t dstC : %lx - %d \n", set_info->dstC_va, set_info->dstC_size);
        //printf("\t p_size = %d \n", set_info->p_size);
        //printf("\t q_size = %d \n", set_info->q_size);
        //printf("\t r_size = %d \n", set_info->r_size);
        if (ioctl(pl_dma_fd, DECOUPLED_FUSION, set_info) < 0) {
            printf("ERROR DMA \n");
            exit(-1);
        }
        auto out_ptr = layout_changed ? aligned_Y_PL_ptr : Y_PL_ptr;

        auto begin_mem_cpy = std::chrono::high_resolution_clock::now();
        decoupled_PIM_result_layout(result_Y_PL_ptr, out_ptr, M, N);
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
        if (layout_changed) {
            pim_free(aligned_Y_PL_ptr);
        }
        pim_free(layouted_in);
        pim_free(result_Y_PL_ptr);

        long long dur_memcpy = TimeDiffMicroSeconds(begin_mem_cpy);
        ctx->tensor_alloc_dur += dur_memcpy;
      } // Decoupled-PIM
    }
    return Status::OK();
  } // c_dim == 1
  else {
    // TODO: Implementation of ele-wise ops. for various dimension
    printf("C_DIM:%d", c_dim);
    ORT_NOT_IMPLEMENTED("Only C dimension 1 is implemented");
  }

  return Status::OK();
}

}  // namespace pim

int decoupled_PIM_input_layout2 (const Bfloat16* src,  Bfloat16* dst, int row_size, int col_size) 
{
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

/*
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
*/
}

void decoupled_PIM_result_layout2 (Bfloat16* src, Bfloat16* dst, int row_size, int col_size) 
{
/*  
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
*/
}

}  // namespace onnxruntime
