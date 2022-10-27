// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the Korea Univ, COMMIT Lab License.
#include "core/framework/data_types_internal.h"
#include "core/util/math.h"
#include "core/providers/cpu/tensor/utils.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/framework/data_types_internal.h"

#include <cmath>
// #include "lut_helper.h"
#include "core/providers/pim/math/lut_ops.h"
#include "core/providers/pim/pim_execution_provider.h"

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


/*Register operation kernel list */
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Erf, 9, 12, float, Erf);
REG_ELEMENTWISE_TYPED_KERNEL(Erf, 13, float, Erf);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Tanh, 6, 12, float, Tanh);
REG_ELEMENTWISE_TYPED_KERNEL(Tanh, 13, float, Tanh);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sqrt, 6, 12, float, Sqrt);
REG_ELEMENTWISE_TYPED_KERNEL(Sqrt, 13, float, Sqrt);

// REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, float, Div);
// REG_ELEMENTWISE_TYPED_KERNEL(Div, 13, float, Div);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Log, 6, 12, float, Log);
REG_ELEMENTWISE_TYPED_KERNEL(Log, 13, float, Log);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Neg, 6, 12, float, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 13, float, Neg);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, float, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, float, Abs);


//Pow functions need to be checked for opset version error!! 
//Now we temporally enlarged kernel start&end version to 7~13. 
//Any exception case with lower opset version can be happened.
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Pow, 7, 13, float, Pow);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Relu, 6, 12, float, Relu);
REG_ELEMENTWISE_TYPED_KERNEL(Relu, 13, float, Relu);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sigmoid, 6, 12, float, Sigmoid);
REG_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 13, float, Sigmoid);



template <typename T>
Status Erf<T>::Compute(OpKernelContext* ctx) const {
  const onnxruntime::IExecutionProvider* provider = Info().GetExecutionProvider();
  Bfloat16* fx_pim_ptr = provider->ReturnLut(1);
  // Bfloat16* fx_pim_ptr;
  //  for(size_t i=0; i<65536; i++){
  //   std::cout<<"16'h"<<std::hex<<fx_pim_ptr[i]<<std::endl;
  //   }
  int pl_dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* set_info = pim_args->GetSetInfo();
  ComputeLutInternal(pl_dma_fd, set_info, fx_pim_ptr, ctx);

return Status::OK();
}

template <typename T>
Status Tanh<T>::Compute(OpKernelContext* ctx) const {
  const onnxruntime::IExecutionProvider* provider = Info().GetExecutionProvider();
  Bfloat16* fx_pim_ptr = provider->ReturnLut(7);
  // Bfloat16* fx_pim_ptr;
  //  for(size_t i=0; i<65536; i++){
  //   std::cout<<"16'h"<<std::hex<<fx_pim_ptr[i]<<std::endl;
  //   }
  int pl_dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* set_info = pim_args->GetSetInfo();

  ComputeLutInternal(pl_dma_fd, set_info, fx_pim_ptr, ctx);

return Status::OK();
}


// template <typename T>
// Status Div<T>::Compute(OpKernelContext* ctx) const {


// return Status::OK();
// }
union BiasValue {
  uint32_t       u32;
  float          f32;
  struct
    {
      unsigned short back : 16;
      unsigned short front : 16;
    } half_U;
};

template <typename T>
Status Pow<T>::Compute(OpKernelContext* ctx) const {
  const onnxruntime::IExecutionProvider* provider = Info().GetExecutionProvider();

  /*Temporal implement of power function, it needs to be check berfor executing graph!
  if (bias dimension==0) && (store as unsigned short) && (value is float)*/
  BiasValue bias;
  bias.u32 = 0U;
  const auto* Bias = ctx->Input<Tensor>(1);
  // const auto& bias_shape = Bias->Shape();
  // auto        bias_dim   = Bias->Shape().NumDimensions(); 
  // onnxruntime::MLDataType data_type = Bias->DataType();
  const auto* bias_value = Bias->DataRaw();
  bias.half_U.front = reinterpret_cast<const unsigned short*>(bias_value)[0];
  
  
  const auto* X = ctx->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  auto x_dim = X->Shape().NumDimensions();
  const Bfloat16* x_data_ptr = X->Data<Bfloat16>();

  int64_t batch_size, p_size, q_size, f_size;
  Bfloat16* y_data_ptr;
  f_size = (1<<16);//Assume full-precision

  if(x_dim==3){
    batch_size = X->Shape()[0];
     ORT_ENFORCE(batch_size == 1, "ONLY BATCH 1 FOR NOW");
    p_size = X->Shape()[1];
    q_size = X->Shape()[2];
  
    Tensor* Y = ctx->Output(0, TensorShape({batch_size, p_size, q_size}), true);
    y_data_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
  }
  else if(x_dim==2){
    p_size = X->Shape()[0];
    q_size = X->Shape()[1];
  
    //Setting output information (y=f(x))
    // Tensor* Y = ctx->Output(0, x_shape);
    Tensor* Y = ctx->Output(0, TensorShape({p_size, q_size}), true);
    y_data_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();    
  }


  int dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* dma_info = pim_args->GetSetInfo();

  Bfloat16* fx_pim_ptr = provider->ReturnLut(8);

  int64_t dma_tx = 0;
  void   *dma_tx_ptr;
  dma_tx_ptr = &dma_tx;

    dma_info->srcA_ptr      = &x_data_ptr[0];
    dma_info->srcB_ptr      = &fx_pim_ptr[0];
    dma_info->dstC_ptr      = &y_data_ptr[0];
    dma_info->srcA_va       = (uint64_t) &x_data_ptr[0];
    dma_info->srcB_va       = (uint64_t) &fx_pim_ptr[0];
    dma_info->dstC_va       = (uint64_t) &y_data_ptr[0];
    dma_info->srcA_size     = p_size*q_size*sizeof(Bfloat16);
    dma_info->srcB_size     = f_size*sizeof(Bfloat16);
    dma_info->dstC_size     = p_size*q_size*sizeof(Bfloat16);
    dma_info->p_size        = p_size;
    dma_info->q_size        = q_size;
    dma_info->r_size        = q_size;
    
    dma_info->dma_tx     = dma_tx;
    dma_info->dma_tx_ptr = dma_tx_ptr;

    auto begin_exe = std::chrono::high_resolution_clock::now();
    if (ioctl(dma_fd, LUT_OPS, dma_info) < 0) {
        printf("ERROR DMA \n");
        exit(-1);
    }   
    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur = exe_dur;
    ctx->dma_tx = dma_tx;

  

  // ComputeLutInternal(pl_dma_fd, set_info, fx_pim_ptr, ctx);

return Status::OK();
}


template <typename T>
Status Relu<T>::Compute(OpKernelContext* ctx) const {
  const onnxruntime::IExecutionProvider* provider = Info().GetExecutionProvider();
  Bfloat16* fx_pim_ptr = provider->ReturnLut(4);
  // Bfloat16* fx_pim_ptr;
  //  for(size_t i=0; i<65536; i++){
  //   std::cout<<"16'h"<<std::hex<<fx_pim_ptr[i]<<std::endl;
  //   }
  int pl_dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* set_info = pim_args->GetSetInfo();
  ComputeLutInternal(pl_dma_fd, set_info, fx_pim_ptr, ctx);

return Status::OK();
}

template <typename T>
Status Sigmoid<T>::Compute(OpKernelContext* ctx) const {
  const onnxruntime::IExecutionProvider* provider = Info().GetExecutionProvider();
  Bfloat16* fx_pim_ptr = provider->ReturnLut(5);
  // Bfloat16* fx_pim_ptr;
  //  for(size_t i=0; i<65536; i++){
  //   std::cout<<"16'h"<<std::hex<<fx_pim_ptr[i]<<std::endl;
  //   }
  int pl_dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* set_info = pim_args->GetSetInfo();
  ComputeLutInternal(pl_dma_fd, set_info, fx_pim_ptr, ctx);

return Status::OK();
}

template <typename T>
Status Sqrt<T>::Compute(OpKernelContext* ctx) const {
  // const OpKernel*     p_op_kernel = ctx->kernel_;
  // const OpKernelInfo& op_kernel_info = p_op_kernel->Info();
  // auto  PimProvider                  = op_kernel_info.GetExecutionProvider();
  const onnxruntime::IExecutionProvider* provider = Info().GetExecutionProvider();

  Bfloat16* fx_pim_ptr = provider->ReturnLut(6);
  // Bfloat16* fx_pim_ptr;
  //  for(size_t i=0; i<65536; i++){
  //   std::cout<<"16'h"<<std::hex<<fx_pim_ptr[i]<<std::endl;
  //   }
  int pl_dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* set_info = pim_args->GetSetInfo();

  ComputeLutInternal(pl_dma_fd, set_info, fx_pim_ptr, ctx);

return Status::OK();
}

template <typename T>
Status Neg<T>::Compute(OpKernelContext* ctx) const {
  const onnxruntime::IExecutionProvider* provider = Info().GetExecutionProvider();
  Bfloat16* fx_pim_ptr = provider->ReturnLut(3);
  // Bfloat16* fx_pim_ptr;
  //  for(size_t i=0; i<65536; i++){
  //   std::cout<<"16'h"<<std::hex<<fx_pim_ptr[i]<<std::endl;
  //   }
  int pl_dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* set_info = pim_args->GetSetInfo();
  ComputeLutInternal(pl_dma_fd, set_info, fx_pim_ptr, ctx);

return Status::OK();
}

template <typename T>
Status Abs<T>::Compute(OpKernelContext* ctx) const {
  const onnxruntime::IExecutionProvider* provider = Info().GetExecutionProvider();
  Bfloat16* fx_pim_ptr = provider->ReturnLut(0);
  // Bfloat16* fx_pim_ptr;
  //  for(size_t i=0; i<65536; i++){
  //   std::cout<<"16'h"<<std::hex<<fx_pim_ptr[i]<<std::endl;
  //   }
  int pl_dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* set_info = pim_args->GetSetInfo();
  ComputeLutInternal(pl_dma_fd, set_info, fx_pim_ptr, ctx);

return Status::OK();
}

template <typename T>
Status Log<T>::Compute(OpKernelContext* ctx) const {
  const onnxruntime::IExecutionProvider* provider = Info().GetExecutionProvider();
  Bfloat16* fx_pim_ptr = provider->ReturnLut(2);
  // Bfloat16* fx_pim_ptr;
  //  for(size_t i=0; i<65536; i++){
  //   std::cout<<"16'h"<<std::hex<<fx_pim_ptr[i]<<std::endl;
  //   }
  int pl_dma_fd = pim_args->GetFileDescriptor();
  ioctl_info* set_info = pim_args->GetSetInfo();
  ComputeLutInternal(pl_dma_fd, set_info, fx_pim_ptr, ctx);

return Status::OK();
}


void ComputeLutInternal(int dma_fd, ioctl_info* dma_info, Bfloat16* f_ptr, OpKernelContext* ctx) {

  //Setting input information (x and f)
  const auto* X = ctx->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  auto x_dim = X->Shape().NumDimensions();
  const Bfloat16* x_data_ptr = X->Data<Bfloat16>();
  
  int64_t batch_size, p_size, q_size, f_size;
  Bfloat16* y_data_ptr;
  f_size = (1<<16);//Assume full-precision

  if(x_dim==3){
    batch_size = X->Shape()[0];
     ORT_ENFORCE(batch_size == 1, "ONLY BATCH 1 FOR NOW");
    p_size = X->Shape()[1];
    q_size = X->Shape()[2];
  
    Tensor* Y = ctx->Output(0, TensorShape({batch_size, p_size, q_size}), true);
    y_data_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();
  }
  else if(x_dim==2){
    p_size = X->Shape()[0];
    q_size = X->Shape()[1];
  
    //Setting output information (y=f(x))
    // Tensor* Y = ctx->Output(0, x_shape);
    Tensor* Y = ctx->Output(0, TensorShape({p_size, q_size}), true);
    y_data_ptr = Y->MutableData<Bfloat16>();
    Y->SetIsPim();    
  }
  
  int64_t dma_tx = 0;
  void   *dma_tx_ptr;
  dma_tx_ptr = &dma_tx;

    dma_info->srcA_ptr      = &x_data_ptr[0];
    dma_info->srcB_ptr      = &f_ptr[0];
    dma_info->dstC_ptr      = &y_data_ptr[0];
    dma_info->srcA_va       = (uint64_t) &x_data_ptr[0];
    dma_info->srcB_va       = (uint64_t) &f_ptr[0];
    dma_info->dstC_va       = (uint64_t) &y_data_ptr[0];
    dma_info->srcA_size     = p_size*q_size*sizeof(Bfloat16);
    dma_info->srcB_size     = f_size*sizeof(Bfloat16);
    dma_info->dstC_size     = p_size*q_size*sizeof(Bfloat16);
    dma_info->p_size        = p_size;
    dma_info->q_size        = q_size;
    dma_info->r_size        = q_size;
    
    dma_info->dma_tx     = dma_tx;
    dma_info->dma_tx_ptr = dma_tx_ptr;

    auto begin_exe = std::chrono::high_resolution_clock::now();
    if (ioctl(dma_fd, LUT_OPS, dma_info) < 0) {
        printf("ERROR DMA \n");
        exit(-1);
    }   
    long long exe_dur = TimeDiffMicroSeconds(begin_exe);
    ctx->exe_dur = exe_dur;
    ctx->dma_tx = dma_tx;

  return;
}

}  // namespace pim
}  // namespace onnxruntime


