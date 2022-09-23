// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the Korea Univ, COMMIT Lab License.

#include "core/providers/pim/math/lut_ops.h"
#include "core/framework/data_types_internal.h"
#include "core/util/math.h"
#include "core/providers/cpu/tensor/utils.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/framework/data_types_internal.h"

#include <cmath>
#include "lut_helper.h"


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

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Div, 7, 12, float, Div);
REG_ELEMENTWISE_TYPED_KERNEL(Div, 13, float, Div);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Log, 6, 12, float, Log);
REG_ELEMENTWISE_TYPED_KERNEL(Log, 13, float, Log);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Neg, 6, 12, float, Neg);
REG_ELEMENTWISE_TYPED_KERNEL(Neg, 13, float, Neg);

REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Abs, 6, 12, float, Abs);
REG_ELEMENTWISE_TYPED_KERNEL(Abs, 13, float, Abs);


//Pow functions need to be checked for opset version error!! 
//Now we temporally enlarged kernel start&end version to 7~13. 
//Any exception case with lower opset version can be happened
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Pow, 7, 13, float, Pow);
// REG_ELEMENTWISE_TYPED_KERNEL(Pow, 12, float, Pow);
// REG_ELEMENTWISE_VERSIONED_KERNEL_NONT(Pow, 7, 11, Pow, float);
// To reduce templetization we choose to support the below types for both
// base and the exponent. This gives us 16 permutations
// REG_ELEMENTWISE_VERSIONED_KERNEL_NONT(Pow, 12, 12, Pow, int32_t, int64_t, float);
// REG_ELEMENTWISE_KERNEL_NONT(Pow, 13, Pow, int32_t, int64_t, float);


REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Relu, 6, 12, float, Relu);
REG_ELEMENTWISE_TYPED_KERNEL(Relu, 13, float, Relu);
REG_ELEMENTWISE_VERSIONED_TYPED_KERNEL(Sigmoid, 6, 12, float, Sigmoid);
REG_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 13, float, Sigmoid);



template <typename T>
Status Erf<T>::Compute(OpKernelContext* ctx) const {

  const auto* A = ctx->Input<Tensor>(0);
  const auto* B = ctx->Input<Tensor>(1);
  const auto* C = ctx->Input<Tensor>(2);

  // Bias could be missing. Treat as scalar 0 if that is the case.
  LutHelper helper(A->Shape(), B->Shape(), "Erf");

  if (!helper.State().IsOK())
    return helper.State();

  int64_t M = helper.M();
  int64_t N = helper.N();
  int64_t K = 0;

  auto Y = ctx->Output(0, {M, N});
  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();
  T* y_data = Y->MutableData<T>();
  ComputeLut(M, N, K);


return Status::OK();
}

template <typename T>
Status Tanh<T>::Compute(OpKernelContext* ctx) const {


return Status::OK();
}


template <typename T>
Status Div<T>::Compute(OpKernelContext* ctx) const {


return Status::OK();
}

template <typename T>
Status Pow<T>::Compute(OpKernelContext* ctx) const {


return Status::OK();
}


template <typename T>
Status Relu<T>::Compute(OpKernelContext* ctx) const {


return Status::OK();
}

template <typename T>
Status Sigmoid<T>::Compute(OpKernelContext* ctx) const {


return Status::OK();
}

template <typename T>
Status Sqrt<T>::Compute(OpKernelContext* ctx) const {


return Status::OK();
}

template <typename T>
Status Neg<T>::Compute(OpKernelContext* ctx) const {


return Status::OK();
}

template <typename T>
Status Abs<T>::Compute(OpKernelContext* ctx) const {


return Status::OK();
}

template <typename T>
Status Log<T>::Compute(OpKernelContext* ctx) const {


return Status::OK();
}


void ComputeLut(int64_t M, int64_t N, int64_t fileptr) {
  return;
}

}  // namespace pim
}  // namespace onnxruntime


