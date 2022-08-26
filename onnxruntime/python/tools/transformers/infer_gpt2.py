import os
import argparse
import json
import onnx
import psutil
import numpy

from onnx import TensorProto
# from onnx_model import OnnxModel


# onnx_model_path = './roberta-sequence-classification-9.onnx'
onnx_model_path = './gpt2-lm-head-10.onnx'
#onnx_model_path = './inception-v2-8.onnx'


#####################################
batch_size = 1
sequence_length = 16

# onnx_model = OnnxModel(onnx.load(onnx_model_path))
onnx_model = onnx.load(onnx_model_path)

org_file_name = onnx_model_path.split('/')[-1]
org_file_path = onnx_model_path.replace(org_file_name, '')

new_file_name = org_file_name.split('.')[0] + '-inferred' + '-' + str(sequence_length) + '.onnx'
#new_file_name = org_file_name.split('.')[0] + '-with-names'+ '.onnx'
new_file_path = org_file_path + new_file_name

# graph = onnx_model.model.graph
graph = onnx_model.graph

#####################################
for inp in graph.input:
    shape_proto = inp.type.tensor_type.shape.dim
    for dim_proto in shape_proto:
        if dim_proto.HasField('dim_param'):
            if dim_proto.dim_param == "input1_dynamic_axes_1":
                dim_proto.ClearField('dim_param')
                dim_proto.dim_value = batch_size
            elif dim_proto.dim_param == "input1_dynamic_axes_2":
                dim_proto.ClearField('dim_param')
                dim_proto.dim_value = batch_size
            elif dim_proto.dim_param == "input1_dynamic_axes_3":
                dim_proto.ClearField('dim_param')
                dim_proto.dim_value = sequence_length              
            else:
                print(dim_proto.dim_param)
                print("ERR")

from onnx import helper, shape_inference
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
# Roberta
#inferred_model = shape_inference.infer_shapes(onnx_model)
# GPT-2
inferred_model = SymbolicShapeInference.infer_shapes(onnx_model)
onnx.save(inferred_model, new_file_path)

#####################################
# op_map = {}
#op_cnt = 0
#for node in graph.node:
#    node.name = node.op_type + '_' + str(op_cnt)
#    op_cnt += 1
#
#onnx.save(onnx_model, new_file_path)


