#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft Corporation. All rights reserved.  
# Licensed under the MIT License.

# # ONNX Runtime: Tutorial for Nuphar execution provider
# **Accelerating model inference via compiler, using Docker Images for ONNX Runtime with Nuphar**
# 
# This example shows how to accelerate model inference using Nuphar, an execution provider that leverages just-in-time compilation to generate optimized executables.
# 
# For more background about Nuphar, please check [Nuphar-ExecutionProvider.md](https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/Nuphar-ExecutionProvider.md) and its [build instructions](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#nuphar).
# 
# #### Tutorial Roadmap:
# 1. Prerequistes
# 2. Create and run inference on a simple ONNX model, and understand how ***compilation*** works in Nuphar.
# 3. Create and run inference on a model using ***LSTM***, run symbolic shape inference, edit LSTM ops to Scan, and check Nuphar speedup.
# 4. ***Quantize*** the LSTM model and check speedup in Nuphar (CPU with AVX2 support is required).
# 5. Working on real models from onnx model zoo: ***BERT squad***, ***GPT-2*** and ***Bidirectional Attention Flow ([BiDAF](https://arxiv.org/pdf/1611.01603))***.
# 6. ***Ahead-Of-Time (AOT) compilation*** to save just-in-time compilation cost on model load.
# 7. Performance tuning for single thread inference.
# 

# ## 1. Prerequistes
# Please make sure you have installed following Python packages. Besides, C++ compiler/linker is required for ahead-of-time compilation. Please make sure you have g++ if running on Linux, or Visual Studio 2017 on Windows.
# 
# For simplicity, you may use [Nuphar docker image](https://github.com/microsoft/onnxruntime/blob/master/dockerfiles/README.md) from Microsoft Container Registry.
# 

# In[1]:


import cpufeature
import hashlib
import numpy as np
import onnx
from onnx import helper, numpy_helper
import os
from timeit import default_timer as timer
import shutil
import subprocess
import sys
import tarfile
import urllib.request

def is_windows():
  return sys.platform.startswith('win')

if is_windows():
  assert shutil.which('cl.exe'), 'Please make sure MSVC compiler and liner are in PATH.'
else:
  assert shutil.which('g++'), 'Please make sure g++ is installed.'

def print_speedup(name, delta_baseline, delta):
    print("{} speed-up {:.2f}%".format(name, 100*(delta_baseline/delta - 1)))
    print("    Baseline: {:.3f} s, Current: {:.3f} s".format(delta_baseline, delta))

def create_cache_dir(cache_dir):
    # remove any stale cache files
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

def md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# And Nuphar package in onnxruntime is required too. Please make sure you are using Nuphar enabled build.

# In[2]:


import onnxruntime
from onnxruntime.nuphar.model_editor import convert_to_scan_model
from onnxruntime.nuphar.model_quantizer import convert_matmul_model
from onnxruntime.nuphar.rnn_benchmark import generate_model, perf_test
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


# ## 2. Create and run inference on a simple ONNX model
# Let's start with a simple model: Y = ((X + X) * X + X) * X + X

# In[3]:


model = onnx.ModelProto()
opset = model.opset_import.add()
opset.domain == 'onnx'
opset.version = 7 # ONNX opset 7 is required for LSTM op later
model.ir_version = onnx.IR_VERSION

graph = model.graph
X = 'input'
Y = 'output'

# declare graph input/ouput with shape [seq, batch, 1024]
dim = 1024
model.graph.input.add().CopyFrom(helper.make_tensor_value_info(X, onnx.TensorProto.FLOAT, ['seq', 'batch', dim]))
model.graph.output.add().CopyFrom(helper.make_tensor_value_info(Y, onnx.TensorProto.FLOAT, ['seq', 'batch', dim]))

# create nodes: Y = ((X + X) * X + X) * X + X
num_nodes = 5
for i in range(num_nodes):
  n = helper.make_node('Mul' if i % 2 else 'Add',
                       [X, X if i == 0 else 'out_'+str(i-1)],
                       ['out_'+str(i) if i < num_nodes - 1 else Y],
                       'node'+str(i))
  model.graph.node.add().CopyFrom(n)

# save the model
simple_model_name = 'simple.onnx'
onnx.save(model, simple_model_name)


# We will use nuphar execution provider to run the inference for the model that we created above, and use settings string to check the generated code.
# 
# Because of the redirection of output, we dump the lowered code from a subprocess to a log file:

# In[4]:


code_to_run = '''
import onnxruntime
s = 'codegen_dump_lower:verbose'
onnxruntime.capi._pybind_state.set_nuphar_settings(s)
sess = onnxruntime.InferenceSession('simple.onnx')
'''

log_file = 'simple_lower.log' 
with open(log_file, "w") as f:
  subprocess.run([sys.executable, '-c', code_to_run], stdout=f, stderr=f)


# The lowered log is similar to C source code, but the whole file is lengthy to show here. Let's just check the last few lines that are most important:

# In[5]:


with open(log_file) as f:
    log_lines = f.readlines()

log_lines[-10:]


# The compiled code showed that the nodes of Add/Mul were fused into a single function, and vectorization was applied in the loop. The fusion was automatically done by the compiler in the Nuphar execution provider, and did not require any manual model editing.
# 
# Next, let's run inference on the model and compare the accuracy and performance with numpy:

# In[6]:


seq = 128
batch = 16
input_data = np.random.rand(seq, batch, dim).astype(np.float32)
sess = onnxruntime.InferenceSession(simple_model_name)
simple_feed = {X:input_data}
simple_output = sess.run([], simple_feed)
np_output = ((((input_data + input_data) * input_data) + input_data) * input_data) + input_data
assert np.allclose(simple_output[0], np_output)

simple_repeats = 100
start_ort = timer()
for i in range(simple_repeats):
    sess.run([], simple_feed)
end_ort = timer()
start_np = timer()
for i in range(simple_repeats):
    np_output = ((((input_data + input_data) * input_data) + input_data) * input_data) + input_data
end_np = timer()
print_speedup('Fusion', end_np - start_np, end_ort - start_ort)


# ## 3. Create and run inference on a model using LSTM
# Now, let's take one step further to work on a 4-layer LSTM model, created from onnxruntime.nuphar.rnn_benchmark module.

# In[7]:


lstm_model = 'LSTMx4.onnx'
input_dim = 256
hidden_dim = 1024
generate_model('lstm', input_dim, hidden_dim, bidirectional=False, layers=4, model_name=lstm_model)


# **IMPORTANT**: Nuphar generates code before knowing shapes of input data, unlike other execution providers that do runtime shape inference. Thus, shape inference information is critical for compiler optimizations in Nuphar. To do that, we run symbolic shape inference on the model. Symbolic shape inference is based on the ONNX shape inference, and enhanced by sympy to better handle Shape/ConstantOfShape/etc. ops using symbolic computation.
# 
# **IMPORTANT**: When running multi-threaded inference, Nuphar currently uses TVM's parallel schedule with has its own thread pool that's compatible with OpenMP and MKLML. The TVM thread pool has not been integrated with ONNX runtime thread pool, so intra_op_num_threads won't control it. Please make sure the build is with OpenMP or MKLML, and use OMP_NUM_THREADS to control thread pool.

# In[8]:


onnx.save(SymbolicShapeInference.infer_shapes(onnx.load(lstm_model)), lstm_model)


# Now, let's check baseline performance on the generated model, using CPU execution provider.

# In[9]:


sess_baseline = onnxruntime.InferenceSession(lstm_model, providers=['CPUExecutionProvider'])
seq = 128
input_data = np.random.rand(seq, 1, input_dim).astype(np.float32)
lstm_feed = {sess_baseline.get_inputs()[0].name:input_data}
lstm_output = sess_baseline.run([], lstm_feed)


# To run RNN models in Nuphar execution provider efficiently, LSTM/GRU/RNN ops need to be converted to Scan ops. This is because Scan is more flexible, and supports quantized RNNs.

# In[10]:


lstm_scan_model = 'Scan_LSTMx4.onnx'
convert_to_scan_model(lstm_model, lstm_scan_model)


# After conversion, let's compare performance and accuracy with baseline:

# In[11]:


sess_nuphar = onnxruntime.InferenceSession(lstm_scan_model)
output_nuphar = sess_nuphar.run([], lstm_feed)
assert np.allclose(lstm_output[0], output_nuphar[0])

lstm_repeats = 10
start_lstm_baseline = timer()
for i in range(lstm_repeats):
    sess_baseline.run([], lstm_feed)
end_lstm_baseline = timer()

start_nuphar = timer()
for i in range(lstm_repeats):
    sess_nuphar.run([], lstm_feed)
end_nuphar = timer()

print_speedup('Nuphar Scan', end_lstm_baseline - start_lstm_baseline, end_nuphar - start_nuphar)


# ## 4. Quantize the LSTM model
# Let's get more speed-ups from Nuphar by quantizing the floating point GEMM/GEMV in LSTM model to int8 GEMM/GEMV.
# 
# **NOTE:** For inference speed of quantizated model, a CPU with AVX2 instructions is preferred.

# In[12]:


cpufeature.CPUFeature['AVX2'] or 'No AVX2, quantization model might be slow'


# We can use onnxruntime.nuphar.model_quantizer to quantize floating point GEMM/GEMVs. Assuming GEMM/GEMV takes form of input * weights, weights are statically quantized per-column, and inputs are dynamically quantized per-row.

# In[13]:


lstm_quantized_model = 'Scan_LSTMx4_int8.onnx'
convert_matmul_model(lstm_scan_model, lstm_quantized_model)


# Now run the quantized model, and check accuracy. Please note that quantization may cause accuracy loss, so we relax the comparison threshold a bit.

# In[14]:


sess_quantized = onnxruntime.InferenceSession(lstm_quantized_model)
output_quantized = sess_quantized.run([], lstm_feed)
assert np.allclose(lstm_output[0], output_quantized[0], rtol=1e-3, atol=1e-3)


# Now check quantized model performance:

# In[15]:


start_quantized = timer()
for i in range(lstm_repeats):
    sess_quantized.run([], lstm_feed)
end_quantized = timer()

print_speedup('Quantization', end_nuphar - start_nuphar, end_quantized - start_quantized)


# To check RNN quantization performance, please use rnn_benchmark.perf_test.

# In[16]:


rnn_type = 'lstm'                         # could be 'lstm', 'gru' or 'rnn'
num_threads = cpufeature.CPUFeature['num_physical_cores'] # no hyper thread
input_dim = 80                            # size of input dimension
hidden_dim = 512                          # size of hidden dimension in cell
bidirectional = True                      # specify RNN being bidirectional
layers = 6                                # number of stacked RNN layers
seq_len = 40                              # length of sequence
batch_size = 1                            # size of batch
original_ms, scan_ms, int8_ms = perf_test(rnn_type, num_threads, input_dim, hidden_dim, bidirectional, layers, seq_len, batch_size)
print_speedup('Nuphar Quantization speed up', original_ms / 1000, int8_ms / 1000)


# ## 5. Working on real models
# 
# ### 5.1 BERT Squad
# 
# BERT (Bidirectional Encoder Representations from Transformers) applies Transformers to language modelling. With Nuphar, we may fuse and compile the model to accelerate inference on CPU.
# 
# #### Download model and test data

# In[17]:


# download BERT squad model
cwd = os.getcwd()
bert_model_url = 'https://onnxzoo.blob.core.windows.net/models/opset_10/bert_squad/download_sample_10.tar.gz'
bert_model_local = os.path.join(cwd, 'download_sample_10.tar.gz')
if not os.path.exists(bert_model_local):
  urllib.request.urlretrieve(bert_model_url, bert_model_local)
with tarfile.open(bert_model_local, 'r') as f:
  f.extractall(cwd)


# #### Run symbolic shape inference
# Note that this model has computations like `min(100000, seq_len)` which could be simplified to `seq_len` if we know `seq_len` is not going to be too big. We can do this by setting int_max. Besides, auto_merge is used to make sure the all nodes in the entire model could have shape inferenced by merging symbolic dims when broadcasting.

# In[18]:


bert_model_dir = os.path.join(cwd, 'download_sample_10')
bert_model = os.path.join(bert_model_dir, 'bertsquad10.onnx')
bert_model_with_shape_inference = os.path.join(bert_model_dir, 'bertsquad10_shaped.onnx')

# run symbolic shape inference
onnx.save(SymbolicShapeInference.infer_shapes(onnx.load(bert_model), auto_merge=True, int_max=100000), bert_model_with_shape_inference)


# #### Run inference on original model, using CPU execution provider, with maximum optimization

# In[19]:


sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_baseline = onnxruntime.InferenceSession(bert_model, sess_options=sess_options, providers=['CPUExecutionProvider'])

# load test data
test_data_dir = os.path.join(bert_model_dir, 'test_data_set_1')
tps = [onnx.load_tensor(os.path.join(test_data_dir, 'input_{}.pb'.format(i))) for i in range(len(sess_baseline.get_inputs()))]
bert_feed = {tp.name:numpy_helper.to_array(tp) for tp in tps}
bert_output_baseline = sess_baseline.run([], bert_feed)

bert_repeats = 20
start_bert_baseline = timer()
for i in range(bert_repeats):
    sess_baseline.run([], bert_feed)
end_bert_baseline = timer()


# #### Run inference on the model with symbolic shape inference, using Nuphar execution provider
# First let's check accuracy:

# In[20]:


sess = onnxruntime.InferenceSession(bert_model_with_shape_inference)
output = sess.run([], bert_feed)
assert all([np.allclose(o, ob, atol=1e-4) for o, ob in zip(output, bert_output_baseline)])


# Then check speed:

# In[21]:


start_nuphar = timer()
for i in range(bert_repeats):
    sess.run([], bert_feed)
end_nuphar = timer()

print_speedup('Nuphar BERT squad', end_bert_baseline - start_bert_baseline, end_nuphar - start_nuphar)


# ### 5.2 GPT-2 with fixed batch size
# GPT-2 is a language model using Generative Pre-Trained Transformer for text generation. With Nuphar, we may fuse and compile the model to accelerate inference on CPU.
# 
# #### Download model and test data

# In[22]:


# download GPT-2 model
cwd = os.getcwd()
gpt2_model_url = 'https://onnxzoo.blob.core.windows.net/models/opset_10/GPT2/GPT-2.tar.gz'
gpt2_model_local = os.path.join(cwd, 'GPT-2.tar.gz')
if not os.path.exists(gpt2_model_local):
  urllib.request.urlretrieve(gpt2_model_url, gpt2_model_local)
with tarfile.open(gpt2_model_local, 'r') as f:
  f.extractall(cwd)


# #### Change batch dimension to fixed value, and run symbolic shape inference
# The GPT-2 model from model zoo has a symbolic batch dimension. By replacing it with a fixed value, compiler would be able to generate better code.

# In[23]:


gpt2_model_dir = os.path.join(cwd, 'GPT2')
gpt2_model = os.path.join(gpt2_model_dir, 'model.onnx')

# edit batch dimension from symbolic to int value for better codegen
mp = onnx.load(gpt2_model)
mp.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
onnx.save(mp, gpt2_model)

gpt2_model_with_shape_inference = os.path.join(gpt2_model_dir, 'model_shaped.onnx')

# run symbolic shape inference
onnx.save(SymbolicShapeInference.infer_shapes(onnx.load(gpt2_model), auto_merge=True), gpt2_model_with_shape_inference)


# #### Run inference and compare accuracy/performance to CPU provider

# In[24]:


sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_baseline = onnxruntime.InferenceSession(gpt2_model, sess_options=sess_options, providers=['CPUExecutionProvider'])

# load test data, note the tensor proto name in data does not match model, so override it in feed
input_name = [i.name for i in sess_baseline.get_inputs()][0] # This model only has one input
test_data_dir = os.path.join(gpt2_model_dir, 'test_data_set_0')
tp = onnx.load_tensor(os.path.join(test_data_dir, 'input_0.pb'))
gpt2_feed = {input_name:numpy_helper.to_array(tp).reshape(1,-1)} # the test data missed batch dimension
gpt2_output_baseline = sess_baseline.run([], gpt2_feed)

gpt2_repeats = 100
start_gpt2_baseline = timer()
for i in range(gpt2_repeats):
    sess_baseline.run([], gpt2_feed)
end_gpt2_baseline = timer()

sess = onnxruntime.InferenceSession(gpt2_model_with_shape_inference)
output = sess.run([], gpt2_feed)
assert all([np.allclose(o, ob, atol=1e-4) for o, ob in zip(output, gpt2_output_baseline)])

start_nuphar = timer()
for i in range(gpt2_repeats):
    output = sess.run([], gpt2_feed)
end_nuphar = timer()

print_speedup('Nuphar GPT-2', end_gpt2_baseline - start_gpt2_baseline, end_nuphar - start_nuphar)


# ### 5.3 BiDAF with quantization
# 
# BiDAF is a machine comprehension model that uses LSTMs. The inputs to this model are paragraphs of contexts and queries, and the outputs are start/end indices of words in the contexts that answers the queries.
# 
# First let's download the model:

# In[25]:


# download BiDAF model
cwd = os.getcwd()
bidaf_url = 'https://onnxzoo.blob.core.windows.net/models/opset_9/bidaf/bidaf.tar.gz'
bidaf_local = os.path.join(cwd, 'bidaf.tar.gz')
if not os.path.exists(bidaf_local):
  urllib.request.urlretrieve(bidaf_url, bidaf_local)
with tarfile.open(bidaf_local, 'r') as f:
  f.extractall(cwd)


# Now let's check the performance of the CPU provider:

# In[26]:


bidaf_dir = os.path.join(cwd, 'bidaf')
bidaf = os.path.join(bidaf_dir, 'bidaf.onnx')
sess_baseline = onnxruntime.InferenceSession(bidaf, providers=['CPUExecutionProvider'])
# load test data
test_data_dir = os.path.join(cwd, 'bidaf', 'test_data_set_3')
tps = [onnx.load_tensor(os.path.join(test_data_dir, 'input_{}.pb'.format(i))) for i in range(len(sess_baseline.get_inputs()))]
bidaf_feed = {tp.name:numpy_helper.to_array(tp) for tp in tps}
bidaf_output_baseline = sess_baseline.run([], bidaf_feed)


# The context in this test data:

# In[27]:


' '.join(list(bidaf_feed['context_word'].reshape(-1)))


# The query:

# In[28]:


' '.join(list(bidaf_feed['query_word'].reshape(-1)))


# And the answer:

# In[29]:


' '.join(list(bidaf_feed['context_word'][bidaf_output_baseline[0][0]:bidaf_output_baseline[1][0]+1].reshape(-1)))


# Now put all steps together:

# In[30]:


# editing
bidaf_converted = 'bidaf_mod.onnx'
onnx.save(SymbolicShapeInference.infer_shapes(onnx.load(bidaf)), bidaf_converted)
convert_to_scan_model(bidaf_converted, bidaf_converted)
# When quantizing, there's an only_for_scan option to quantize only the GEMV inside Scan ops.
# This is useful when the input dims of LSTM being much bigger than hidden dims.
# BiDAF has several LSTMs with input dim being 800/1400/etc, while hidden dim is 100.
# So unlike the LSTMx4 model above, we use only_for_scan here
convert_matmul_model(bidaf_converted, bidaf_converted, only_for_scan=True)

# inference and verify accuracy
sess = onnxruntime.InferenceSession(bidaf_converted)
output = sess.run([], bidaf_feed)
assert all([np.allclose(o, ob) for o, ob in zip(output, bidaf_output_baseline)])


# Check performance after all these steps:

# In[31]:


bidaf_repeats = 100
start_bidaf_baseline = timer()
for i in range(bidaf_repeats):
    sess_baseline.run([], bidaf_feed)
end_bidaf_baseline = timer()

start_nuphar = timer()
for i in range(bidaf_repeats):
    sess.run([], bidaf_feed)
end_nuphar = timer()

print_speedup('Nuphar quantized BiDAF', end_bidaf_baseline - start_bidaf_baseline, end_nuphar - start_nuphar)


# The benefit of quantization in BiDAF is not as great as in the LSTM sample above, because BiDAF has relatively small hidden dimensions, which limited the gain from optimization inside Scan ops. However, this model still benefits from fusion/vectorization/etc.

# ## 6. Ahead-Of-Time (AOT) compilation
# Nuphar runs Just-in-time (JIT) compilation when loading models. The compilation may lead to slow cold start. We can use create_shared script to build dll from JIT code and accelerate model loading.

# In[32]:


start_jit = timer()
sess = onnxruntime.InferenceSession(bidaf_converted)
end_jit = timer()
'JIT took {:.3f} seconds'.format(end_jit - start_jit)


# In[33]:


# use settings to enable JIT cache
bidaf_cache_dir = os.path.join(bidaf_dir, 'cache')
create_cache_dir(bidaf_cache_dir)
settings = 'nuphar_cache_path:{}'.format(bidaf_cache_dir)
onnxruntime.capi._pybind_state.set_nuphar_settings(settings)
sess = onnxruntime.InferenceSession(bidaf_converted)


# Now object files of JIT code is stored in cache_dir, let's link them into dll:

# In[34]:


bidaf_cache_versioned_dir = os.path.join(bidaf_cache_dir, os.listdir(bidaf_cache_dir)[0])
# use onnxruntime.nuphar.create_shared module to create dll
subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.create_shared', '--input_dir', bidaf_cache_versioned_dir], check=True)
os.listdir(bidaf_cache_versioned_dir)


# Check the model loading speed-up with AOT dll:

# In[35]:


start_aot = timer()
# NOTE: Nuphar settings string is not sticky. It needs to be reset before creating InferenceSession
settings = 'nuphar_cache_path:{}'.format(bidaf_cache_dir)
onnxruntime.capi._pybind_state.set_nuphar_settings(settings)
sess = onnxruntime.InferenceSession(bidaf_converted)
end_aot = timer()
print_speedup('AOT', end_jit - start_jit, end_aot - start_aot)


# Moreover, Nuphar AOT also supports:
# * Generate JIT cache with AVX/AVX2/AVX-512 and build a AOT dll including support for all these CPUs, which makes deployment easier when targeting different CPUs in one package.
# * Bake model checksum into AOT dll to validate model with given AOT dll.

# In[36]:


# create object files for different CPUs
cache_dir = os.path.join(os.getcwd(), 'lstm_cache')
model_name = lstm_quantized_model
model_checksum = md5(model_name)
repeats = lstm_repeats
feed = lstm_feed
time_baseline = end_lstm_baseline - start_lstm_baseline
multi_isa_so = 'avx_avx2_avx512.so'

create_cache_dir(cache_dir)
settings = 'nuphar_cache_path:{}'.format(cache_dir)
for isa in ['avx512', 'avx2', 'avx']:
    settings_with_isa = settings + ', nuphar_codegen_target:' + isa
    onnxruntime.capi._pybind_state.set_nuphar_settings(settings_with_isa)
    sess = onnxruntime.InferenceSession(model_name)
    cache_versioned_dir = os.path.join(cache_dir, os.listdir(cache_dir)[0])

# link object files to AOT dll
subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.create_shared', '--input_dir', cache_versioned_dir, '--input_model', model_name, '--output_name', multi_isa_so], check=True)

# now load the model with AOT dll
# NOTE: when nuphar_codegen_target is not set, it defaults to current CPU ISA
settings = 'nuphar_cache_path:{}, nuphar_cache_so_name:{}, nuphar_cache_model_checksum:{}, nuphar_cache_force_no_jit:on'.format(cache_dir, multi_isa_so, model_checksum)
onnxruntime.capi._pybind_state.set_nuphar_settings(settings)
sess = onnxruntime.InferenceSession(model_name)

# force to a different ISA which is a subset of current CPU
# NOTE: if an incompatible ISA is used, exception on invalid instructions would be thrown
for valid_isa in ['avx2', 'avx']:
    settings_with_isa = 'nuphar_cache_path:{}, nuphar_cache_so_name:{}, nuphar_cache_model_checksum:{}, nuphar_codegen_target:{}, nuphar_cache_force_no_jit:on'.format(cache_dir, multi_isa_so, model_checksum, valid_isa)
    onnxruntime.capi._pybind_state.set_nuphar_settings(settings_with_isa)
    sess = onnxruntime.InferenceSession(model_name)

    start_nuphar = timer()
    for i in range(repeats):
        sess.run([], feed)
    end_nuphar = timer()

    print_speedup('{} in {}'.format(model_name, valid_isa), time_baseline, end_nuphar - start_nuphar)


# ## 7. Performance tuning for single thread inference.
# By default, Nuphar enables parallel schedule for lower inference latency with multiple threads, when building with MKLML or OpenMP. For some models, user may want to run single-thread inference for better throughput with multiple concurrent inference threads, and turning off parallel schedule may make inference a bit faster in single thread.

# In[37]:


# set OMP_NUM_THREADS to 1 for single thread inference
# this would mak
os.environ['OMP_NUM_THREADS'] = '1'

sess = onnxruntime.InferenceSession(bidaf_converted)
start_baseline = timer()
for i in range(bidaf_repeats):
    sess_baseline.run([], bidaf_feed)
end_baseline = timer()

# use NUPHAR_PARALLEL_MIN_WORKLOADS=0 to turn off parallel schedule, using settings string
# it can be set from environment variable too: os.environ['NUPHAR_PARALLEL_MIN_WORKLOADS'] = '0'
settings = 'nuphar_parallel_min_workloads:0'
onnxruntime.capi._pybind_state.set_nuphar_settings(settings)
sess = onnxruntime.InferenceSession(bidaf_converted)

start = timer()
for i in range(bidaf_repeats):
    sess_baseline.run([], bidaf_feed)
end = timer()
print_speedup('Single thread perf w/o parallel schedule', end_baseline - start_baseline, end - start)

del os.environ['OMP_NUM_THREADS']

