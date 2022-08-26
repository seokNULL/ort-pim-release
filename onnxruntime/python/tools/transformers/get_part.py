import json
import os
from dataclasses import dataclass
from statistics import median
import time

# roberta
cpu_profile_file = './roberta/cpu_roberta_16.json'
pim_profile_file = './roberta/pim_roberta_16.json'

# # #bert
# # cpu_profile_file = 'cpu_bert.json'
# # pim_profile_file = 'pim_bert.json'

# #googlenet --> DCG creation error
# # cpu_profile_file = 'cpu_googlenet.json'
# # pim_profile_file = 'pim_googlenet.json'

# #resnet
# cpu_profile_file = 'cpu_resnet.json'
# pim_profile_file = 'pim_resnet.json'

## BERT
#cpu_profile_file = './bert/cpu_bert_16.json'
#pim_profile_file = './bert/pim_bert_16.json'

## GPT-2
#cpu_profile_file = './gpt2/cpu_gpt2_16.json'
#pim_profile_file = './gpt2/pim_gpt2_16.json'

@dataclass
class Datafield:
    sink_ptr: tuple
    size: int
    sink_next_ptr: []
    src_ptr: tuple
    src_next_ptr: tuple
    color : str
    explored_edge_list : []
    explored_path : []

@dataclass
class Cost:
    node_cost: int
    edge_cost: int

NUM_OF_NODES = 0 
NUM_OF_DEVICES = 2
prof_data = {}

# BUILD DCG
pim_op_list = []
cpu_prof = os.path.join(cpu_profile_file)
with open(cpu_prof, 'r') as f:
    data = json.load(f)
    for item in data:
        if (item['name'].find('_kernel_time') != -1):
            node_name = item['name'].replace('_kernel_time', '')

            op_name = item['args']['op_name']
            ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()

            prof_data[node_name] = {}
            prof_data[node_name]['op_kind'] = op_name
            prof_data[node_name]['ep_type'] = ep_type
            prof_data[node_name]['graph_index'] = int(item['args']['graph_index'])
            # Output size / sizeof(float)
            prof_data[node_name]['elem_size'] = int(int(item['args']['output_size']) / 4)
            # if op_name == "ReduceMean":
            #     prof_data[node_name]['elem_size'] = int(int(item['args']['activation_size']) / 4)
            incoming_nodes = item['args']['input_nodes'].split()
            outgoing_nodes = item['args']['output_nodes'].split() 

            prof_data[node_name]['src_nodes'] = incoming_nodes
            prof_data[node_name]['sink_nodes'] = outgoing_nodes

            NUM_OF_NODES += 1

# print(NUM_OF_NODES)
# print("PROFILE DATA")
# for key, value in prof_data.items():
#     print(key)
#     print(value)
#     print("\n")

# Collect pim only
pim_prof = os.path.join(pim_profile_file)
with open(pim_prof, 'r') as f:
    data = json.load(f)
    for item in data:
        if (item['name'].find('_kernel_time') != -1):
            node_name = item['name'].replace('_kernel_time', '')
            ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
            if node_name in prof_data.keys() and ep_type == 'pim':
                pim_op_list.append(node_name)

# print("PIM OP LIST")
# print(pim_op_list)

empty_datafield = Datafield(sink_ptr=None, size=None, sink_next_ptr=None, src_ptr=None, src_next_ptr=None, color=None, explored_edge_list=None, explored_path=None)
node_list = list(prof_data.keys())

start_gen_dcg = time.time()

DCG = {}
dev_list = ["cpu", "pim"]
for node_name in prof_data.keys():
    DCG[node_name] = [None] * NUM_OF_DEVICES
    for k in range(NUM_OF_DEVICES):
        if k == 0:
            DCG[node_name][k] = Datafield(None, None, None, None, None, 'white', [], None)
        else:
            DCG[node_name][k] = Datafield(None, None, None, None, None, None, [], None)

edge_dict = {}
source_list = []
for cur_node, attr in prof_data.items():
    # print("Current node: ", cur_node)
    sink_nodes = attr['sink_nodes'][:]
    src_nodes = attr['src_nodes'][:]
    for dev in dev_list:
        dev_idx = dev_list.index(dev)
        # 1. CPU device
        if dev_idx == 0:
            pim_sink_nodes = [x for x in sink_nodes if x in pim_op_list]
            pim_src_nodes = [x for x in src_nodes if x in pim_op_list]
            if pim_sink_nodes:
                DCG[cur_node][dev_idx].sink_ptr = (pim_sink_nodes[0], 1)
                DCG[cur_node][dev_idx].size = attr['elem_size']
                DCG[cur_node][dev_idx].color = 'white'
                # DCG[cur_node][dev_idx].edge_list = []
                # DCG[cur_node][dev_idx].ua_edge = str(attr['elem_size']) + str(dev_idx) + str(1)
                # Add to edge_dict
                edge = ((cur_node, dev_idx), (pim_sink_nodes[0], 1))
                edge_attr = (dev_idx, 1, attr['elem_size'])
                # print("\t", (pim_sink_nodes[0], 1))
                # print("\t", edge_attr)
                edge_dict[edge] = [edge_attr, 'white']
            if pim_src_nodes:
                if (pim_src_nodes[0], 1) not in source_list:
                    DCG[cur_node][dev_idx].src_ptr = (pim_src_nodes[0], 1)
                    # Add to edge_dict
                    edge = ((pim_src_nodes[0], 1), (cur_node, dev_idx))
                    edge_attr = (1, dev_idx, attr['elem_size'])
                    # print("\t", (pim_src_nodes[0], 1))
                    # print("\t", edge_attr)
                    # edge_dict[edge] = [edge_attr, 'white']
                    source_list.append((pim_src_nodes[0], 1))
                if len(pim_src_nodes) > 1:
                    DCG[cur_node][dev_idx].src_next_ptr = []
                    for i in range(1, len(pim_src_nodes)):
                        src_data_field = Datafield(None, None, None, None, None, 'white', None, None)
                        if (pim_src_nodes[i], 1) not in source_list:
                            source_list.append((pim_src_nodes[i], 1))
                            src_data_field.src_ptr = (pim_src_nodes[i], 1)
                            # Add to edge_dict
                            edge = ((pim_src_nodes[i], 1), (cur_node, dev_idx))
                            edge_attr = (1, dev_idx, attr['elem_size'])
                            # print("\t", (pim_src_nodes[i], 1))
                            # print("\t", edge_attr)
                            # edge_dict[edge] = [edge_attr, 'white']
                            
                            # DCG[cur_node][dev_idx].src_next_ptr.append(src_data_field)                            
                            DCG[cur_node][dev_idx].src_next_ptr = (pim_src_nodes[i], 1)                         
        # 2. PIM device
        else:
            if cur_node not in pim_op_list:
                continue
            else:
                DCG[cur_node][dev_idx].color = 'white'
                # DCG[cur_node][dev_idx].edge_list = []  
                if sink_nodes:
                    DCG[cur_node][dev_idx].sink_ptr = (sink_nodes[0], 0)
                    DCG[cur_node][dev_idx].size = attr['elem_size']
                    # DCG[cur_node][dev_idx].ua_edge = str(attr['elem_size']) + str(dev_idx) + str(0)
                    # Add to edge_dict
                    edge = ((cur_node, dev_idx), (sink_nodes[0], 0))
                    edge_attr = (dev_idx, 0, attr['elem_size'])
                    # print("\t", (sink_nodes[0], 0))
                    # print("\t", edge_attr)
                    edge_dict[edge] = [edge_attr, 'white']                              
                if src_nodes:
                    if (src_nodes[0], 0) not in source_list:
                        source_list.append((src_nodes[0], 0))
                         # Add to edge_dict
                        edge = ((src_nodes[0], 0), (cur_node, dev_idx))
                        edge_attr = (0, dev_idx, attr['elem_size'])
                        # print("\t", (src_nodes[0], 0))
                        # print("\t", edge_attr)
                        # edge_dict[edge] = [edge_attr, 'white']
                        DCG[cur_node][dev_idx].src_ptr = (src_nodes[0], 0)
                    if len(src_nodes) > 1:
                        DCG[cur_node][dev_idx].src_next_ptr = []
                        for i in range(1, len(src_nodes)):
                            src_data_field = Datafield(None, None, None, None, None, 'white', None, None)
                            if (src_nodes[i], 0) not in source_list:
                                source_list.append((src_nodes[i], 0))
                                src_data_field.src_ptr = (src_nodes[i], 0)
                                # Add to edge_dict
                                edge = ((src_nodes[i], 0), (cur_node, dev_idx))
                                edge_attr = (0, dev_idx, attr['elem_size'])
                                # print("\t", (src_nodes[i], 0))
                                # print("\t", edge_attr)
                                # edge_dict[edge] = [edge_attr, 'white']
                                
                                # DCG[cur_node][dev_idx].src_next_ptr.append(src_data_field)  
                                DCG[cur_node][dev_idx].src_next_ptr =  (src_nodes[i], 0)

dcg_node_list = list(DCG.keys())

# # CHECK DCG
# for key, value in DCG.items():
#     print("\n", key)
#     for val in value:
#         print(val)

ua_edge_list = []
for key, values in edge_dict.items():
    value = values[0]
    attr = str(value[2]) + str(value[0]) + str(value[1])
    if attr not in ua_edge_list:
        ua_edge_list.append(attr)

# print('DCG generation done! Distinct edge lists:')
# print(ua_edge_list)
# print(len(ua_edge_list))

end_gen_dcg = time.time()
# print("DCG generation time: ", end_gen_dcg - start_gen_dcg)

import numpy as np
explored_edge_list_max = len(ua_edge_list)

dnn_partition = {'CPUExecutionProvider' : [], 'PIMExecutionProvider' : []}
added_node_list = []
###########
## BFS
###########
start_bfs = time.time()
visit = list()
queue = list()

node_idx = 0

start_nodes = DCG[dcg_node_list[node_idx]]
for dev_idx, node_attr in enumerate(start_nodes):
    if node_attr.color != None:
        queue.append((node_idx, dev_idx))
        # print(DCG[dcg_node_list[node_idx]])

while queue:
    ## (1) Dequeue queue's head
    current_node_idx, current_dev_idx = queue.pop(0)
    current_node_attr = DCG[dcg_node_list[current_node_idx]][current_dev_idx]
    if current_node_attr.color == None:
        continue
    else:
        # print("\nPopped node: ", dcg_node_list[current_node_idx], "\t", current_dev_idx)
        if (current_node_idx, current_dev_idx) not in visit:
            if current_node_attr.color != None:
                visit.append((current_node_idx, current_dev_idx))

                ## (2) Enqueue Successor
                if current_node_idx < len(dcg_node_list) - 1:
                    next_node_idx = current_node_idx + 1
                    next_node_list = []
                    for next_dev_idx, next_node in enumerate(DCG[dcg_node_list[next_node_idx]]):
                        if next_node.color != None:
                            if next_node.color =='white':
                                next_node.color = 'black'
                                next_node_list.append((next_node_idx, next_dev_idx))
                    queue.extend(next_node_list)
                current_node_name = dcg_node_list[current_node_idx]             

                ## (3)
                backward_node_list = []
                if current_node_attr.src_ptr !=None:
                    backward_node_name, backward_dev_idx = DCG[dcg_node_list[current_node_idx]][current_dev_idx].src_ptr
                    backward_node_idx = dcg_node_list.index(backward_node_name)
                    backward_node_list.append((backward_node_idx, backward_dev_idx))                    
                
                if current_node_attr.src_next_ptr !=None:
                    backward_temp_node_attr = DCG[dcg_node_list[current_node_idx]][current_dev_idx].src_next_ptr
                    backward_node_name, backward_dev_idx = backward_temp_node_attr
                    backward_node_idx = dcg_node_list.index(backward_node_name)
                    backward_node_list.append((backward_node_idx, backward_dev_idx))                    
                # print(backward_node_list)

                ##Edge merging & Explored edge list update                 
                if backward_node_list !=[]:
                    set_explored_edge_list = []
                    set_explored_path = []
                    for i in range(len(backward_node_list)):                        
                        incoming_node_idx, incoming_dev_idx = backward_node_list[i] 
                        incoming_node_name = dcg_node_list[incoming_node_idx]
                        dma_size = DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].size
                        incoming_node_ua_edge = str(dma_size) + str(incoming_dev_idx) + str(current_dev_idx)
                        ua_edge_list_idx = ua_edge_list.index(incoming_node_ua_edge)

                        before = np.array(DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].explored_edge_list).astype(bool)
                        current = np.array(np.eye(explored_edge_list_max)[ua_edge_list_idx]).astype(bool)
                        result = np.bitwise_or(before, current)

                        # if len(backward_node_list) >= 2:
                        set_explored_edge_list.append(result)
                        set_explored_path.append(str(DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].explored_path))                        
                        # print("Incoming edge number:", len(backward_node_list))
                        # print(set_explored_edge_list, set_explored_path)
                    if len(set_explored_edge_list) == 1: 
                        merged_edge_list = set_explored_edge_list[0]
                        merged_path  = set_explored_path[0]
                        DCG[current_node_name][current_dev_idx].explored_edge_list = merged_edge_list
                        
                        distance_merge_to_current = current_node_idx - incoming_node_idx
                        append_list = []
                        for j in range(distance_merge_to_current-1):    
                            append_node_name = dcg_node_list[incoming_node_idx+j+1]
                            append_list.append(append_node_name)
                            append_list.append('->')
                        if distance_merge_to_current == 1:
                            DCG[current_node_name][current_dev_idx].explored_path = str(DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].explored_path)+'->'+str(current_node_name)+':'+str(current_dev_idx)
                        else:
                            DCG[current_node_name][current_dev_idx].explored_path = str(DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].explored_path)+'->' + str(append_list)+':0->'+str(current_node_name)+':'+str(current_dev_idx)

                    else:
                        # set_table = np.empty(shape=[len(ua_edge_list),len(set_explored_edge_list)])
                        
                        set_incoming = []
                        for i in range (0, len(set_explored_edge_list)):
                            set_index_vector = []
                            for j in range (0, len(ua_edge_list)):
                                if set_explored_edge_list[i][j] == True:
                                    set_index_vector.append(j)
                            set_incoming.append(set_index_vector)
                        
                        ## Only implement with 2 set
                        set0 = set_incoming[0]
                        set1 = set_incoming[1]
                        if all(i in set1 for i in set0):
                            DCG[current_node_name][current_dev_idx].explored_edge_list = set_explored_edge_list[1]
                            incoming_node_idx = backward_node_list[1][0]
                            distance_merge_to_current = current_node_idx - incoming_node_idx
                            append_list = []
                            for j in range(0, distance_merge_to_current-1):    
                                append_node_name = dcg_node_list[incoming_node_idx+j+1]
                                append_list.append(append_node_name)
                                append_list.append('->')
                            if append_list!=[]:
                                # by jwlee
                                tmp_append_node_name = append_list[0]
                                # DCG[current_node_name][current_dev_idx].explored_path = str(set_explored_path[1])+'->' + str(append_list)+':0->'+str(current_node_name)+':'+str(current_dev_idx)
                                DCG[current_node_name][current_dev_idx].explored_path = str(set_explored_path[1])+'->' + str(tmp_append_node_name)+':0->'+str(current_node_name)+':'+str(current_dev_idx)

                                # print(append_list)
                                # print("node: ", current_node_name, "\t", "dev_idx: ", current_dev_idx)
                            else:
                                DCG[current_node_name][current_dev_idx].explored_path = str(set_explored_path[1])+'->' + str(current_node_name)+':'+str(current_dev_idx)
                                # print("node: ", current_node_name, "\t", "dev_idx: ", current_dev_idx)
                        ##Case for not mergable set operation or Nth set... Need to modify
                        # else:                                

                else :
                    if current_node_idx == 0:
                        DCG[current_node_name][current_dev_idx].explored_path = str('START')+'->'+ str(current_node_name)+':'+str(current_dev_idx)
                        # print("node: ", current_node_name, "\t", "dev_idx: ", current_dev_idx)
                        DCG[current_node_name][current_dev_idx].explored_edge_list = np.zeros(explored_edge_list_max)
                    else :                        
                        DCG[current_node_name][current_dev_idx].explored_path = str(DCG[dcg_node_list[current_node_idx-1]][0].explored_path) +'->'+ str(current_node_name)+':'+str(current_dev_idx)
                        DCG[current_node_name][current_dev_idx].explored_edge_list = DCG[dcg_node_list[current_node_idx-1]][0].explored_edge_list
                        # print("node: ", current_node_name, "\t", "dev_idx: ", current_dev_idx)
                
                
                ##Early termination 
                end_point = np.array(DCG[current_node_name][current_dev_idx].explored_edge_list).astype(bool)
                all_pass = np.array(np.ones(len(ua_edge_list)).astype(bool))
                is_end = np.bitwise_and(end_point, all_pass)
                if  all(is_end):
                    # print('All explored edge list is True:', current_node_name)
                    # print(DCG[current_node_name][current_dev_idx].explored_edge_list)
                    # print(DCG[current_node_name][current_dev_idx].explored_path)
                    tmp_str = DCG[current_node_name][current_dev_idx].explored_path.split('->')
                    # print(tmp_str)
                    for tmp in tmp_str:
                        # print(tmp)
                        if tmp != 'START':
                            # print(tmp)
                            ttmp = tmp.split(':')
                            tmp_node_name = ttmp[0]
                            tmp_dev_idx = ttmp[1]
                            # print(tmp_node_name, "\t", tmp_dev_idx)
                            if tmp_dev_idx == '0':
                                dnn_partition['CPUExecutionProvider'].append(tmp_node_name)
                            else:
                                dnn_partition['PIMExecutionProvider'].append(tmp_node_name)
                            added_node_list.append(tmp_node_name)
                    break
        
        # print(DCG[current_node_name][current_dev_idx].explored_edge_list)
        # print(DCG[current_node_name][current_dev_idx].explored_path)

end_bfs = time.time()
# print("BFS time: ", end_bfs - start_bfs)

if set(pim_op_list) != set(added_node_list):
    not_included = set(pim_op_list) - set(added_node_list)
    for tmp_node in not_included:
        dnn_partition['CPUExecutionProvider'].append(tmp_node)
print(dnn_partition)
# CHECK DCG
# for key, value in DCG.items():
#     print("\n", key)
#     for val in value:
#         print(val)

            # if forward_node_ua_edge in ua_edge_list:
            #     ua_edge_list_idx = ua_edge_list.index(forward_node_ua_edge)
            #     DCG[forward_node_name][forward_dev_idx].explored_edge_list = np.eye(explored_edge_list_max)[ua_edge_list_idx]

            #     print("\nExplored edge list", ua_edge_list_idx)
            #     print("\nPopped node: ", dcg_node_list[node_idx], "\t", dev_idx)
            # print("\nForward node: ", dcg_node_list[forward_node_idx], "\t", forward_dev_idx)
            # print(DCG[forward_node_name][forward_dev_idx])



