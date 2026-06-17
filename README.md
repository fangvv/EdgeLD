# EdgeLD

This is the source code for our paper: **EdgeLD: Locally Distributed Deep Learning Inference on Edge Device Clusters**. A brief introduction of this work is as follows:

> Deep Neural Networks (DNN) have been widely used in a large number of application scenarios. However, DNN models are generally both computation-intensive and memory-intensive, thus difficult to be deployed on resource-constrained edge devices. Most previous studies focus on local model compression or remote cloud offloading, but overlook the potential benefits brought by distributed DNN execution on multiple edge devices. In this paper, we propose EdgeLD, a new framework for locally distributed execution of DNN-based inference tasks on a cluster of edge devices. In EdgeLD, DNN models' time cost will be firstly profiled in terms of computing capability and network bandwidth. Guided by profiling, an efficient model partition scheme is designed in EdgeLD to balance the assigned workload and the inference runtime among different edge devices. We also propose to employ layer fusion to reduce communication overheads on exchanging intermediate data among devices. Experiment results show that our proposed partition scheme saves up to 15.82% of inference time with regard to the conventional solution. Besides, applying layer fusion can speedup the DNN inference by 1.11-1.13X. When combined, EdgeLD can accelerate the original inference time by 1.77-3.57X on a cluster of 2-4 edge devices.
> 深度神经网络（DNN）已被广泛应用于众多场景。然而，DNN模型通常具有计算密集和内存密集的双重特性，导致其难以部署在资源受限的边缘设备上。现有研究多集中于本地模型压缩或远程云端卸载方案，却忽视了在多台边缘设备上分布式执行DNN带来的潜在优势。本文提出EdgeLD框架，实现在边缘设备集群上本地分布式执行基于DNN的推理任务。该框架首先从计算能力和网络带宽两个维度对DNN模型进行时间成本分析，并以此为指导设计高效的模型划分方案，以平衡不同边缘设备间的工作负载分配与推理运行时长。我们还采用层融合技术来降低设备间交换中间数据产生的通信开销。实验结果表明：与传统方案相比，我们提出的划分方案最高可节省15.82%的推理时间；应用层融合技术可使DNN推理速度提升1.11至1.13倍；当两者结合时，EdgeLD在2-4台边缘设备组成的集群上可实现1.77至3.57倍的原始推理加速。

This work has been published by IEEE HPCC 2020 [link](https://ieeexplore.ieee.org/document/9408006). The technique report can be downloaded from [here](https://github.com/fangvv/EdgeLD/raw/master/TR-EdgeLD.pdf).

## Required software

- PyTorch
- NumPy

## Project Structure
```
EdgeLD/
├── 项目代码/                                       # Project code
│   └── EdgeMI/                                    # Edge Model Inference main module
│       ├── VGG/                                   # VGG model definitions and tensor operations
│       │   ├── mydefine_VGG13.py                  # VGG13 with layer-wise partition support (start/end)
│       │   ├── mydefine_VGG16.py                  # VGG16 model
│       │   ├── mydefine_VGG19.py                  # VGG19 model
│       │   ├── tensor_op.py                       # Tensor partition/merge: by computing power / by computing+network
│       │   └── vgg.py                             # Original VGG reference
│       ├── image/                                 # Test images and data loading
│       │   ├── cat.jpg
│       │   └── getImageData.py
│       ├── inference_stage/                       # Inference pipeline
│       │   └── muilt_inference.py                 # Multi-device inference orchestration
│       ├── network_and_computing/                 # Network & computing profiling
│       │   ├── measure_computing.py               # Measure device computing capability
│       │   ├── network_and_computing_record.py    # Computing power (a,b) and bandwidth records
│       │   └── divid_test.py
│       ├── node_test/                             # Distributed node deployment
│       │   ├── namenode_0.py                      # NameNode (master): partition & dispatch
│       │   ├── datanode_0.py                      # DataNode (worker): receive & compute
│       │   ├── network_op.py                      # TCP socket communication
│       │   └── num_set_up.py                      # Node number configuration
│       └── test.py
├── 实验介绍/                                       # Experiment introduction
│   ├── 实验参数设置.docx                           # Experiment parameter settings
│   └── 软件、硬件介绍.docx                         # Software/hardware introduction
├── TR-EdgeLD.pdf                                  # Technique report
└── README.md
```
## Core Modules
### VGG_model (`VGG/mydefine_VGG13.py` etc.)
The VGG model split into layer-wise modules. Each `module_list[i]` corresponds to one network layer (conv/maxpool/classifier), enabling partial inference via `forward(x, start, end)`.
| Method | Description |
|--------|-------------|
| `forward(x, start, end)` | Run layers from `start` to `end` (inclusive); supports conv-only, fc-only, or mixed ranges |
| `get_conv_length()` | Return the number of conv/maxpool layers (15 for VGG13) |
| `get_total_length()` | Return total layer count (including classifier) |
| `get_maxpool_layer()` | Return maxpool layer indices `[3, 6, 9, 12, 15]` used as partition boundaries |
| `get_c_out()` | Return output channel list per layer |
### tensor_op (`VGG/tensor_op.py`)
Tensor partition and merge operations, the core of the model partition scheme.
| Function | Description |
|----------|-------------|
| `tensor_divide_by_computing_and_fill` | Partition by computing power, with `cross_layer` overlap for conv correctness |
| `tensor_divide_by_computing_and_network` | Partition jointly by computing power **and** network bandwidth; iteratively balances predicted time |
| `get_prediction_time` | Predict per-device time = `a * FLOPs + b + comm_time` |
| `merge_total_tensor` | Merge partitioned outputs back, stripping the overlap regions |
| `divied_middle_output` | For differential exchange: split output into `saved_tensor` (kept locally) and `divied_tensor` (sent to namenode) |
### Network_And_Computing (`network_and_computing/network_and_computing_record.py`)
Records the computing and network parameters of each device.
| Attribute | Description |
|-----------|-------------|
| `computing_a` / `computing_b` | Linear regression coefficients for compute time: `t = a * FLOPs + b` |
| `computing_power` | `1/a`, relative computing capability per device |
| `network_state` | Network bandwidth (bps) per device |
### NameNode (`node_test/namenode_0.py`)
The master node that drives distributed inference. For each partition stage (bounded by maxpool layers), it partitions the feature map via `tensor_divide_by_computing_and_network`, sends sub-tensors to each DataNode in parallel threads, collects results, and merges them. Four execution scenarios are included:
1. Equal computing power, full data exchange
2. Different computing power, single-layer differential exchange
3. Different computing & network, single-layer differential exchange
4. Different computing & network, multi-layer differential exchange with layer fusion
### DataNode (`node_test/datanode_0.py`)
The worker node that receives a sub-tensor and layer range `(start, end)` from NameNode, runs `inference_model(recv_tensor, start, end)`, and returns the result. In differential-exchange scenarios, it keeps `saved_tensor` locally and only sends back the boundary slices.
### network_op (`node_test/network_op.py`)
TCP socket-based communication. `Network_init_namenode` connects to all DataNodes; `Network_init_datanode` accepts the NameNode connection. Tensors are serialized via NumPy bytes with a `start@end@size@bytes` protocol.
## Usage
Deploy on multiple edge devices in the same LAN. Configure IP addresses and ports in `network_op.py`, set the node count in `num_set_up.py`, then launch the NameNode and each DataNode.
```bash
# Install dependencies
pip install torch numpy

# On each DataNode device (set datanode_name in each file accordingly)
cd 项目代码/EdgeMI
python node_test/datanode_0.py    # datanode_1.py, datanode_2.py, ...

# On the NameNode device (after all DataNodes are ready)
cd 项目代码/EdgeMI
python node_test/namenode_0.py
```

## Citation
If you find EdgeLD useful or relevant to your project and research, please kindly cite our paper:
	@inproceedings{xue2020edgeld,
  		title={Edgeld: Locally distributed deep learning inference on edge device clusters},
  		author={Xue, Feng and Fang, Weiwei and Xu, Wenyuan and Wang, Qi and Ma, Xiaodong and Ding, Yi},
  		booktitle={2020 IEEE 22nd International Conference on High Performance Computing and Communications; IEEE 18th International Conference on Smart City; IEEE 6th International Conference on Data Science and Systems (HPCC/SmartCity/DSS)},
  		pages={613--619},
  		year={2020},
  		organization={IEEE}
	}

## Stargazers over time

[![Stargazers over time](https://starchart.cc/fangvv/EdgeLD.svg)](https://starchart.cc/fangvv/EdgeLD)

## For more

We have another work on [UAV-DDPG](https://github.com/fangvv/UAV-DDPG) for UAV-assisted mobile edge computing, and you can simply use [Ray](https://github.com/ray-project/ray) for implementing distributed computing now.

## Contact

Feng Xue (17120431@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
