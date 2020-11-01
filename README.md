## EdgeLD

This is the source code for our paper: **Locally Distributed Deep Learning Inference on Edge Device Clusters**. A brief introduction of this work is as follows:

> Deep Neural Networks (DNN) have been widely used in a large number of application scenarios. However, DNN models are generally both computation-intensive and memory-intensive, thus difficult to be deployed on resource-constrained edge devices. Most previous studies focus on local model compression or remote cloud offloading, but overlook the potential benefits brought by distributed DNN execution on multiple edge devices. In this paper, we propose EdgeLD, a new framework for locally distributed execution of DNN-based inference tasks on a cluster of edge devices. In EdgeLD, DNN models' time cost will be firstly profiled in terms of computing capability and network bandwidth. Guided by profiling, an efficient model partition scheme is designed in EdgeLD to balance the assigned workload and the inference runtime among different edge devices. We also propose to employ layer fusion to reduce communication overheads on exchanging intermediate data among devices. Experiment results show that our proposed partition scheme saves up to 15.82% of inference time with regard to the conventional solution. Besides, applying layer fusion can speedup the DNN inference by 1.11-1.13X. When combined, EdgeLD can accelerate the original inference time by 1.77-3.57X on a cluster of 2-4 edge devices.

This work has been accepted by IEEE HPCC 2020. The technique report can be downloaded from [here](TR-EdgeLD.pdf)

## Required software

PyTorch

## Contact

Feng Xue (17120431@bjtu.edu.cn)

Weiwei Fang (fangvv@qq.com)

