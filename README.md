# 🚀Revisiting Applicable and Comprehensive Knowledge Tracing in Large-Scale Data (ECML-PKDD 2025)

PyTorch implementation of [DKT2](https://arxiv.org/abs/2501.14256).

<h5 align=center>
      
[![arXiv](https://img.shields.io/badge/Arxiv-2501.14256-red?logo=arxiv&label=Arxiv&color=red)](https://arxiv.org/abs/2501.14256)
[![License](https://img.shields.io/badge/Code%20License-MIT%20License-yellow)](https://github.com/zyy-2001/DKT2/blob/master/LICENSE)
![GitHub Repo stars](https://img.shields.io/github/stars/zyy-2001/DKT2/)

</h5>


## 🌟Data and Data Preprocessing

Place the [Assist17](https://sites.google.com/view/assistmentsdatamining/dataset?authuser=0), [EdNet](https://github.com/riiid/ednet), and [Comp](https://github.com/wahr0411/PTADisc) source files in the dataset directory, and process the data using the following commands respectively:

```python
python preprocess_data.py --data_name assistments17
python preprocess_data.py --data_name ednet
python preprocess_data.py --data_name comp
```

You can also download the dataset from the [link](https://drive.google.com/file/d/1PMikGhRwSVAFc0319vxGkoZslM_jvYI_/view?usp=sharing) and unzip it in the current directory.

The statistics of the three datasets after processing are as follows:

| Datasets | #students | #questions | #concepts | #interactions |
| :------: | :-------: | :--------: | :-------: | :-----------: |
| Assist17 |   1,708   |   3,162   |    411    |    934,638    |
|  EdNet  |  20,000  |   12,215   |   1,781   |   2,709,132   |
|   Comp   |  45,180  |   8,392   |    472    |   6,072,632   |

## ➡️Quick Start

### Installation

Git clone this repository and create conda environment:

```python
conda create -n dkt2 python=3.11
conda activate dkt2
pip install -r requirements.txt 
conda create -n mamba4kt python=3.11
conda activate mamba4kt
pip install -r requirements.txt 
```

It's important to note that xLSTM and Mamba require different CUDA versions, so it's necessary to install two separate Conda virtual environments. At the same time, please strictly follow the installation instructions for [xLSTM](https://github.com/NX-AI/xlstm) and [Mamba](https://github.com/state-spaces/mamba) as provided in their respective GitHub repositories. Downloading the correct CUDA packages is crucial.



### Training & Testing

You can execute it directly using the following commands:

- One-step Prediction

```python
CUDA_VISIBLE_DEVICES=0 python main.py --model_name dkt2 --data_name assistments17
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --trans True
```

- Multi-step Prediction

```python
CUDA_VISIBLE_DEVICES=0 python main.py --model_name dkt2 --data_name assistments17 --len 5
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --trans True --len 5
```

- Varying-history-length Prediction

```python
CUDA_VISIBLE_DEVICES=0 python main.py --model_name dkt2 --data_name assistments17 --seq_len 500
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --trans True --seq_len 500
```

- Different Input Settings

```python
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --len 5 --mask_future (△ setting)
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --len 5 --mask_response (◦ setting)
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --len 5 --pred_last (• setting)
```

- Multi-concept Prediction

```python
CUDA_VISIBLE_DEVICES=0 python main.py --model_name dkt2 --data_name assistments17 --joint True
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --trans True --joint True
```

## ⚠️Citation
If you find our work valuable, we would appreciate your citation: 
```text
@article{zhou2025revisiting,
  title={Revisiting Applicable and Comprehensive Knowledge Tracing in Large-Scale Data},
  author={Zhou, Yiyun and Han, Wenkang and Chen, Jingyuan},
  journal={arXiv preprint arXiv:2501.14256},
  year={2025}
}
```
