# 🚀Revisiting Applicable and Comprehensive Knowledge Tracing in Large-Scale Data (ARR2025 submitted)

PyTorch implementation of [DKT2](https://openreview.net/pdf?id=3mXwcMmYIg).

## 🌟Data and Data Preprocessing

Place the [Assist17](https://sites.google.com/view/assistmentsdatamining/dataset?authuser=0), [EdNet](https://github.com/riiid/ednet), and [Comp](https://github.com/wahr0411/PTADisc) source files in the dataset directory, and process the data using the following commands respectively:

```python
python preprocess_data.py --data_name assistments17
python preprocess_data.py --data_name ednet
python preprocess_data.py --data_name comp
```

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

Our model experiments are conducted on two NVIDIA RTX 3090 24GB GPUs. You can execute it directly using the following commands:

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
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --trans True --len 5 --mask_future
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --trans True --len 5 --mask_response
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --trans True --len 5 --pred_last
```

- Multi-concept Prediction

```python
CUDA_VISIBLE_DEVICES=0 python main.py --model_name dkt2 --data_name assistments17 --joint True
CUDA_VISIBLE_DEVICES=0 python main.py --model_name akt --data_name assistments17 --trans True --joint True
```
