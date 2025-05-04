# 以太坊欺诈节点 数据集构建

## 虚拟环境
```bash
conda create -n eth_dataset python=3.12
conda activate eth_dataset
```

## 文档目录
### api
api 目录下存放的是调用 api 的脚本，目前有两个脚本：

### get_eth_info
对api的方法进行封装

### api_key
dune: 78XIGQD58zFINI0GojG5mxRvdrhP0q6D

## 训练环境配置
```bash
conda create -n eth_dataset python=3.9
conda activate eth_dataset
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam/label/cu118 dgl
conda install packaging
conda update setuptools
```

