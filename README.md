# TW-GAN: Topology and Width Aware GAN for Retinal Artery/Vein Classification
This repository is an official PyTorch implementation of the paper **"TW-GAN: Topology and width aware GAN for retinal artery/vein classification"** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841521003856)] from Medical Image Analysis 2022.


## TW-GAN

<div align=center><img width="600" src=./figs/architecture.png></div>

* In this paper, we propose a novel Topology and Width Aware Generative Adversarial Network (named as TW-GAN), which, for the first time, integrates the topology connectivity and vessel width information into the deep learning framework for A/V classification.
* To improve the topology connectivity, a topology-aware module is proposed, which contains a topology ranking discriminator based on ordinal classification to rank the topological connectivity level of the ground-truth mask, the generated A/V mask and the intentionally shuffled mask. 
* In addition, a topology preserving triplet loss is also proposed to extract the high-level topological features and further to narrow the feature distance between the predicted A/V mask and the ground-truth mask. 
* Moreover, to enhance the model’s perception of vessel width, a width-aware module is proposed to predict the width maps for the dilated/non-dilated ground-truth masks.

## Prequisites
You can "pip install" the packages in "./requirement.txt"

## Dataset
* To prepare the dataset, you can download AV-DRIVE and HRF datasets from [google drive](https://drive.google.com/drive/folders/1mMkKJ3fpwamf1TVwym9IsZm9f8c6uHnf?usp=sharing). 
* Please place dataset in ./data directory.
* ./data folder includes the datasets for AV-DRIVE and HRF, their corresponding centerline distance maps and shuffled masks.

** The A/V label for [HRF dataset](https://drive.google.com/drive/folders/1Uluvc8Cib-acddIkj4Mk5o49U9t7ps60?usp=sharing) is mannually labeled by us.<br/>

## Data preprocessing
To prepare the centerline distance map and shuffled A/V label for dataset, please run:
```bash
    sh ./launch/preprocess_data.sh
```
(The downloaded "./data" folder includes the processed centerline distance map and shuffled A/V label. So you don't need to run it if you download it.)

## Usage
Please make a new "log" folder first:
```bash
    mkdir log
```

For AV-DRIVE dataset
* Train:
```bash
    sh ./launch/train_AV_DRIVE.sh
```
* Test:
```bash
    sh ./launch/test_AV_DRIVE.sh
```
For HRF dataset
* Train:
```bash
    sh ./launch/train_HRF.sh
```
* Test:
```bash
    sh ./launch/test_HRF.sh
```

## Pretrained models
Please download the pretrained models from google drive(https://drive.google.com/drive/folders/1idqTGV22qVsKDAO0tgOkMtuPbdZCNMys?usp=sharing)
To test the pretrained model, you can change the ./config/config_test_HRF.py or ./config/config_test_AV_DRIVE.py : 
```
model_path_pretrained_G = './pretrained_model_path'
```

## Cite
If you find our work useful in your research or publication, please cite our work:
```
@article{CHEN2022102340,
title = {TW-GAN: Topology and width aware GAN for retinal artery/vein classification},
journal = {Medical Image Analysis},
volume = {77},
pages = {102340},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102340},
author = {Wenting Chen and Shuang Yu and Kai Ma and Wei Ji and Cheng Bian and Chunyan Chu and Linlin Shen and Yefeng Zheng}
}
```

## Contact

If you have any question, please feel free to contact me. ^_^ wentichen7-c[at]my.cityu.edu.hk




