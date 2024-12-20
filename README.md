# detection_logits

This is the official repository for the paper [Toxicity Detection for Free](https://arxiv.org/abs/2405.18822) accepted as a spotlight in Neurips 2024.

<p align="center">
  <img src="readme_figs/fig1.jpg" alt="drawing" width="80%"/>
</p>

#### 1. Installation
All the codes are tested in the following environment:
* python 3.11.0
* pytorch 2.0.1
* numpy 1.26.0
* scikit-learn 1.3.1
* transformers 4.34.0
* fschat 0.2.30
* easydict 1.10
* tqdm 4.66.1

#### 2. Run

Configurations are in [configs.py](https://github.com/WhoTHU/detection_logits/blob/36bb1dc74ef91a714f4a9057a69b9387c1697e78/configs.py)

Run the code by
```
python main.py
```
Uncomment the code for plotting figures if you require visual feedback. It will show plots like (an example for ToxicChat):

<p align="center">
  <img src="readme_figs/test_plot.jpg" alt="drawing" width="100%"/>
</p>