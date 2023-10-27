## Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition [IMWUT/UbiComp 2024]

This is the official implementation for "Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition".
#### Installation:
Please create and activate the following conda envrionment.
```bash
# It may take several minutes for conda to solve the environment
conda create -y -n oftta python=3.9
conda activate oftta
pip install -r requirements.txt 
```

#### HAR Dataset
Three datasets (UCI-HAR, Opportunity, and UniMiB-SHAR) are utilized in the experiments. The pre-processed outcome can be downloaded from [here](https://drive.google.com/drive/folders/1Y8jLalh2IFCf0lcG8bivTpTylbkdeoNr?usp=sharing). The datasets is adopt from [GILE](https://drive.google.com/drive/folders/1Y8jLalh2IFCf0lcG8bivTpTylbkdeoNr?usp=sharing). Please save datasets under folder `./data`. 

#### Pre-trained Model
Since test-time adaptation needs pre-trained on source domains. We provide our used model in Table 4. You can download the model from [here](https://drive.google.com/drive/folders/1_GR6W0va5kd25aU2n21myKfhTY17V3F3?usp=share_link). Please save datasets under folder `./ckpt`. If you want to train your model from scratch, you can refer to code for [generalizable HAR](https://github.com/Claydon-Wang/DG_HAR). 

#### Reproduce our results
To reproduce the leave-one-out adaptation results in Table 4, you just need:
```bash
bash adapt.sh
```
You can get the all results in Table 4. 

#### Supported algorithms
We support all the TTA algorithms used in the paper. Feel free to adopt them on other types of dataset.

| Title                                                                                                                            | Venue | 
|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|
| PL: [Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks](https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks)                          | ICML Workshop 2013   | 
| SHOT: [Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](https://arxiv.org/abs/2002.08546)   | ICML 2020    | 
|BN: [Improving robustness against common corruptions by covariate shift adaptation](https://arxiv.org/abs/2006.16971) | NeurIPS 2020  |                                                                                           |
| TENT: [Tent: Fully test-time adaptation by entropy minimization](https://openreview.net/forum?id=uXl3bZLkr3c)   | ICLR 2021  |
|T3A: [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://openreview.net/forum?id=e_yvNqkJKAW)   | NeurIPS 2021                | 
|TAST: [ Test-time Adaptation via Self-training with Nearest Neighbor information](https://arxiv.org/abs/2207.10792)   | ICLR 2023     |
|SAR: [ Towards Stable Test-time Adaptation in Dynamic Wild World ](https://openreview.net/forum?id=g2YraF75Tj)   | ICLR 2023    |
