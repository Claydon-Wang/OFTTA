## Optimization-Free Test-Time Adaptation for Cross-Person Activity Recognition

#### Installation:
Please create and activate the following conda envrionment.
```bash
# It may take several minutes for conda to solve the environment
conda create -y -n tta python=3.8
conda activate tta
pip install -r requirements.txt 
```

#### HAR Dataset
Three datasets (UCIHAR, Opportunity, and UniMiB SHAR) are utilized in the experiments. The pre-processed outcome can be downloaded from an anonymous [repository](https://drive.google.com/drive/folders/1Y8jLalh2IFCf0lcG8bivTpTylbkdeoNr?usp=sharing). The dataset is adopt from another repository. Please save datasets under folder `./data`. 

#### Pre-trained Model
Since the model need pre-trained in source domain. We provide our used model in Table 4. You can download the model from an anonymous [repository](https://drive.google.com/drive/folders/1_GR6W0va5kd25aU2n21myKfhTY17V3F3?usp=share_link). Please save datasets under folder `./ckpt`. Code for training will be released later. 

#### Reproduce our results
To reproduce the leave-one-out adaptation results in Table 4, you just need:
```bash
bash adapt.sh
```
You can get the all results in Table 4.