# DeepMe: AI-driven Deep Microbial–enzyme Modeling to Forecast Soil CO2 Flux under Warming at Harvard Forest
This project consists of the official implementation of DeepMe, which is transformer based neural network for CO2 efflux forecasting.

1. [Abstract](#Abstract)
2. [Implementation](#Implementation)
3. [Result](#Result)

## Abstract
Our DeepMe framework is a deep-learning microbial–enzyme modeling framework that predicts hourly soil CO2 efflux. Our contribution can be summarized as:
- We intergrate environmental drivers with biologically informed constraints through a data assimilation scheme.
- We thoroughly evaluate the transfer performance of both our method and existing approaches on the extended dataset.

<img width="1616" height="814" alt="image" src="https://github.com/user-attachments/assets/7be9beb3-5657-47a2-8a36-5290757e552c" />

## Implementation

### 1. Data Preparation

### 2. MEND Implementation

The MEND matlab codes are under the `MEND` folder.

To train MEND model under control condition, run `MEND_Sim_Con_12p.m`

To train under heated condition, run `MEND_Sim_Heat_12p.m`

### 3. DeepMe Implementation

All DeepMe codes are under the `DeepMe` folder.

To train the model, use

`python train.py`

To evaluate the model, use

`python eval.py`


## Result

DeepMe model outperforms MEND in both control and heated group, raising the R-Sqaure to around 0.5.

<img width="894" height="352" alt="image" src="https://github.com/user-attachments/assets/ac2302e5-823d-40cf-ac12-1496dc749ebf" />

## Citation




