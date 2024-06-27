# The RainNet2024 family of deep neural networks for precipitation nowcasting

This repository supports our paper submitted to [NHESS](https://www.natural-hazards-and-earth-system-sciences.net/):

> Ayzel, G., and Heistermann, M. "Brief Communication: Training of AI-based nowcasting models for rainfall early warning should take into account user requirements."

The RainNet2024 family's model configurations alongside the pre-trained weights are available at Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12547127.svg)](https://doi.org/10.5281/zenodo.12547127)

## TL;DR

We have developed the new set of deep learning models for precipitation nowcasting which continue our work in the field started from the development of the [RainNet](https://github.com/hydrogo/rainnet) (hereafter RainNet2020; [paper](https://gmd.copernicus.org/articles/13/2631/2020/)).

The RainNet2024 family consists of two types of models:
1. RainNet2024: a significant update of the RainNet2020 model, still providing predictions of rainfall intensities over the next 5 minutes (regression task).
2. RaiNet2024-S: set of models, each of those predicts the probability of threshold exceedance over the particular threshold of hourly rainfall accumulation (segmentation task). The thresholds are 5, 10, 15, 20, 25, 30, and 40 mm. 

<img src="misc/the-rainnet2024-family.png" alt="RainNet2024 family models" width="100%"/>

The source of model configurations -- the [segmentation-models](https://github.com/qubvel/segmentation_models) library developed by [Pavel Iakubovskii](https://github.com/qubvel).

## RainNet2024


```python
import segmentation_models as sm

rainnet2024 = sm.Unet(backbone_name="efficientnetb4",
                      encoder_weights=None,
                      classes=1,
                      activation="linear",
                      input_shape=(256, 256, 4))
```


## RainNet2024-S

```python
import segmentation_models as sm

rainnet2024 = sm.Unet(backbone_name="efficientnetb4",
                      encoder_weights=None,
                      classes=1,
                      activation="sigmoid",
                      input_shape=(256, 256, 4))
```


## Computation environment

`rainnet2024_environment.yml` file provides all the necessary dependencies for working with the RainNet2024 family of models, as well as standard models from the [PySteps](https://github.com/pySTEPS/pysteps) library and radar data processing procedures.

## Data

+ YW
+ CatRaRE

## Training

Table with data split: number of events/instances in each fold

preprocessing, loss, optimizer, epochs (20). LR reduction.

Model weights are available on zenodo: "fill in".

## Evaluation

+ CSI
+ FSS

## Sample event

+ YW data for a single event

<img src="misc/20815_24.png" alt="RainNet2024 family models" width="100%"/>

## Operational setting

+ copy from KISTERS's script

<!-- Note on overconfidence with jaccard loss  vs. bce -->

