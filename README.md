# Code Repository for Neural Network-Based Robotic Optical System for Real-Time Control of Mineral Fertilizer Granulation Process

This repository contains the code implementation for the research presented in the paper (EDM-IEEE 2025):

> Dmitrii Iunovidov, Vyacheslav Shevchenko, Elizaveta Iunovidova, Ikechi Ndukwe. 
> "Neural Network-Based Robotic Optical System for Real-Time Control of Mineral Fertilizer Granulation Process."

## Description

This project focuses on developing a robotic optical system for controlling the granulation process of mineral 
fertilizers in industrial production. The key component is a custom neural network for instance segmentation, 
built upon UNet and MobileNet v3 architectures, designed for real-time image processing. 


The repository includes:

* [Link](https://cloud.mail.ru/public/BqmQ/tLCPw9zUY) to weights for all neural networks in `ONNX` and `pht` format. Load files into `models` folder.
* Examples of model inferences in [examples](examples), including [segmentation post-processing procedures](examples/EDM_semantic-segm_inference.ipynb). 
* Test dataset in COCO format (for IoU calculation) in [IoU_test](IoU_test).
* Configuration files for model implementation in [src](src).
* Test images, which models didn't see early (data-shifted images) in [test_images](test_images).
* [Dataset implementation for semantic segmentation from creation (from COCO instance segmentation)](examples/EDM_semantic-from-COCO.ipynb).


## [Dependencies](requirements.txt)

* Python 3.9
* PyTorch 2.1.1
* OpenCV 4.10.0.84
* Other libraries in [requirements.txt](requirements.txt)


## Installation

  Set the `VENV` variable according to your python virtual environment path and install packages with:

    ```bash
    make install
    ```

##   Usage

1.  **Data Preparation:**

    * Prepare your dataset of granule images or use [IoU_test](IoU_test) sample.
    * Annotate the images for instance segmentation (e.g., in COCO format) is needed.

2.  **Training:**

    * Train models as described in the related article or use already trained weights in [models](models).
      * For segmentation model please use [pytorch-lightning](https://lightning.ai/) with [smp](https://github.com/qubvel-org/segmentation_models.pytorch).
      * For YOLO please use [Ultralytics](https://docs.ultralytics.com).
      * For other instance-segmentation models please use [mmdet](https://github.com/open-mmlab/mmdetection).

3.  **Evaluation:**

    * Use the evaluation scripts in [examples](examples) to assess the model's performance: IoU and number of granules evaluation (already includes in each inference file); speed of inference evaluates in [EDM_model-evaluations.ipynb](examples/EDM_model-evaluations.ipynb).

4.  **Inference:**

    * Use the inference scripts to run the model on new images according to [examples](examples).


## Makefile

File with common commands:

* `make install` - install libraries in python virtual environment
* `make lint` - nbstripout linter for notebooks
