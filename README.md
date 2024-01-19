# Leafy Spurge Demography: Remote Sensing Analysis

## Introduction

Despite a growing understanding of the mechanisms and consequences of biological invasions, forecasting the spread of introduced populations remains challenging. This repository focuses on leveraging remote sensing techniques for cost-effective strategies to locate and predict the spread of invasive species.

## Remote Sensing and Species Distribution Models (SDMs)

This repository explores the benefits of remote sensing, particularly satellite imagery, as a powerful tool for collecting information on the spatial distribution and abundance of invasive species. The focus is on developing convolutional neural networks that integrate time-series remote sensing data to enhance the accuracy of predictions.

## Study Focus: Leafy Spurge

We concentrate on leafy spurge (Euphorbia virgata; Euphorbiaceae) in Minnesota, USA. Leafy spurge is the most economically damaging invasive plant in the US, with total costs exceeding $1 billion. The study uses Landsat scenes from 2000 to 2020 to build deep learning convolutional neural networks, incorporating remote sensing data for predicting the probability of occurrence and inferring population growth/decline.

## Repository Structure

- `analysesNotebooks/`: Contains Jupyter notebooks for data analysis.
- `pythonFiles/`: Includes Python scripts used in the remote sensing analysis.
- `temporalCNN/`: Focuses specifically on the temporal Convolutional Neural Network.
- `slurmScripts/`: Contains SLURM scripts for high-performance computing.
- `README.md`: You are currently reading this file.
- `LICENSE`: Repository license information.

![alt text](https://github.com/lake-thomas/leafy-spurge-demographyblob/main/Predicted_LeafySpurge_2019.JPG?raw=true)

---

**Citation:**
Include the relevant citations for the studies and methodologies mentioned in the introduction and throughout the README.
