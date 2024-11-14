# Leafy Spurge Demography: Remote Sensing Analysis

## Introduction

Despite a growing understanding of the mechanisms and consequences of biological invasions, forecasting the spread of introduced populations remains challenging. This repository focuses on leveraging remote sensing techniques for cost-effective strategies to locate and predict the spread of invasive species.

## Remote Sensing and Species Distribution Models (SDMs)

This repository explores the benefits of remote sensing, particularly satellite imagery, as a powerful tool for collecting information on the spatial distribution and abundance of invasive species. The focus is on developing convolutional neural networks that integrate time-series remote sensing data to enhance the accuracy of predictions.

## Study Focus: Leafy Spurge

We concentrate on leafy spurge (Euphorbia virgata; Euphorbiaceae) in Minnesota, USA. Leafy spurge is the most economically damaging invasive plant in the US, with total costs exceeding $1 billion. The study uses Landsat scenes from 2000 to 2020 to build deep learning convolutional neural networks, incorporating remote sensing data for predicting the probability of occurrence and inferring population growth/decline.

## Repository Structure

- Python files (.py) include code for preparing Landsat data, for creating training datasets, and for training and evaluating a temporal convolutional neural network.
- CSV (.csv) files included an example training dataset for the temporal convolutional neural network.
- `temporalCNN/`: Code for the temporal Convolutional Neural Network (forked from https://github.com/charlotte-pel/temporalCNN)
- `LICENSE`: Repository license information.

---

**Citation:**
Include the relevant citations for the studies and methodologies mentioned in the introduction and throughout the README.
