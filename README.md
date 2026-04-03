# 🌊 OceanTube-ML: Probabilistic Forecasting & Bias Correction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Active_Research-success)

This repository contains the Machine Learning codebase for **OceanTube**, an advanced deep learning framework designed to correct error accumulation (drift) in global climate models and generate **probabilistic forecasts** for oceanographic variables.

Unlike standard deterministic models that output a single, fragile point prediction, this project utilizes a novel custom loss function to generate **Prediction Intervals (Tubes)**. This allows us to mathematically capture the inherent chaos and uncertainty of ocean dynamics across the Indian Ocean and global datasets.

---

## 🎯 Project Objectives

1. **CMIP6 Bias Correction (MOS):** Acting as a Model Output Statistics (MOS) filter, our LSTM networks learn the long-term drift and biases of flawed global climate models (like `MPI-ESM`) and correct them against highly accurate satellite ground-truths.
2. **Probabilistic Forecasting:** Predicting future states for **Sea Surface Height (SSH/ZOS)** and **Significant Wave Height (SWH)** with mathematically rigorous upper and lower confidence bounds.
3. **Multi-Region Analysis:** Forecasting across highly volatile sub-regions of the Indian Ocean, including:
   * The Arabian Sea
   * The Bay of Bengal
   * The Andaman Sea
   * The Equatorial Indian Ocean

---

## 🧮 Mathematical Architecture: The Tube Loss

Standard Mean Squared Error (MSE) is insufficient for capturing ocean volatility. Instead of predicting a single line, this codebase implements custom gradients in TensorFlow/Keras to optimize for prediction intervals.

* **Custom Tube Loss:** A dynamic loss function that predicts two bounds ($\mu_1, \mu_2$) simultaneously. It applies a width penalty ($\delta$) to aggressively shrink the forecasting envelope while heavily penalizing the model if the true satellite observation falls outside the tube. 
* **Quantile (Pinball) Loss:** Asymmetric penalty functions utilized to mathematically bind predictions to specific percentiles (e.g., 5th and 95th).

> **The Result:** Instead of an exact prediction that is guaranteed to be slightly wrong, the model provides a guaranteed "Safe Range" (e.g., *95% confident the wave height will fall exactly between 2.05m and 2.14m*).

---

## 📊 Evaluation Metrics

To measure the success of our probabilistic bias correction against raw CMIP6 projections and benchmark ML models, we evaluate using:

* **PICP (Prediction Interval Coverage Probability):** The percentage of actual satellite observations successfully captured inside our forecasted green tube. *(Target Goal: ~95%)*
* **MPIW (Mean Prediction Interval Width):** The average thickness of the predicted tube. Optimized to be as tight as possible without sacrificing PICP coverage.

---

## 📂 Data Sources

This framework bridges the gap between long-term flawed projections and short-term accurate observations:
1. **Model Inputs (Features):** Multi-variable climate projection data from **CMIP6** (e.g., `MPI-ESM1-2-LR`) and **COWCLIP** global wave models.
2. **Ground Truth (Targets):** Daily historical satellite radar altimetry data provided by the **Copernicus Marine Environment Monitoring Service (CMEMS)**.

---

## 🏗️ Repository Structure

```text
ocean-probabilistic-forecasting-ml/
│
├── data/                  # Instructions for accessing CMIP6 and CMEMS .nc files
├── models/                # Saved pre-trained Keras models (.keras / .h5)
├── notebooks/             # Interactive Jupyter/Colab notebooks for extraction and training
│   ├── 1_CMIP6_SSH_Correction.ipynb
│   └── 2_Wave_Height_Altimeter.ipynb
│
├── src/                   # Core Python modules
│   ├── data_pipeline.py   # NetCDF alignment and spatial extraction
│   └── custom_losses.py   # TensorFlow implementations of Tube Loss
│
└── README.md
