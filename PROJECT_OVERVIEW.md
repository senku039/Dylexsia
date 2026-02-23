# Project Overview

This repository explores dyslexia detection from eye-tracking signals using mostly unsupervised ML workflows.

## Pipeline structure

1. **Early work** (`1_Early work/`)
   - Data gathering and multiple sequence-length equalization strategies (padding, interpolation, exterpolation).
2. **Binning** (`2_Binning/`)
   - FFT-based vector binning to map variable-length trajectories into fixed-length vectors, then KMeans clustering.
3. **Analyzing binned data** (`3_Analysing Binned Data/`)
   - Section-wise clustering and PCA-based weighting / distance analysis utilities.
4. **STFT and Perceptron** (`4_STFT and Perceptron/`)
   - STFT feature extraction variants and perceptron-oriented experimentation notebooks.

## Core modeling idea

Eye-tracking traces from left/right eyes are reduced to combined X/Y or complex-valued trajectories, transformed into spectral representations (FFT/STFT), then converted into equal-length vectors so clustering/classification can be run consistently across candidates with different reading durations.

## Dataset assumptions reflected in code

- 98 dyslexic and 88 control candidates are repeatedly assumed.
- Candidate sequence lengths appear grouped in bins corresponding to ~999, 1249, 1499, 1749, and 1999 samples.

## Typical tools used

- NumPy / Pandas / SciPy
- scikit-learn (`KMeans`, confusion matrices, accuracy)
- Matplotlib / Plotly (for plotting in notebooks)
