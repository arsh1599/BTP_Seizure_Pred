# Seizure Prediction using Machine Learning on EEG Signals

## Introduction

This is project is an attempt to predict epileptic seizure onset using machine learning techniques on EEG signals.

## Outline

### 1. Dataset

The [CHB-MIT Scalp EEG Database](https://archive.physionet.org/pn6/chbmit/) has been used in this project.

![dataset snapshot](https://user-images.githubusercontent.com/47525983/94988582-f7bec600-058b-11eb-8e1c-4a1512332fbb.png)

### 2. Feature extraction

Features have been extracted from the dataset using [mne](https://github.com/mne-tools/mne-python) and [pyeeg](https://github.com/forrestbao/pyeeg)

The extracted features are:

- Mean Variance
- Mean Kurtosis
- Mean Skewness
- Petrosian Fractal Dimension
- Hjorth Mobility
- Hjorth Complexity
- Mean Spectral Entropy

![extracted features snapshot](https://user-images.githubusercontent.com/47525983/94988626-33f22680-058c-11eb-9250-c5aaa3561139.png)

### 3. Classifiers

- SVM
- RNN
- R-LSTM
