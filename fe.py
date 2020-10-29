#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import mne
import pandas as pd
import sys

import csv
from scipy.stats import skew, kurtosis
import pyeeg as p
from numpy import nan
import math


# In[2]:


def mean_variance(df):
    variance_vals = np.var(df)
    return np.mean(variance_vals)


# In[3]:


def mean_kurtosis(df):
    kurtosis_vals = kurtosis(df)
    return np.mean(kurtosis_vals)


# In[4]:


def mean_skewness(df):
    skew_vals = skew(df)
    return np.mean(skew_vals)


# In[5]:


def mean_pfd(df):
    pfd_vals = []
    for col in df.columns:
        col = df[col].to_numpy()
        pfd_val = p.pfd(col)
        pfd_vals.append(pfd_val)
    return np.mean(pfd_vals)


# In[6]:


def mean_hjorth_mob_comp(df):
    mob_vals = []
    comp_vals = []
    for col in df.columns:
        col = df[col].to_numpy()
        mob_col, comp_col = p.hjorth(col)
        mob_vals.append(mob_col)
        comp_vals.append(comp_col)
    return np.mean(mob_vals), np.mean(comp_vals)


# In[7]:


def all_psd(data):
    fs = 256
    N = data.shape[1]  # total num of points

    # Get only in postive frequencies
    fft_vals = np.absolute(np.fft.rfft(data))

    n_rows = fft_vals.shape[0]
    n_cols = fft_vals.shape[1]
    psd_vals = np.zeros(shape=(n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            psd_vals[i][j] = (N/fs) * fft_vals[i][j] * fft_vals[i][j]

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(data.shape[1], 1.0/fs)

    # Define EEG bands
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}

    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    psd_vals_list = []
    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                           (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(psd_vals[:, freq_ix])
        psd_vals_list.append(eeg_band_fft[band] * 1000000)
    return psd_vals_list


# In[8]:


def sum_psd(data):
    psd_vals = all_psd(data)
    return np.sum(psd_vals)


# In[9]:


def mean_spectral_entropy(data):
    psd_vals = all_psd(data)
    power_ratio = []
    sum_psd_vals = sum_psd(data)
    for val in psd_vals:
        power_ratio.append(val/sum_psd_vals)
    bands = [0, 4, 8, 12, 30, 45]
    Fs = 256
    spec_entropy = p.spectral_entropy(data, bands, Fs, power_ratio)
    return spec_entropy


# In[168]:


def add_row(df_input, psd_ip, start, end, duration, index, seizure):
    row_to_add = []
    d = df_input[index:index + duration]
    psd_ip = psd_ip[:, start:end]
    psd_ip = psd_ip[:][0]

    mean_var = mean_variance(d)
    mean_k = mean_kurtosis(d)
    mean_skew = mean_skewness(d)
    pfd = mean_pfd(d)
    h_mob, h_comp = mean_hjorth_mob_comp(d)
    mean_spec = mean_spectral_entropy(psd_ip)

    row_to_add.append(mean_var)
    row_to_add.append(mean_k)
    row_to_add.append(mean_skew)
    row_to_add.append(pfd)
    row_to_add.append(h_mob)
    row_to_add.append(h_comp)
    row_to_add.append(mean_spec)
    # Label: 1 = seizure, 0 = non-seizure. Change before running.
    row_to_add.append(seizure)

    return row_to_add


# In[169]:

def read_data(patient, file):
    data = mne.io.read_raw_edf(
        'data/{patient}/{file}.edf'.format(patient=patient, file=file))
    # exclude =  Channel order : 'FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2','FZ-CZ','CZ-PZ','P7-T7','T7-FT9','FT9-FT10','FT10-T8','T8-P8'
    header = ','.join(data.ch_names)
    df = pd.DataFrame(data[:][0])
    df = df.transpose()

    return df, data
    # df = df.iloc[1467*256:1477*256]
    # df3


# In[170]:


# for 11 , 14 , 20
# included channels: 0, 5, 10, 15, 19, 25
def drop_11(df):
    df.drop([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18,
             20, 21, 22, 23, 24, 26, 27], axis=1, inplace=True)
    df.shape


# In[127]:


# #for 1 and 8

# included channels: 0, 4, 8, 12, 14, 20
def drop_8(df):
    df.drop([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 15, 16,
             17, 18, 19, 21, 22], axis=1, inplace=True)
    df.shape


# In[165]:

def trim(df, start_time, end_time):
    start = temp = start_time*256
    end = end_time*256

    df1 = df.iloc[start:end, :]
    df1 = pd.DataFrame(df1)
    df1.shape

    return df1, start, end, temp


# In[166]:


# processes one seizure in 10s windows
# adds rows with features extracted from these windows
# print(df1)
def write_to_file(file, mode, start, end, temp, df1, data, seizure):
    index = 0
    duration = 10*256
    res = pd.DataFrame()

    # first iteration run in 'w' mode, all subsequent iteration run in 'a' mode
    with open(file, mode) as file:
        writer = csv.writer(file)
        while temp < end:
            row = add_row(df1, data, temp, temp + duration, duration, index, seizure)
            res = res.append(pd.Series(row), ignore_index=True)
            writer.writerow(row)
            temp += duration
            index += duration

    #res.columns = ['Variance', 'Kurtosis', 'Skewness', 'Petrosian Fractal Dimension', 'Hjorth Mobility', 'Hjorth Complexity', 'Spectral Entropy', 'Label']
    res


# In[167]:



# In[ ]:
