import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque


def preprocess_df(df):

    for col in df.columns:
        if col != 'target':
            df[col] = df[col].replace(0, np.nan).ffill()
            df[col] = df[col].pct_change()
        
    df.dropna(inplace=True)
    
    for col in df.columns:
        if col != 'target':
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

def generate_sequences(df, SEQ_LEN):
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    
    # random.shuffle(sequential_data)

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    
    return np.array(X), np.array(y)

def scale_matrix_minmax(X, x_min, x_max, range_max, range_min):
    nom = (X-range_min) * (x_max-x_min)
    denom = range_max - range_min
    denom = denom + (denom == 0)
    return x_min + nom/denom


def normalize_sequences(X, y):
    
    for seq in range(len(y)):
        # price_max = np.max(X[seq,:,:-1])
        # price_min = np.min(X[seq,:,:-1])

        highs_lows_closes = X[seq,:,:-1]
        # scaled_HLCs = scale_matrix_minmax(highs_lows_closes, 0, 1, price_max, price_min)
        price_mean = np.mean(highs_lows_closes)
        scaled_HLCs = highs_lows_closes / price_mean
        
        volumes = X[seq,:,-1:]
        # scaled_volumes = scale_matrix_minmax(volumes, 0, 1, volumes.max(), volumes.min())
        scaled_volumes = volumes / np.mean(volumes)
        
        X[seq] = np.append(scaled_HLCs, scaled_volumes, axis=1)
        # y[seq] = scale_matrix_minmax(y[seq], 0, 1, price_max, price_min)
        y[seq] = y[seq] / price_mean

    return X, y



