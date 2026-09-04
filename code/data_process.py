"""Preprocess raw CSV recordings into 500-point waveforms for gas discrimination.

Pipeline: K-means anomaly removal -> per-window Min-Max normalization ->
air-baseline linear detrending -> interpolation to 500 points ->
Gaussian smoothing (sigma=1.0). Air samples use identity detrending.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_parser = argparse.ArgumentParser()
_parser.add_argument("--gas", default="H2S")
TYPE = _parser.parse_args().gas

edge_sec_dict = {"H2S": 0.2, "CH4": 0.15, "CO2": 0.25, "AIR": 1, "MIX": 0.2,
               "MIX_H2S_CO2": 0.2, "MIX_H2S_CH4": 0.2, "MIX_H2S_CH4_CO2": 0.2,
               "MIX_H2S_SO2": 0.2, "MIX_H2S_SO2_CO2": 0.2}
n_cluster_dict = {"H2S": 4, "CH4": 2, "CO2": 2, "AIR": 2, "MIX": 4,
                "MIX_H2S_CO2": 4, "MIX_H2S_CH4": 4, "MIX_H2S_CH4_CO2": 4,
                "MIX_H2S_SO2": 4, "MIX_H2S_SO2_CO2": 4}


def read_all_csvs(folder_path):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file, skiprows=3, header=None, names=["TIME(S)", "R(OHM)"])
        for col in ["TIME(S)", "R(OHM)"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        dfs.append(df)
    return dfs, csv_files


def detrend(df):
    coeffs = np.polyfit(df['TIME(S)'], df['R(OHM)'], deg=1)
    trend_line = np.polyval(coeffs, df['TIME(S)'])
    df['R(OHM)_detrended'] = df['R(OHM)'] - trend_line


def split_and_normalize(df, period_length=20, air_sec=10, all_air=False):
    periods = []
    max_time = df['TIME(S)'].max()

    for start in np.arange(0, max_time, period_length):
        end = start + period_length
        period = df[(df['TIME(S)'] >= start) & (df['TIME(S)'] < end)].copy()
        if period.empty:
            continue

        actual_duration = period['TIME(S)'].max() - period['TIME(S)'].min()
        if actual_duration < period_length * 0.8:
            continue

        period['R(OHM)_norm'] = (period['R(OHM)'] - period['R(OHM)'].min()) / \
                                (period['R(OHM)'].max() - period['R(OHM)'].min())

        if all_air:
            period['R(OHM)_detrended'] = period['R(OHM)_norm']
            periods.append(period)
            continue

        air_mask = period['TIME(S)'] < start + air_sec
        t_air = period.loc[air_mask, 'TIME(S)'].values
        y_air = period.loc[air_mask, 'R(OHM)_norm'].values

        if len(t_air) < 2:
            continue
        try:
            coeffs = np.polyfit(t_air - t_air[0], y_air, deg=1)
            t_full = period['TIME(S)'].values - t_air[0]
            trend = np.polyval(coeffs, t_full)
            period['R(OHM)_detrended'] = period['R(OHM)_norm'] - trend
            periods.append(period)
        except Exception:
            continue

    return periods


def align_time(df, target_len=500, sigma=1.0):
    t = df['TIME(S)'].values - df['TIME(S)'].iloc[0]
    y = df['R(OHM)_detrended'].values
    t_new = np.linspace(0, t[-1], target_len)
    y_new = np.interp(t_new, t, y)
    y_smooth = gaussian_filter1d(y_new, sigma=sigma)
    return pd.DataFrame({'TIME(S)': t_new, 'R(OHM)_detrended': y_smooth})


def remove_anomaly_10s_window(df, file, n_clusters=4, edge_sec=0.2):
    """Per 10s beat: find outliers within +/-edge_sec,
    remove span between earliest and latest outlier."""
    df = df.copy()
    df['diff'] = df['R(OHM)'].diff().abs().fillna(0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(df[['diff']])
    df['cluster'] = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()
    normal_label = int(np.argmin(np.abs(centers)))
    is_outlier = df['cluster'] != normal_label

    for beat_t in np.arange(10, df['TIME(S)'].max() + 10, 10):
        low, high = beat_t - edge_sec, beat_t + edge_sec
        mask_window = (df['TIME(S)'] >= low) & (df['TIME(S)'] <= high)
        window_df = df[mask_window]
        if window_df.empty:
            continue

        left_val = window_df['R(OHM)'].iloc[0]
        right_val = window_df['R(OHM)'].iloc[-1]
        low_bound, high_bound = (min(left_val, right_val), max(left_val, right_val))
        cond_outlier = (df['R(OHM)'] < low_bound) | (df['R(OHM)'] > high_bound)
        outlier_in_win = df[is_outlier & mask_window & cond_outlier]
        if outlier_in_win.empty:
            continue

        t_start = outlier_in_win['TIME(S)'].min()
        t_end = outlier_in_win['TIME(S)'].max()

        mask_bad = (df['TIME(S)'] >= t_start) & (df['TIME(S)'] <= t_end)
        df = df[~mask_bad]

        window_df['z_score'] = (window_df['R(OHM)'] - window_df['R(OHM)'].mean()) / window_df['R(OHM)'].std()
        window_df = window_df[np.abs(window_df['z_score']) < 3]
        df.loc[mask_window, 'R(OHM)'] = window_df['R(OHM)']

    return df.reset_index(drop=True)[['TIME(S)', 'R(OHM)']]


def preprocess_data(dfs, csv_files, n_clusters=4, edge_sec=2, period_length=20, time_length=500):
    all_periods = []
    for df, file in zip(dfs, csv_files):
        file = file.split('/')[-1]
        df = remove_anomaly_10s_window(df, file, n_clusters, edge_sec=edge_sec)
        all_air = True if TYPE == 'AIR' else False
        periods = split_and_normalize(df, period_length, all_air=all_air)
        for period in periods:
            period = align_time(period, target_len=500)
            all_periods.append(period)
    return all_periods


# Read and process all CSV files
folder_path = f"data/raw_data/{TYPE}"
dfs, csv_files = read_all_csvs(folder_path)
all_periods = preprocess_data(dfs, csv_files, n_clusters=n_cluster_dict.get(TYPE, 4),
                           edge_sec=edge_sec_dict.get(TYPE, 0.2), period_length=20, time_length=500)

# Save processed data
os.makedirs(f'data/{TYPE}/', exist_ok=True)
for i, period in enumerate(all_periods):
    period.to_csv(f'data/{TYPE}/period_{i+1:03d}.csv', index=False)
print(f"[{TYPE}] {len(all_periods)} periods saved to data/{TYPE}/")
