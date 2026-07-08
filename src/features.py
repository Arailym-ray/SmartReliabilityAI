"""
features.py — извлечение диагностических признаков (ТЗ 9.3)
"""
import numpy as np
import pandas as pd

from data import SENSOR_COLS

TREND_SIGNALS = ["vibration_rms", "current_a", "temperature", "flow_rate"]


def add_features(df):
    """
    Признаки из вибрации/тока/технологических сигналов (ТЗ 9.3):
    тренды, скорости роста, перекос фаз, отклонение мощности.
    """
    df = df.sort_values(["equipment_id", "timestamp"]).copy()

    # тренды и скорости роста (скользящее отклонение от baseline)
    for s in TREND_SIGNALS:
        df[f"{s}_trend"] = df.groupby("equipment_id")[s].transform(
            lambda x: x - x.rolling(12, min_periods=1).mean())
        df[f"{s}_roc"] = df.groupby("equipment_id")[s].transform(
            lambda x: x.diff().rolling(6, min_periods=1).mean())

    # перекос фаз тока (ТЗ 9.3: перекос фаз)
    if all(c in df.columns for c in ["current_a", "current_b", "current_c"]):
        ph = df[["current_a", "current_b", "current_c"]]
        df["current_imbalance"] = (ph.max(axis=1) - ph.min(axis=1)) / (ph.mean(axis=1) + 1e-6)

    # коэффициент ударности (crest factor) из вибрации
    df["crest_factor"] = df["vibration_peak"] / (df["vibration_rms"] + 1e-6)

    return df.fillna(0.0)


def feature_columns():
    cols = list(SENSOR_COLS)
    for s in TREND_SIGNALS:
        cols += [f"{s}_trend", f"{s}_roc"]
    cols += ["current_imbalance", "crest_factor"]
    return cols


def build_matrix(df):
    df = add_features(df)
    X = df[feature_columns()].fillna(0.0)
    return X, df


if __name__ == "__main__":
    from data import load_timeseries, preprocess
    X, df = build_matrix(preprocess(load_timeseries()))
    print("Признаков:", len(feature_columns()))
    print("Матрица:", X.shape)
    print("NaN:", X.isna().sum().sum())
