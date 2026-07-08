"""
data.py — загрузка и предобработка данных (ТЗ разделы 9.1, 9.2)
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SENSOR_COLS = ["vibration_rms", "vibration_peak", "current_a", "current_b",
               "current_c", "active_power_kw", "temperature",
               "pressure_in", "pressure_out", "flow_rate", "rpm"]

# служебные колонки — эталон/цели, НЕ признаки (защита от утечки)
TARGET_COLS = ["failure_flag", "failure_type", "days_to_failure", "health_index",
               "risk_7_days", "risk_14_days", "risk_30_days", "anomaly_status",
               "recommendation", "simulated_fault_severity"]


def load_timeseries(path=None):
    """Загрузка основной телеметрии (ТЗ 9.1: CSV)."""
    path = path or os.path.join(DATA_DIR, "synthetic_sensor_timeseries.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["equipment_id", "timestamp"]).reset_index(drop=True)


def load_registry(path=None):
    path = path or os.path.join(DATA_DIR, "equipment_registry.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def load_failures(path=None):
    path = path or os.path.join(DATA_DIR, "failure_history.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def load_alarms(path=None):
    path = path or os.path.join(DATA_DIR, "scada_alarms.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DataFrame(columns=["timestamp", "equipment_id", "alarm_code",
                                 "alarm_text", "alarm_level", "alarm_status"])


def preprocess(df):
    """
    Предобработка (ТЗ 9.2): очистка пропусков, обработка выбросов,
    синхронизация по времени, маркировка режимов.
    """
    df = df.copy()
    # 1. очистка пропусков в сенсорах — интерполяция внутри агрегата
    for c in SENSOR_COLS:
        df[c] = df.groupby("equipment_id")[c].transform(
            lambda s: s.interpolate(limit=3).ffill().bfill())
    # 2. обработка выбросов — клип по 1/99 перцентилям внутри агрегата
    for c in SENSOR_COLS:
        lo = df.groupby("equipment_id")[c].transform(lambda s: s.quantile(0.01))
        hi = df.groupby("equipment_id")[c].transform(lambda s: s.quantile(0.99))
        df[c] = df[c].clip(lo, hi)
    # 3. маркировка пусков/остановов (ТЗ 9.2)
    if "operating_mode" in df.columns:
        df["is_transient"] = df["operating_mode"].isin(
            ["startup", "shutdown", "standby", "maintenance"]).astype(int)
    else:
        df["is_transient"] = 0
    return df


def data_summary(df):
    """Сводка для дашборда."""
    return {
        "rows": len(df),
        "assets": df["equipment_id"].nunique(),
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "fault_windows": int((df["failure_type"] != "normal").sum()),
    }


if __name__ == "__main__":
    df = preprocess(load_timeseries())
    print("Загружено:", data_summary(df))
    print("Пропусков после обработки:", df[SENSOR_COLS].isna().sum().sum())
    print("Transient-окон:", int(df["is_transient"].sum()))
