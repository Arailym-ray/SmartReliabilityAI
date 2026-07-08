"""
validation.py — оценка модели на данных симулятора.

Прогоняет контролируемые сценарии дефектов через диагностический движок
и считает метрики детекции, классификации и lead time (раннего
предупреждения). Используется во вкладке «Валидация» дашборда.

ВАЖНО (методологическая честность): это стресс-тест на контролируемых
сценариях, а не валидация на реальных промышленных данных. Симулятор и
классификатор используют согласованные сигнатуры дефектов, поэтому метрики
отражают способность системы отслеживать деградацию, а не обобщение на
незнакомые реальные сигналы.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from simulator import LiveSimulator, SENSOR_COLS

FAULTS = ["bearing_wear", "imbalance", "cavitation",
          "overload", "clogging", "motor_fault"]

FAULT_RU = {"bearing_wear": "Износ подшипника", "imbalance": "Дисбаланс",
            "cavitation": "Кавитация", "overload": "Перегрузка",
            "clogging": "Засорение", "motor_fault": "Дефект двигателя"}


def _run_one(engine, fault, seed, n=80, degrade_rate=0.02):
    """Один прогон сценария, возвращает пошаговый DataFrame диагностики."""
    sim = LiveSimulator(fault=fault, degrade_rate=degrade_rate, warmup=12, seed=seed)
    buf, rows = [], []
    for i in range(n):
        r = sim.step()
        row = {c: r[c] for c in SENSOR_COLS}
        row["timestamp"] = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i)
        row["equipment_id"] = "SIM"
        row["equipment_type"] = "pump"
        for c in ["failure_type", "anomaly_status", "recommendation"]:
            row[c] = "normal"
        for c in ["failure_flag", "days_to_failure", "health_index",
                  "risk_7_days", "risk_14_days", "risk_30_days",
                  "simulated_fault_severity", "operating_mode"]:
            row[c] = 0
        buf.append(row)
        if len(buf) < 3:
            continue
        last = engine.predict(pd.DataFrame(buf)).iloc[-1]
        rows.append(dict(step=i, severity=r["severity"], hi=last["hi"],
                         level=last["anomaly_level"], pred=last["pred_fault"]))
    return pd.DataFrame(rows)


def run_validation(engine, n_runs=3):
    """
    Прогоняет все дефекты × n_runs и считает метрики.
    Возвращает dict с: детекцией, классификацией, lead time, confusion matrix.
    """
    y_true, y_pred = [], []
    lead_times = []
    detect_flags = []
    per_fault = {}

    for fault in FAULTS:
        f_detect, f_lead, f_type = [], [], []
        for seed in range(n_runs):
            df = _run_one(engine, fault, seed)

            # 1. детекция: была ли тревога до "отказа" (severity>=0.8)
            failure = df[df["severity"] >= 0.8]
            alarm = df[df["level"].isin(["warning", "anomaly", "critical"])]
            detected = len(alarm) > 0
            f_detect.append(detected)
            detect_flags.append(detected)

            # 2. lead time: за сколько шагов ДО отказа первая тревога
            if len(failure) and len(alarm):
                lead = failure.iloc[0]["step"] - alarm.iloc[0]["step"]
                if lead >= 0:
                    lead_times.append(lead)
                    f_lead.append(lead)

            # 3. классификация: тип в поздней фазе (severity>0.6)
            late = df[df["severity"] > 0.6]
            if len(late):
                common = late["pred"].value_counts().index[0]
                y_true.append(fault)
                y_pred.append(common)
                f_type.append(common == fault)

        per_fault[fault] = dict(
            detect=np.mean(f_detect) if f_detect else 0,
            lead=np.mean(f_lead) if f_lead else 0,
            type_acc=np.mean(f_type) if f_type else 0,
        )

    # агрегированные метрики классификации
    labels = FAULTS
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return dict(
        detection_rate=np.mean(detect_flags),
        type_accuracy=np.mean([t == pd_ for t, pd_ in zip(y_true, y_pred)]),
        precision=p, recall=r, f1=f1,
        lead_mean=np.mean(lead_times) if lead_times else 0,
        lead_median=np.median(lead_times) if lead_times else 0,
        lead_min=min(lead_times) if lead_times else 0,
        lead_max=max(lead_times) if lead_times else 0,
        confusion=cm, labels=labels,
        per_fault=per_fault,
        n_runs=n_runs, n_total=len(FAULTS) * n_runs,
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load_timeseries, preprocess
    from models import DiagnosticEngine

    eng = DiagnosticEngine().fit(preprocess(load_timeseries()))
    res = run_validation(eng, n_runs=3)
    print(f"Детекция дефектов: {res['detection_rate']:.0%}")
    print(f"Точность типа: {res['type_accuracy']:.0%}")
    print(f"Macro-F1: {res['f1']:.3f}")
    print(f"Lead time: средний {res['lead_mean']:.0f}, медиана {res['lead_median']:.0f} шагов")
    print("Confusion matrix:")
    print(pd.DataFrame(res["confusion"], index=res["labels"], columns=res["labels"]))
