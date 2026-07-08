"""
models.py — уровни диагностики (ТЗ раздел 10)
  10.1 Обнаружение аномалий -> normal/warning/anomaly/critical
  10.2 Health Index (0-100)
  10.3 Оценка риска отказа (7/14/30 дней)
  10.4 Классификация типа дефекта
"""
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from features import build_matrix, feature_columns


class DiagnosticEngine:
    """Единый движок диагностики: обучает все модели и выдаёт полную оценку."""

    def __init__(self):
        self.scaler = None
        self.svm = None          # ML-контур (One-Class SVM)
        self.clf = None
        self._smin = None
        self._smax = None
        self._thresholds = None
        self.classes_ = None
        self._w_svm = 0.6        # вес ML-контура в гибридном score
        self._w_cc = 0.4         # вес контрольных карт

    # --------------------------------------------------------------
    def fit(self, df):
        X, df = build_matrix(df)

        # здоровые окна для обучения детектора аномалий
        healthy = ((df["failure_type"] == "normal") &
                   (df["simulated_fault_severity"] < 0.1)).values

        # если меток нет (загружены данные без разметки) — почти всё "здорово";
        # тогда обучаемся на всех данных в unsupervised-режиме
        if healthy.mean() > 0.98 or healthy.sum() < 50:
            healthy = np.ones(len(df), dtype=bool)

        self.scaler = StandardScaler().fit(X[healthy])
        Xs = self.scaler.transform(X)
        Xs_h = self.scaler.transform(X[healthy])

        # 10.1 ГИБРИДНЫЙ детектор аномалий:
        #   - контрольные карты (3σ): быстрый прозрачный контур
        #   - One-Class SVM: чувствительный ML-контур
        # SVM квадратичен по числу точек, поэтому обучаем на подвыборке
        # здоровых (качество почти не теряется, скорость выше в разы).
        rng = np.random.RandomState(42)
        h_idx = np.where(healthy)[0]
        if len(h_idx) > 4000:
            h_idx = rng.choice(h_idx, size=4000, replace=False)
        Xs_train = self.scaler.transform(X.iloc[h_idx])
        self.svm = OneClassSVM(nu=0.05, gamma="scale").fit(Xs_train)

        raw = self._hybrid_raw(Xs)               # комбинированный сырой score
        self._smin, self._smax = raw[healthy].min(), raw.max()
        score01 = self._norm_score(raw)
        # пороги 4 уровней по перцентилям всего score
        self._thresholds = dict(
            warning=np.percentile(score01, 70),
            anomaly=np.percentile(score01, 85),
            critical=np.percentile(score01, 95))

        # 10.4 классификатор дефектов (обучаем на всех размеченных)
        self.clf = RandomForestClassifier(n_estimators=120, max_depth=12,
                                          random_state=42, n_jobs=-1)
        self.clf.fit(X, df["failure_type"])
        self.classes_ = self.clf.classes_
        return self

    def _norm_score(self, raw):
        return np.clip((raw - self._smin) / (self._smax - self._smin + 1e-9), 0, 1)

    def _hybrid_raw(self, Xs):
        """Гибридный сырой anomaly score = взвешенная сумма двух контуров.

        Контур 1 (контрольные карты): макс |z| по сенсорам — выход за 3σ.
        Контур 2 (One-Class SVM): -score_samples (больше = аномальнее).
        Оба нормируются к сопоставимому масштабу перед смешиванием.
        """
        # контур контрольных карт: максимальное отклонение по любому признаку
        cc = np.abs(Xs).max(axis=1)
        cc_n = cc / (cc.std() + 1e-9)

        # контур SVM
        svm = -self.svm.score_samples(Xs)
        svm_n = (svm - svm.min()) / (svm.max() - svm.min() + 1e-9)
        cc_n = (cc_n - cc_n.min()) / (cc_n.max() - cc_n.min() + 1e-9)

        return self._w_svm * svm_n + self._w_cc * cc_n

    # --------------------------------------------------------------
    def predict(self, df):
        """Возвращает df с колонками диагностики для каждого окна."""
        X, df = build_matrix(df)
        Xs = self.scaler.transform(X)

        raw = self._hybrid_raw(Xs)
        score01 = self._norm_score(raw)

        # 10.1 четыре уровня
        df = df.copy()
        df["anomaly_score"] = score01
        df["anomaly_level"] = [self._level(s) for s in score01]

        # 10.2 Health Index
        df["hi"] = (100 * (1 - score01)).round(1)

        # 10.3 риск (эвристика: растёт с score, разные горизонты)
        df["risk_7"] = np.clip(score01 ** 1.5, 0, 1).round(3)
        df["risk_14"] = np.clip(score01 ** 1.2, 0, 1).round(3)
        df["risk_30"] = np.clip(score01 ** 1.0, 0, 1).round(3)

        # 10.4 тип дефекта + уверенность
        probs = self.clf.predict_proba(X)
        idx = probs.argmax(axis=1)
        df["pred_fault"] = self.classes_[idx]
        df["pred_conf"] = probs[np.arange(len(idx)), idx].round(3)
        # если уровень аномалии normal — считаем состояние нормой
        df.loc[df["anomaly_level"] == "normal", "pred_fault"] = "normal"

        return df

    def _level(self, s):
        t = self._thresholds
        if s < t["warning"]:
            return "normal"
        if s < t["anomaly"]:
            return "warning"
        if s < t["critical"]:
            return "anomaly"
        return "critical"

    def feature_importance(self):
        return dict(zip(feature_columns(), self.clf.feature_importances_))


if __name__ == "__main__":
    from data import load_timeseries, preprocess
    df = preprocess(load_timeseries())
    eng = DiagnosticEngine().fit(df)
    out = eng.predict(df)
    print("Уровни аномалий:")
    print(out["anomaly_level"].value_counts().to_string())
    print("\nHealth Index диапазон:", out["hi"].min(), "-", out["hi"].max())
    print("\nТоп-5 признаков:")
    imp = sorted(eng.feature_importance().items(), key=lambda x: -x[1])[:5]
    for n, v in imp:
        print(f"  {n:20s} {v:.3f}")
