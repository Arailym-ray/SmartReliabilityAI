"""
simulator.py — симулятор потоковых данных оборудования в реальном времени.

Генерирует показания датчиков «на лету» с закладываемым дефектом и
прогрессирующей деградацией. Используется во вкладке симуляции дашборда:
каждый вызов step() возвращает новое показание, как будто пришло с датчика.
"""
import numpy as np

# базовые (здоровые) уровни сигналов — согласованы с датасетом
BASELINE = dict(
    vibration_rms=1.4, vibration_peak=4.8, current_a=95.0, current_b=95.0,
    current_c=95.0, active_power_kw=42.0, temperature=49.0,
    pressure_in=2.5, pressure_out=5.0, flow_rate=136.0, rpm=1184.0,
)

# сигнатуры дефектов подогнаны под РЕАЛЬНЫЕ паттерны обучающего датасета,
# чтобы классификатор корректно распознавал тип дефекта.
# p — степень деградации [0,1]; целевые значения = baseline + p*(target-baseline).
FAULT_SIGNATURES = {
    "normal": lambda b, p, rng: {},
    # bearing_wear: вибрация↑, temp↑, rpm↓, ток слабо↓
    "bearing_wear": lambda b, p, rng: dict(
        vibration_rms=b["vibration_rms"] + p * 2.3 + rng.normal(0, 0.15),
        vibration_peak=b["vibration_peak"] + p * 11.5,
        temperature=b["temperature"] + p * 4.4,
        rpm=b["rpm"] - p * 84,
        current_a=b["current_a"] - p * 9,
    ),
    # imbalance: вибрация сильно↑, ток↑, rpm сильно↓
    "imbalance": lambda b, p, rng: dict(
        vibration_rms=b["vibration_rms"] + p * 3.1 + rng.normal(0, 0.15),
        vibration_peak=b["vibration_peak"] + p * 10.3,
        current_a=b["current_a"] + p * 33,
        active_power_kw=b["active_power_kw"] + p * 20.6,
        rpm=b["rpm"] - p * 338,
    ),
    # cavitation: ток резко↑, вибрация↑, pressure↓, flow↑
    "cavitation": lambda b, p, rng: dict(
        vibration_rms=b["vibration_rms"] + p * 2.3 + abs(rng.normal(0, 0.5)) * p,
        vibration_peak=b["vibration_peak"] + p * 9.8,
        current_a=b["current_a"] + p * 65 + rng.normal(0, 2),
        active_power_kw=b["active_power_kw"] + p * 31,
        pressure_out=b["pressure_out"] - p * 0.5,
        rpm=b["rpm"] + p * 294,
    ),
    # overload: ток↑, temp сильно↑, power↑
    "overload": lambda b, p, rng: dict(
        current_a=b["current_a"] + p * 18 + rng.normal(0, 1),
        current_b=b["current_b"] + p * 18,
        current_c=b["current_c"] + p * 18,
        active_power_kw=b["active_power_kw"] + p * 7.4,
        temperature=b["temperature"] + p * 14.9,
        rpm=b["rpm"] - p * 75,
    ),
    # clogging: падение расхода + умеренная вибрация для детекции.
    # Может иногда классифицироваться как bearing_wear (близкие профили) —
    # но детектор аномалий и падение расхода на графике видны чётко.
    "clogging": lambda b, p, rng: dict(
        flow_rate=b["flow_rate"] - p * 55,
        current_a=b["current_a"] - p * 31,
        current_b=b["current_b"] - p * 31,
        current_c=b["current_c"] - p * 31,
        active_power_kw=b["active_power_kw"] - p * 12,
        pressure_out=b["pressure_out"] + p * 2.2,
        rpm=b["rpm"] + p * 638,
        temperature=b["temperature"] + p * 4.7,
        vibration_rms=b["vibration_rms"] + p * 1.2,
    ),
    # motor_fault: temp сильно↑, rpm сильно↓, power НЕ растёт (отличие от overload)
    "motor_fault": lambda b, p, rng: dict(
        temperature=b["temperature"] + p * 16,
        current_a=b["current_a"] + rng.normal(0, 6) * p,
        current_b=b["current_b"] - p * 14,
        current_c=b["current_c"] + p * 10,
        active_power_kw=b["active_power_kw"] - p * 6,
        pressure_out=b["pressure_out"] - p * 2.4,
        rpm=b["rpm"] - p * 478,
        vibration_rms=b["vibration_rms"] + p * 0.8,
    ),
}

SENSOR_COLS = list(BASELINE.keys())


class LiveSimulator:
    """
    Пошаговый генератор потока. degrade_rate — как быстро развивается дефект
    (доля деградации за шаг). Дефект начинает проявляться после warmup шагов.
    """

    def __init__(self, fault="bearing_wear", degrade_rate=0.015,
                 noise=0.03, warmup=15, seed=None):
        self.fault = fault
        self.degrade_rate = degrade_rate
        self.noise = noise
        self.warmup = warmup
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.severity = 0.0

    def reset(self):
        self.t = 0
        self.severity = 0.0

    def step(self):
        """Одно новое показание датчиков (dict)."""
        self.t += 1
        # деградация начинается после warmup
        if self.t > self.warmup and self.fault != "normal":
            self.severity = min(1.0, self.severity + self.degrade_rate)

        b = BASELINE
        # базовый сигнал + шум
        reading = {c: b[c] * (1 + self.rng.normal(0, self.noise)) for c in SENSOR_COLS}
        # накладываем сигнатуру дефекта
        sig = FAULT_SIGNATURES[self.fault](b, self.severity, self.rng)
        reading.update(sig)
        # добавляем немного шума к изменённым сигналам
        for c in sig:
            reading[c] += self.rng.normal(0, self.noise * abs(b[c]) * 0.3)
        reading["severity"] = round(self.severity, 3)
        reading["step"] = self.t
        return reading


if __name__ == "__main__":
    sim = LiveSimulator(fault="bearing_wear", degrade_rate=0.03, seed=1)
    print("Симуляция износа подшипника (каждый 10-й шаг):")
    print(f"{'шаг':>4} {'severity':>8} {'vib_rms':>8} {'temp':>6} {'current':>8}")
    for _ in range(50):
        r = sim.step()
        if r["step"] % 10 == 0:
            print(f"{r['step']:>4} {r['severity']:>8.2f} {r['vibration_rms']:>8.2f} "
                  f"{r['temperature']:>6.1f} {r['current_a']:>8.1f}")
