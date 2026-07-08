"""
fft_analysis.py — частотный анализ вибрации (вибродиагностика).

Анализирует сырой вибросигнал (ускорение) и извлекает частотные признаки
дефектов: FFT-спектр, RMS по частотным диапазонам, kurtosis, crest factor,
гармоники оборотной частоты. Для вращающегося оборудования частотный анализ
выявляет дефекты подшипников, дисбаланс и несоосность раньше, чем анализ
мгновенных значений.
"""
import os
import numpy as np
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                        "synthetic_raw_vibration_PUMP-204.csv")
FS = 20000  # частота дискретизации, Гц


def load_raw(path=None):
    path = path or RAW_PATH
    return pd.read_csv(path)


def compute_fft(signal, fs=FS):
    """FFT-спектр сигнала. Возвращает частоты и амплитуды."""
    n = len(signal)
    windowed = signal * np.hanning(n)
    amp = np.abs(np.fft.rfft(windowed)) / n * 2
    freqs = np.fft.rfftfreq(n, 1 / fs)
    return freqs, amp


def band_rms(signal, fs=FS):
    """RMS энергии по частотным диапазонам (зоны дефектов)."""
    freqs, amp = compute_fft(signal, fs)
    bands = {
        "0-50 Гц (оборотная)": (0, 50),
        "50-200 Гц (гармоники)": (50, 200),
        "200-1000 Гц (подшипник)": (200, 1000),
        "1000+ Гц (удары)": (1000, fs / 2),
    }
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.sqrt(np.sum(amp[mask] ** 2)))
    return out


def time_features(signal):
    """Временные признаки: RMS, peak, crest factor, kurtosis."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest = peak / (rms + 1e-9)
    # kurtosis (эксцесс) — чувствителен к ударным дефектам подшипника
    m = signal.mean()
    kurt = float(np.mean((signal - m) ** 4) / (np.var(signal) ** 2 + 1e-9) - 3)
    return dict(rms=rms, peak=peak, crest=crest, kurtosis=kurt)


def top_peaks(signal, fs=FS, fmax=2000, n_peaks=5):
    """Топ-N частотных пиков до fmax Гц."""
    freqs, amp = compute_fft(signal, fs)
    mask = freqs < fmax
    fm, am = freqs[mask], amp[mask]
    idx = np.argsort(am)[-n_peaks:][::-1]
    return [(float(fm[i]), float(am[i])) for i in idx]


def analyze_all_snapshots(df=None, fs=FS):
    """Анализ всех снапшотов (стадий деградации). Для дашборда."""
    df = df if df is not None else load_raw()
    rows = []
    spectra = {}
    for snap in sorted(df["snapshot"].unique()):
        seg = df[df["snapshot"] == snap]
        sig = seg["accel_g"].values
        sev = float(seg["severity"].iloc[0])
        tf = time_features(sig)
        bands = band_rms(sig, fs)
        peaks = top_peaks(sig, fs)
        rows.append({
            "snapshot": int(snap), "severity": sev,
            "rms": round(tf["rms"], 3), "crest": round(tf["crest"], 2),
            "kurtosis": round(tf["kurtosis"], 2),
            "peak_freq": round(peaks[0][0], 0),
            **{k: round(v, 3) for k, v in bands.items()},
        })
        freqs, amp = compute_fft(sig, fs)
        spectra[int(snap)] = dict(severity=sev,
                                  freqs=freqs[freqs < 2000].tolist(),
                                  amp=amp[freqs < 2000].tolist())
    return pd.DataFrame(rows), spectra


if __name__ == "__main__":
    table, spectra = analyze_all_snapshots()
    print("Частотный анализ по стадиям деградации:")
    print(table.to_string(index=False))
    print("\nВывод: с ростом severity растёт RMS, появляются высокочастотные")
    print("гармоники (признак дефекта подшипника), crest factor растёт.")
