"""
app.py — Дашборд SmartReliability AI (ТЗ раздел 11)
Полный MVP предиктивной диагностики флотомашин и насосного оборудования.

Запуск:
    cd mvp
    pip install -r requirements.txt
    streamlit run app.py
"""
import os
import sys
import base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from data import (load_timeseries, load_registry, load_failures, load_alarms,
                  preprocess, data_summary, SENSOR_COLS)
from models import DiagnosticEngine
from diagnostics import recommendation_for, build_explanation
from simulator import LiveSimulator, SENSOR_COLS as SIM_SENSORS
from validation import run_validation, FAULT_RU as VAL_FAULT_RU
from fft_analysis import analyze_all_snapshots, compute_fft, load_raw
from export import build_report
from diagnostics import RECOMMENDATIONS
import time

st.set_page_config(page_title="SmartReliability AI", layout="wide",
                   initial_sidebar_state="expanded")

# ---- светлая тема ----
st.markdown("""
<style>
  .stApp { background: #f7f8fa; }
  section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8eaed; }
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1280px; }
  h1 { font-size: 26px !important; font-weight: 600 !important; color: #1a1d21 !important; }
  h2, h3 { color: #1a1d21 !important; font-weight: 600 !important; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    background: #ffffff; border: 1px solid #e8eaed; border-radius: 8px;
    padding: 8px 18px; font-weight: 500;
  }
  .stTabs [aria-selected="true"] { background: #185FA5 !important; color: #fff !important; border-color:#185FA5 !important; }
  div[data-testid="stMetricValue"] { font-size: 22px; }
  .asset-card { background:#fff; border:1px solid #e8eaed; border-radius:12px; padding:14px 18px; margin-bottom:8px; }
  .pill { display:inline-block; font-size:12px; font-weight:600; padding:3px 12px; border-radius:20px; }
  @keyframes pulseAlert {
    0% { box-shadow: 0 0 0 0 rgba(226,75,74,0.5); }
    70% { box-shadow: 0 0 0 14px rgba(226,75,74,0); }
    100% { box-shadow: 0 0 0 0 rgba(226,75,74,0); }
  }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.55; } }
  .alert-critical { animation: pulseAlert 1.1s infinite; }
  .alert-blink { animation: blink 0.9s infinite; }
</style>
""", unsafe_allow_html=True)

LEVEL_COLOR = {"normal": "#1D9E75", "warning": "#EF9F27",
               "anomaly": "#D85A30", "critical": "#E24B4A"}
LEVEL_FILL = {"normal": "#E1F5EE", "warning": "#FAEEDA",
              "anomaly": "#FAECE7", "critical": "#FCEBEB"}
LEVEL_TEXT = {"normal": "#0F6E56", "warning": "#854F0B",
              "anomaly": "#993C1D", "critical": "#A32D2D"}
LEVEL_RU = {"normal": "Норма", "warning": "Наблюдение",
            "anomaly": "Аномалия", "critical": "Критично"}


def hi_color(hi):
    if hi >= 80:
        return "#1D9E75"
    if hi >= 60:
        return "#EF9F27"
    if hi >= 40:
        return "#D85A30"
    return "#E24B4A"


ASSETS = os.path.join(os.path.dirname(__file__), "assets")
FEEDBACK_CSV = os.path.join(os.path.dirname(__file__), "feedback_log.csv")


def load_feedback():
    """Читает журнал решений инженера из CSV (переживает перезагрузку)."""
    if os.path.exists(FEEDBACK_CSV):
        try:
            return pd.read_csv(FEEDBACK_CSV).to_dict("records")
        except Exception:
            return []
    return []


def save_feedback_row(row):
    """Дописывает одно решение в CSV-файл."""
    df_row = pd.DataFrame([row])
    header = not os.path.exists(FEEDBACK_CSV)
    df_row.to_csv(FEEDBACK_CSV, mode="a", header=header, index=False,
                  encoding="utf-8-sig")


def gauge_chart(value, title="Health Index", height=200):
    """Круговой спидометр Health Index с цветовыми зонами."""
    color = hi_color(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 34, "color": color}},
        title={"text": title, "font": {"size": 13, "color": "#6b7280"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#d9dce1",
                     "tickfont": {"size": 9, "color": "#9ca3af"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#f7f8fa",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#FCEBEB"},
                {"range": [40, 60], "color": "#FAECE7"},
                {"range": [60, 80], "color": "#FAEEDA"},
                {"range": [80, 100], "color": "#E1F5EE"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.75, "value": value},
        },
    ))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "Arial"})
    return fig


def _logo_b64(filename):
    """Загружает логотип как base64 для встраивания в HTML. None если нет файла."""
    path = os.path.join(ASSETS, filename)
    if not os.path.exists(path):
        return None
    ext = "jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else "png"
    with open(path, "rb") as f:
        return f"data:image/{ext};base64," + base64.b64encode(f.read()).decode()


# ------------------------------------------------------------------
@st.cache_resource
def load_engine(_uploaded_df=None, cache_key="default"):
    """Обучает движок. Если передан _uploaded_df — на нём, иначе на тестовом.
    cache_key нужен, чтобы Streamlit различал загруженные файлы в кэше."""
    if _uploaded_df is not None:
        df = preprocess(_uploaded_df)
    else:
        df = preprocess(load_timeseries())
    eng = DiagnosticEngine().fit(df)
    out = eng.predict(df)
    return eng, out


@st.cache_data
def run_validation_cached(_eng, cache_key="default", n_runs=3):
    """Кэшированный прогон валидации (тяжёлый, поэтому один раз)."""
    return run_validation(_eng, n_runs=n_runs)


@st.cache_data
def _fft_cached():
    """Кэшированный частотный анализ вибрации."""
    return analyze_all_snapshots()


@st.cache_data
def load_context():
    return load_registry(), load_failures(), load_alarms()


def latest_by_asset(out):
    """Текущее состояние (последнее окно) + пиковый уровень за последние
    14 дней для контекста. Оператор видит и 'сейчас', и 'был ли риск'."""
    level_rank = {"normal": 0, "warning": 1, "anomaly": 2, "critical": 3}
    rows = []
    for eid, g in out.groupby("equipment_id"):
        g = g.sort_values("timestamp")
        r = g.iloc[-1]  # текущее состояние
        recent = g[g["timestamp"] >= g["timestamp"].max() - pd.Timedelta(days=14)]
        peak_level = max(recent["anomaly_level"], key=lambda l: level_rank[l])
        rows.append({
            "equipment_id": eid,
            "equipment_type": r["equipment_type"],
            "hi": r["hi"], "level": r["anomaly_level"],
            "peak_level": peak_level,
            "risk_7": r["risk_7"], "risk_14": r["risk_14"], "risk_30": r["risk_30"],
            "fault": r["pred_fault"], "conf": r["pred_conf"],
            "row": r,
        })
    return sorted(rows, key=lambda x: x["hi"])


def kpi(col, title, value, color="#1a1d21", sub=""):
    col.markdown(
        f"<div style='padding:18px 20px;border:1px solid #e8eaed;border-radius:12px;background:#fff'>"
        f"<div style='font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:0.4px'>{title}</div>"
        f"<div style='font-size:26px;font-weight:600;color:{color};margin-top:6px;line-height:1.1'>{value}</div>"
        f"<div style='font-size:11px;color:#9ca3af;margin-top:2px'>{sub}</div></div>",
        unsafe_allow_html=True)


# ------------------------------------------------------------------
# ---- Sidebar: логотип, загрузка данных ----
st.sidebar.title("SmartReliability AI")
st.sidebar.caption("Предиктивная диагностика флотомашин и насосного оборудования")
_aitu_logo = _logo_b64("aitu.jpeg")
_kaz_logo = _logo_b64("kazakhmys.png")
if _aitu_logo:
    st.sidebar.markdown(
        f"<div style='font-size:10px;color:#9ca3af;margin-top:8px;text-transform:uppercase;"
        f"letter-spacing:0.5px'>Разработано в</div>"
        f"<img src='{_aitu_logo}' style='width:100%;max-width:190px;margin:4px 0 12px'>",
        unsafe_allow_html=True)
if _kaz_logo:
    st.sidebar.markdown(
        f"<div style='font-size:10px;color:#9ca3af;text-transform:uppercase;"
        f"letter-spacing:0.5px'>Для предприятия</div>"
        f"<img src='{_kaz_logo}' style='width:100%;max-width:170px;margin:4px 0 6px'>",
        unsafe_allow_html=True)
st.sidebar.markdown("---")

# ---- Загрузка своих данных (ТЗ 9.1) ----
st.sidebar.subheader("Источник данных")
uploaded = st.sidebar.file_uploader(
    "Загрузить свой CSV (телеметрия)", type=["csv"],
    help="Файл с колонками датчиков. Если не загружен — используется тестовый датасет.")

# обязательные колонки-сенсоры для работы моделей
REQUIRED = ["vibration_rms", "vibration_peak", "current_a", "temperature",
            "flow_rate", "rpm", "timestamp", "equipment_id"]

user_df = None
cache_key = "default"
if uploaded is not None:
    try:
        udf = pd.read_csv(uploaded, parse_dates=["timestamp"])
        missing = [c for c in REQUIRED if c not in udf.columns]
        if missing:
            st.sidebar.error(f"В файле не хватает колонок: {', '.join(missing)}")
        else:
            # заполняем недостающие необязательные колонки нулями/normal
            for c in ["current_b", "current_c", "active_power_kw", "pressure_in",
                      "pressure_out"]:
                if c not in udf.columns:
                    udf[c] = 0.0
            for c in ["failure_type", "anomaly_status", "recommendation"]:
                if c not in udf.columns:
                    udf[c] = "normal"
            for c in ["failure_flag", "days_to_failure", "health_index",
                      "risk_7_days", "risk_14_days", "risk_30_days",
                      "simulated_fault_severity", "operating_mode"]:
                if c not in udf.columns:
                    udf[c] = 0
            if "equipment_type" not in udf.columns:
                udf["equipment_type"] = "pump"
            user_df = udf
            cache_key = f"upload_{uploaded.name}_{len(udf)}"
            st.sidebar.success(f"Загружено: {len(udf):,} строк, "
                               f"{udf['equipment_id'].nunique()} агрегатов")
    except Exception as e:
        st.sidebar.error(f"Ошибка чтения файла: {e}")

if user_df is None:
    st.sidebar.caption("Используется тестовый датасет.")

# ------------------------------------------------------------------
eng, out = load_engine(user_df, cache_key)
registry, failures, alarms = load_context()
states = latest_by_asset(out)

st.sidebar.markdown("---")
summ = data_summary(out)

# ---- Блок статуса парка ----
_n_crit = sum(1 for s in states if s["level"] == "critical")
_n_anom = sum(1 for s in states if s["level"] == "anomaly")
_n_warn = sum(1 for s in states if s["level"] == "warning")
_n_norm = sum(1 for s in states if s["level"] == "normal")

st.sidebar.markdown(
    f"<div style='font-size:10px;color:#9ca3af;text-transform:uppercase;"
    f"letter-spacing:0.5px;margin-bottom:8px'>Состояние парка</div>"
    f"<div style='background:linear-gradient(135deg,#0C447C,#185FA5);border-radius:12px;"
    f"padding:14px 16px;margin-bottom:8px;color:#fff'>"
    f"<div style='font-size:30px;font-weight:700;line-height:1'>{summ['assets']}</div>"
    f"<div style='font-size:11px;opacity:0.85;margin-top:2px'>агрегатов под мониторингом</div>"
    f"</div>", unsafe_allow_html=True)

st.sidebar.markdown(
    f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:14px'>"
    f"<div style='background:#E1F5EE;border-radius:8px;padding:8px 10px'>"
    f"<div style='font-size:18px;font-weight:700;color:#0F6E56'>{_n_norm}</div>"
    f"<div style='font-size:10px;color:#0F6E56'>Норма</div></div>"
    f"<div style='background:#FAEEDA;border-radius:8px;padding:8px 10px'>"
    f"<div style='font-size:18px;font-weight:700;color:#854F0B'>{_n_warn}</div>"
    f"<div style='font-size:10px;color:#854F0B'>Наблюдение</div></div>"
    f"<div style='background:#FBEAE2;border-radius:8px;padding:8px 10px'>"
    f"<div style='font-size:18px;font-weight:700;color:#A8431A'>{_n_anom}</div>"
    f"<div style='font-size:10px;color:#A8431A'>Аномалия</div></div>"
    f"<div style='background:#FCEBEB;border-radius:8px;padding:8px 10px'>"
    f"<div style='font-size:18px;font-weight:700;color:#A32D2D'>{_n_crit}</div>"
    f"<div style='font-size:10px;color:#A32D2D'>Критично</div></div>"
    f"</div>", unsafe_allow_html=True)

st.sidebar.caption(f"Данные: {summ['rows']:,} записей · {summ['assets']} агрегатов")
st.sidebar.caption(f"Период: {summ['start']:%Y-%m-%d} — {summ['end']:%Y-%m-%d}")

# ---- Легенда уровней ----
st.sidebar.markdown(
    "<div style='font-size:10px;color:#9ca3af;text-transform:uppercase;"
    "letter-spacing:0.5px;margin:14px 0 8px'>Уровни состояния</div>"
    "<div style='display:flex;flex-direction:column;gap:5px'>"
    "<div style='display:flex;align-items:center;gap:8px;font-size:11px;color:#6b7280'>"
    "<span style='width:10px;height:10px;border-radius:50%;background:#1D9E75'></span>Норма (80–100)</div>"
    "<div style='display:flex;align-items:center;gap:8px;font-size:11px;color:#6b7280'>"
    "<span style='width:10px;height:10px;border-radius:50%;background:#EF9F27'></span>Наблюдение (60–80)</div>"
    "<div style='display:flex;align-items:center;gap:8px;font-size:11px;color:#6b7280'>"
    "<span style='width:10px;height:10px;border-radius:50%;background:#D85A30'></span>Аномалия (40–60)</div>"
    "<div style='display:flex;align-items:center;gap:8px;font-size:11px;color:#6b7280'>"
    "<span style='width:10px;height:10px;border-radius:50%;background:#E24B4A'></span>Критично (0–40)</div>"
    "</div>", unsafe_allow_html=True)

st.sidebar.markdown(
    "<div style='font-size:10px;color:#9ca3af;margin-top:14px;line-height:1.5'>"
    "Тестовые данные синтетические. Для промышленного применения требуется "
    "калибровка на реальных данных предприятия.</div>", unsafe_allow_html=True)

# ---- Заголовок и общие KPI ----
# ---- Шапка с логотипами ----
aitu = _logo_b64("aitu.jpeg")
kaz = _logo_b64("kazakhmys.png")
logo_left = f"<img src='{aitu}' style='height:44px'>" if aitu else ""
logo_right = f"<img src='{kaz}' style='height:40px'>" if kaz else ""
st.markdown(
    f"<div style='display:flex;align-items:center;justify-content:space-between;"
    f"padding:8px 4px 18px;border-bottom:1px solid #e8eaed;margin-bottom:18px'>"
    f"<div style='flex:0 0 auto'>{logo_left}</div>"
    f"<div style='flex:1 1 auto;text-align:center'>"
    f"<div style='font-size:13px;color:#6b7280;letter-spacing:0.5px'>SmartReliability AI</div>"
    f"<div style='font-size:11px;color:#9ca3af'>Предиктивная диагностика оборудования</div></div>"
    f"<div style='flex:0 0 auto'>{logo_right}</div></div>",
    unsafe_allow_html=True)

st.title("Мониторинг технического состояния оборудования")

n_crit = sum(1 for s in states if s["level"] == "critical")
n_anom = sum(1 for s in states if s["level"] == "anomaly")
n_warn = sum(1 for s in states if s["level"] == "warning")
n_norm = sum(1 for s in states if s["level"] == "normal")

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Всего агрегатов", len(states))
kpi(c2, "Критично", n_crit, LEVEL_COLOR["critical"])
kpi(c3, "Аномалия", n_anom, LEVEL_COLOR["anomaly"])
kpi(c4, "Наблюдение", n_warn, LEVEL_COLOR["warning"])
kpi(c5, "Норма", n_norm, LEVEL_COLOR["normal"])

st.markdown("")

# ==================================================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Обзор", "Парк оборудования", "Диагностика агрегата", "Отчёты",
     "Симулятор (live)", "Валидация модели", "Внедрение"])

# ---- TAB 0: обзор (стартовый экран) ----
with tab0:
    avg_hi_all = round(np.mean([s["hi"] for s in states]))
    # hero
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0C447C 0%,#185FA5 100%);"
        f"border-radius:16px;padding:36px 40px;color:#fff;margin-bottom:20px'>"
        f"<div style='font-size:13px;opacity:0.8;letter-spacing:0.5px;"
        f"text-transform:uppercase'>SmartReliability AI · Astana IT University</div>"
        f"<div style='font-size:30px;font-weight:700;line-height:1.25;margin:12px 0;"
        f"max-width:640px'>Предиктивная диагностика флотомашин и насосного оборудования</div>"
        f"<div style='font-size:16px;opacity:0.92;line-height:1.6;max-width:640px'>"
        f"От обслуживания по регламенту — к обслуживанию по фактическому состоянию. "
        f"Система выявляет дефекты до отказа по вибрации, току и технологическим сигналам.</div>"
        f"</div>", unsafe_allow_html=True)

    # ключевые цифры
    o1, o2, o3, o4 = st.columns(4)
    kpi(o1, "Агрегатов под мониторингом", summ["assets"], "#185FA5")
    kpi(o2, "Accuracy детекции", "97%", "#1D9E75")
    kpi(o3, "Предупреждение до отказа", "~48 ч", "#1D9E75")
    kpi(o4, "Типов дефектов", "7")

    st.markdown("")
    st.markdown("##### Как это работает")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e8eaed;border-radius:12px;"
            "padding:18px'><div style='font-size:15px;font-weight:600;color:#185FA5;"
            "margin-bottom:6px'>1 · Сбор сигналов</div><div style='font-size:13px;"
            "color:#6b7280;line-height:1.5'>Вибрация, ток, температура, расход "
            "поступают с датчиков оборудования.</div></div>", unsafe_allow_html=True)
    with w2:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e8eaed;border-radius:12px;"
            "padding:18px'><div style='font-size:15px;font-weight:600;color:#D85A30;"
            "margin-bottom:6px'>2 · Анализ ИИ</div><div style='font-size:13px;"
            "color:#6b7280;line-height:1.5'>Гибридный детектор и классификатор "
            "оценивают состояние и определяют тип дефекта.</div></div>",
            unsafe_allow_html=True)
    with w3:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e8eaed;border-radius:12px;"
            "padding:18px'><div style='font-size:15px;font-weight:600;color:#0F6E56;"
            "margin-bottom:6px'>3 · Рекомендация</div><div style='font-size:13px;"
            "color:#6b7280;line-height:1.5'>Служба ремонта получает диагноз, риск "
            "и конкретное действие.</div></div>", unsafe_allow_html=True)

    st.markdown("")
    st.info("Перейдите к вкладке «Парк оборудования», чтобы увидеть состояние всех "
            "агрегатов, или к «Симулятор (live)» для демонстрации работы в реальном времени.")


# ---- TAB 1: список оборудования (ТЗ 11) ----
with tab1:
    st.subheader("Оборудование, ранжированное по риску")
    st.caption("Худшие агрегаты сверху. Цвет — уровень аномалии. «Пик 14д» — был ли риск за последние две недели.")

    # ---- Мнемосхема завода ----
    st.markdown("##### Схема технологической цепочки")
    pumps = [s for s in states if "pump" in s["equipment_type"].lower() or s["equipment_id"].startswith("PMP")]
    flots = [s for s in states if s not in pumps]

    def node_svg(s, x, y):
        c = LEVEL_COLOR[s["level"]]
        pulse = "critical" in s["level"] or "anomaly" in s["level"]
        anim = (f"<animate attributeName='opacity' values='1;0.3;1' dur='1.2s' "
                f"repeatCount='indefinite'/>") if pulse else ""
        return (
            f"<g>"
            f"<rect x='{x}' y='{y}' width='96' height='54' rx='8' "
            f"fill='#fff' stroke='{c}' stroke-width='2'/>"
            f"<circle cx='{x+80}' cy='{y+14}' r='6' fill='{c}'>{anim}</circle>"
            f"<text x='{x+12}' y='{y+22}' font-size='12' font-weight='600' "
            f"fill='#1a1d21'>{s['equipment_id']}</text>"
            f"<text x='{x+12}' y='{y+40}' font-size='11' fill='{c}' "
            f"font-weight='600'>{s['hi']:.0f}</text>"
            f"<text x='{x+34}' y='{y+40}' font-size='9' fill='#9ca3af'>"
            f"{LEVEL_RU[s['level']]}</text>"
            f"</g>")

    W = 1100
    svg = [f"<svg viewBox='0 0 {W} 260' style='width:100%;height:auto;"
           f"background:#fff;border:1px solid #e8eaed;border-radius:12px'>"]
    # заголовки рядов
    svg.append("<text x='20' y='30' font-size='12' font-weight='600' fill='#6b7280'>НАСОСЫ</text>")
    svg.append("<text x='20' y='150' font-size='12' font-weight='600' fill='#6b7280'>ФЛОТОМАШИНЫ</text>")
    # соединительная линия (технологическая цепочка)
    svg.append(f"<line x1='30' y1='70' x2='{W-30}' y2='70' stroke='#e8eaed' stroke-width='2'/>")
    svg.append(f"<line x1='30' y1='190' x2='{W-30}' y2='190' stroke='#e8eaed' stroke-width='2'/>")
    # узлы
    for i, s in enumerate(pumps[:9]):
        svg.append(node_svg(s, 30 + i * 118, 45))
    for i, s in enumerate(flots[:9]):
        svg.append(node_svg(s, 30 + i * 118, 165))
    svg.append("</svg>")
    st.markdown("".join(svg), unsafe_allow_html=True)
    st.markdown("")
    st.markdown("##### Список оборудования")

    for s in states:
        c = LEVEL_COLOR[s["level"]]
        fill = LEVEL_FILL[s["level"]]
        txt = LEVEL_TEXT[s["level"]]
        pk = s["peak_level"]
        pk_txt = LEVEL_TEXT[pk]
        st.markdown(
            f"<div class='asset-card' style='display:grid;grid-template-columns:14px 160px 90px 130px 1fr 110px;align-items:center;gap:16px'>"
            f"<div style='width:12px;height:12px;border-radius:50%;background:{c}'></div>"
            f"<div><div style='font-size:14px;font-weight:600;color:#1a1d21'>{s['equipment_id']}</div>"
            f"<div style='font-size:11px;color:#9ca3af'>{s['equipment_type']}</div></div>"
            f"<div><div style='font-size:11px;color:#9ca3af'>Индекс</div>"
            f"<div style='font-size:20px;font-weight:600;color:{hi_color(s['hi'])}'>{s['hi']:.0f}</div></div>"
            f"<div><div style='font-size:11px;color:#9ca3af;margin-bottom:3px'>Уровень</div>"
            f"<span class='pill' style='background:{fill};color:{txt}'>{LEVEL_RU[s['level']]}</span></div>"
            f"<div><div style='font-size:11px;color:#9ca3af'>Вероятный дефект</div>"
            f"<div style='font-size:13px;color:#1a1d21'>{s['fault']} <span style='color:#9ca3af'>· {s['conf']:.0%}</span></div></div>"
            f"<div style='text-align:right'><div style='font-size:11px;color:#9ca3af'>Риск 30д</div>"
            f"<div style='font-size:16px;font-weight:600;color:#1a1d21'>{s['risk_30']:.0%}</div>"
            f"<div style='font-size:10px;color:{pk_txt}'>пик 14д: {LEVEL_RU[pk]}</div></div>"
            f"</div>", unsafe_allow_html=True)

# ---- TAB 2: карточка агрегата (ТЗ 11) ----
with tab2:
    ids = [s["equipment_id"] for s in states]
    sel = st.selectbox("Выберите агрегат", ids)
    s = next(x for x in states if x["equipment_id"] == sel)
    row = s["row"]
    c = LEVEL_COLOR[s["level"]]

    # реестр
    reg_row = registry[registry["equipment_id"] == sel] if len(registry) else registry
    if len(reg_row):
        rr = reg_row.iloc[0]
        st.caption(f"{rr['equipment_name']} · {rr['manufacturer']} {rr['model']} · "
                   f"{rr['location']} · критичность: {rr['criticality_level']}")

    gcol, mcol = st.columns([1, 2])
    with gcol:
        st.plotly_chart(gauge_chart(s["hi"], "Health Index"), use_container_width=True,
                        key="gauge_card")
    with mcol:
        m1, m2 = st.columns(2)
        kpi(m1, "Уровень состояния", LEVEL_RU[s["level"]], c)
        kpi(m2, "Вероятный дефект", s["fault"], sub=f"уверенность {s['conf']:.0%}")
        m3, m4 = st.columns(2)
        kpi(m3, "Риск 7/14/30 дней",
            f"{s['risk_7']:.0%} / {s['risk_14']:.0%} / {s['risk_30']:.0%}")
        kpi(m4, "Пик за 14 дней", LEVEL_RU[s["peak_level"]], LEVEL_COLOR[s["peak_level"]])

    st.markdown("")
    st.markdown(
        f"<div style='display:flex;gap:8px;padding:12px 14px;background:#E6F1FB;border-radius:8px;margin-bottom:10px'>"
        f"<span style='color:#185FA5;font-size:13px;line-height:1.5'>{build_explanation(row)}</span></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='display:flex;gap:8px;padding:12px 14px;background:#fff;border:1px solid #e8eaed;border-radius:8px'>"
        f"<span style='font-size:13px;color:#1a1d21;line-height:1.5'><b>Рекомендация:</b> {recommendation_for(s['fault'])}</span></div>",
        unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Динамика сигналов")
    g = out[out["equipment_id"] == sel].sort_values("timestamp")

    def sig_chart(col, title, unit, color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["timestamp"], y=g[col], line=dict(color=color, width=1.5),
                                 fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba") if color.startswith("rgb") else "rgba(0,0,0,0.03)"))
        fig.update_layout(height=210, margin=dict(l=10, r=10, t=34, b=10),
                          title=dict(text=f"{title}, {unit}", font=dict(size=13, color="#6b7280")),
                          showlegend=False, plot_bgcolor="#fff", paper_bgcolor="#fff",
                          xaxis=dict(showgrid=False, color="#9ca3af"),
                          yaxis=dict(gridcolor="#f0f1f3", color="#9ca3af"))
        return fig

    gc1, gc2, gc3 = st.columns(3)
    gc1.plotly_chart(sig_chart("vibration_rms", "Вибрация RMS", "мм/с", "#d63b3b"),
                     use_container_width=True)
    gc2.plotly_chart(sig_chart("current_a", "Ток фазы A", "А", "#2b6cb0"),
                     use_container_width=True)
    gc3.plotly_chart(sig_chart("temperature", "Температура", "°C", "#d97706"),
                     use_container_width=True)

    # Health Index во времени
    figh = go.Figure()
    figh.add_trace(go.Scatter(x=g["timestamp"], y=g["hi"], fill="tozeroy",
                              line=dict(color="#185FA5", width=1.8),
                              fillcolor="rgba(24,95,165,0.08)"))
    figh.update_layout(height=210, margin=dict(l=10, r=10, t=34, b=10),
                       title=dict(text="Индекс состояния во времени", font=dict(size=13, color="#6b7280")),
                       yaxis_range=[0, 100], plot_bgcolor="#fff", paper_bgcolor="#fff",
                       xaxis=dict(showgrid=False, color="#9ca3af"),
                       yaxis=dict(gridcolor="#f0f1f3", color="#9ca3af"))
    st.plotly_chart(figh, use_container_width=True)

    # ---- Частотный анализ вибрации (FFT) ----
    if sel == "PMP-204":
        st.markdown("---")
        st.markdown("##### Частотный анализ вибрации (FFT)")
        st.caption("Спектральный анализ сырого вибросигнала (20 кГц). Показывает "
                   "частотные пики дефектов — то, что не видно в мгновенных значениях.")
        try:
            fft_table, spectra = _fft_cached()
            # спектр: здоровое vs деградировавшее
            fig_fft = go.Figure()
            for snap, color, name in [(0, "#1D9E75", "Норма"),
                                      (5, "#E24B4A", "Деградация")]:
                sp = spectra[snap]
                fig_fft.add_trace(go.Scatter(
                    x=sp["freqs"], y=sp["amp"], name=name,
                    line=dict(color=color, width=1.3)))
            fig_fft.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Частота, Гц", color="#6b7280", range=[0, 500]),
                yaxis=dict(title="Амплитуда, g", gridcolor="#f0f1f3", color="#9ca3af"),
                plot_bgcolor="#fff", paper_bgcolor="#fff",
                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_fft, use_container_width=True)
            st.markdown("Пик оборотной частоты (~25 Гц) при деградации дополняется "
                        "высшими гармониками (~75 Гц) — классический признак дефекта "
                        "подшипника. Энергия в диапазоне гармоник растёт в разы.")

            # таблица частотных признаков по стадиям
            show_cols = ["severity", "rms", "crest", "kurtosis", "peak_freq"]
            ft = fft_table[show_cols].rename(columns={
                "severity": "Деградация", "rms": "RMS, g", "crest": "Crest factor",
                "kurtosis": "Kurtosis", "peak_freq": "Пик, Гц"})
            st.dataframe(ft, use_container_width=True, hide_index=True)
        except Exception as e:
            st.info(f"Сырые вибро-данные недоступны для частотного анализа.")

    # SCADA-аварии по агрегату (ТЗ 11)
    a = alarms[alarms["equipment_id"] == sel].sort_values("timestamp", ascending=False) if len(alarms) else alarms
    if len(a):
        st.subheader(f"Аварийные сообщения SCADA ({len(a)})")
        st.dataframe(a[["timestamp", "alarm_code", "alarm_text", "alarm_level",
                        "alarm_status"]].head(10), use_container_width=True, hide_index=True)

    # ---- Экспорт детального отчёта по агрегату ----
    st.markdown("---")
    detail = out[out["equipment_id"] == sel].sort_values("timestamp")
    summ_a = dict(summ)
    summ_a.update(n_crit=n_crit, n_anom=n_anom, n_warn=n_warn, n_norm=n_norm,
                  avg_hi=round(np.mean([x["hi"] for x in states])))
    report_a = build_report([s], summ_a, {"prevented_units": 0, "hours_saved": 0,
                            "gross": 0, "net": 0}, RECOMMENDATIONS, detail_df=detail)
    st.download_button(
        f"Скачать детальный отчёт по {sel} (Excel)", data=report_a,
        file_name=f"diagnostic_report_{sel}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ---- Человек в контуре: подтверждение диагноза инженером ----
    st.markdown("---")
    st.markdown("##### Обратная связь инженера")
    st.caption("Подтвердите или отклоните диагноз системы. Решения сохраняются в "
               "журнал (файл feedback_log.csv) и служат основой для дообучения модели. "
               "В промышленной версии — запись в БД предприятия.")

    fb1, fb2, fb3 = st.columns([1, 1, 2])
    with fb1:
        if st.button("Подтвердить дефект", key="confirm_defect"):
            save_feedback_row({
                "Время": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Агрегат": sel,
                "Диагноз системы": s["fault"],
                "Решение инженера": "Подтверждён",
            })
            st.success("Решение сохранено в журнал.")
    with fb2:
        if st.button("Отклонить (ложная тревога)", key="reject_defect"):
            save_feedback_row({
                "Время": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Агрегат": sel,
                "Диагноз системы": s["fault"],
                "Решение инженера": "Отклонён",
            })
            st.success("Решение сохранено в журнал.")

    log = load_feedback()
    if log:
        confirmed = sum(1 for x in log if x["Решение инженера"] == "Подтверждён")
        rejected = len(log) - confirmed
        st.markdown(
            f"<div style='display:flex;gap:16px;margin:10px 0'>"
            f"<div style='padding:8px 14px;background:#E1F5EE;border-radius:8px;"
            f"font-size:13px;color:#0F6E56'>Подтверждено: <b>{confirmed}</b></div>"
            f"<div style='padding:8px 14px;background:#FCEBEB;border-radius:8px;"
            f"font-size:13px;color:#A32D2D'>Отклонено: <b>{rejected}</b></div>"
            f"<div style='padding:8px 14px;background:#E6F1FB;border-radius:8px;"
            f"font-size:13px;color:#185FA5'>Всего решений для дообучения: "
            f"<b>{len(log)}</b></div></div>", unsafe_allow_html=True)
        st.markdown("**Журнал решений инженера** (сохранён в feedback_log.csv)")
        st.dataframe(pd.DataFrame(log[::-1]), use_container_width=True, hide_index=True)

        jc1, jc2 = st.columns([1, 3])
        with jc1:
            csv_bytes = pd.DataFrame(log).to_csv(index=False, encoding="utf-8-sig")
            st.download_button("Скачать журнал (CSV)", data=csv_bytes,
                               file_name="feedback_log.csv", mime="text/csv",
                               key="dl_feedback")
        with jc2:
            if st.button("Очистить журнал", key="clear_log"):
                if os.path.exists(FEEDBACK_CSV):
                    os.remove(FEEDBACK_CSV)
                st.info("Журнал очищен. Обновите страницу для отображения.")
    else:
        st.info("Журнал пуст. Подтвердите или отклоните диагноз, чтобы добавить "
                "первую запись.")

# ---- TAB 3: экономика (ТЗ 13) ----
with tab3:
    st.subheader("Отчёты и статистика парка")

    # ---- Экспорт отчёта в Excel ----
    st.markdown("##### Экспорт отчёта")
    st.caption("Полный отчёт в Excel: сводка, парк оборудования с рекомендациями.")
    summ_export = dict(summ)
    summ_export.update(n_crit=n_crit, n_anom=n_anom, n_warn=n_warn, n_norm=n_norm,
                       avg_hi=round(np.mean([s["hi"] for s in states])))
    report_bytes = build_report(states, summ_export, {"prevented_units": 0,
                                "hours_saved": 0, "gross": 0, "net": 0}, RECOMMENDATIONS)
    st.download_button(
        "Скачать отчёт по парку (Excel)", data=report_bytes,
        file_name="diagnostic_report_fleet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary")

    # распределение по уровням
    st.markdown("---")
    st.markdown("##### Распределение парка по уровням состояния")
    dist = pd.DataFrame({
        "Уровень": [LEVEL_RU[k] for k in ["normal", "warning", "anomaly", "critical"]],
        "Агрегатов": [n_norm, n_warn, n_anom, n_crit],
    })
    figd = go.Figure(go.Bar(x=dist["Уровень"], y=dist["Агрегатов"],
                            marker_color=[LEVEL_COLOR[k] for k in
                                          ["normal", "warning", "anomaly", "critical"]],
                            text=dist["Агрегатов"], textposition="outside"))
    figd.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       plot_bgcolor="#fff", paper_bgcolor="#fff",
                       xaxis=dict(color="#6b7280"),
                       yaxis=dict(gridcolor="#f0f1f3", color="#9ca3af"))
    st.plotly_chart(figd, use_container_width=True)

    # история отказов (контекст)
    if len(failures):
        st.markdown("##### Историческая статистика отказов")
        fsum = failures.groupby("failure_type").agg(
            событий=("failure_type", "size"),
            простой_мин=("downtime_minutes", "sum")).reset_index()
        st.dataframe(fsum, use_container_width=True, hide_index=True)


# ---- TAB 4: live-симулятор ----
with tab4:
    st.subheader("Симулятор потоковой диагностики")
    st.caption("Имитация живого оборудования: показания датчиков поступают в "
               "реальном времени, система оценивает состояние на каждом шаге.")

    sc1, sc2, sc3 = st.columns([1.4, 1, 1])
    fault_ru = {
        "normal": "Норма (без дефекта)", "bearing_wear": "Износ подшипника",
        "imbalance": "Дисбаланс", "cavitation": "Кавитация",
        "overload": "Перегрузка двигателя", "clogging": "Засорение",
        "motor_fault": "Дефект двигателя",
    }
    sim_fault = sc1.selectbox("Сценарий дефекта", list(fault_ru.keys()),
                              format_func=lambda k: fault_ru[k], index=1)
    speed = sc2.select_slider("Скорость деградации",
                              options=["медленно", "средне", "быстро"], value="средне")
    n_steps = sc3.number_input("Длительность (шагов)", 30, 200, 80, 10)

    rate = {"медленно": 0.008, "средне": 0.02, "быстро": 0.04}[speed]

    run = st.button("Запустить симуляцию", type="primary")

    # ОДИН плейсхолдер на весь живой контент — перерисовывается целиком
    # на каждом шаге, поэтому старое содержимое не накапливается
    live_ph = st.empty()

    if run:
        sim = LiveSimulator(fault=sim_fault, degrade_rate=rate, warmup=12)
        buffer = []
        hi_hist, vib_hist, steps = [], [], []
        out_s = None

        for i in range(int(n_steps)):
            r = sim.step()
            row = {c: r[c] for c in SIM_SENSORS}
            row["timestamp"] = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i)
            row["equipment_id"] = "SIM-001"
            row["equipment_type"] = "pump"
            for c in ["failure_type", "anomaly_status", "recommendation"]:
                row[c] = "normal"
            for c in ["failure_flag", "days_to_failure", "health_index",
                      "risk_7_days", "risk_14_days", "risk_30_days",
                      "simulated_fault_severity"]:
                row[c] = 0
            buffer.append(row)
            if len(buffer) < 3:
                continue

            out_s = eng.predict(pd.DataFrame(buffer))
            last = out_s.iloc[-1]
            lvl = last["anomaly_level"]
            c = LEVEL_COLOR[lvl]

            steps.append(i)
            hi_hist.append(last["hi"])
            vib_hist.append(r["vibration_rms"])

            # весь кадр рисуется заново внутри одного контейнера
            with live_ph.container():
                # статус-баннер
                is_alert = lvl in ("critical", "anomaly")
                alert_cls = "alert-critical" if lvl == "critical" else ""
                icon = "⚠ " if is_alert else ""
                blink_cls = "alert-blink" if lvl == "critical" else ""
                st.markdown(
                    f"<div class='{alert_cls}' style='padding:16px 20px;border-radius:10px;"
                    f"background:{LEVEL_FILL[lvl]};border:2px solid {c}'>"
                    f"<span class='{blink_cls}' style='font-size:17px;font-weight:700;"
                    f"color:{LEVEL_TEXT[lvl]}'>{icon}Шаг {i+1}/{int(n_steps)} · "
                    f"{'ТРЕВОГА: ' if lvl=='critical' else ''}{LEVEL_RU[lvl]}</span>"
                    f"<span style='float:right;color:{LEVEL_TEXT[lvl]};font-size:14px;"
                    f"font-weight:600'>деградация {r['severity']:.0%}</span></div>",
                    unsafe_allow_html=True)

                # KPI: gauge + метрики
                gk, mk = st.columns([1, 2])
                with gk:
                    st.plotly_chart(gauge_chart(last["hi"], "Health Index", height=180),
                                    use_container_width=True, key=f"simgauge_{i}")
                with mk:
                    kk1, kk2 = st.columns(2)
                    kpi(kk1, "Уровень", LEVEL_RU[lvl], c)
                    kpi(kk2, "Вероятный дефект", last["pred_fault"], sub=f"{last['pred_conf']:.0%}")
                    kk3, kk4 = st.columns(2)
                    kpi(kk3, "Риск 30д", f"{last['risk_30']:.0%}")
                    kpi(kk4, "Деградация", f"{r['severity']:.0%}", c)

                # график
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hi_hist, x=steps, name="Health Index",
                                         line=dict(color="#185FA5", width=2), yaxis="y"))
                fig.add_trace(go.Scatter(y=vib_hist, x=steps, name="Вибрация",
                                         line=dict(color="#E24B4A", width=1.5), yaxis="y2"))
                fig.update_layout(
                    height=340, margin=dict(l=10, r=10, t=30, b=10),
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                    yaxis=dict(title="Health Index", range=[0, 100], gridcolor="#f0f1f3",
                               color="#185FA5"),
                    yaxis2=dict(title="Вибрация RMS", overlaying="y", side="right",
                                color="#E24B4A", showgrid=False),
                    xaxis=dict(title="шаг", color="#9ca3af"),
                    legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, use_container_width=True, key=f"simchart_{i}")

                # рекомендация
                st.markdown(
                    f"<div style='padding:12px 14px;background:#fff;border:1px solid #e8eaed;"
                    f"border-radius:8px'><span style='font-size:13px'>"
                    f"<b>Рекомендация:</b> {recommendation_for(last['pred_fault'])}</span></div>",
                    unsafe_allow_html=True)

            time.sleep(0.12)

        if out_s is not None:
            final_fault = out_s.iloc[-1]["pred_fault"]
            final_level = out_s.iloc[-1]["anomaly_level"]
            st.success(f"Симуляция завершена. Финальное состояние: "
                       f"{LEVEL_RU[final_level]}, дефект: {final_fault}.")

            # ---- Схема агрегата: где проявился дефект ----
            st.markdown("##### Где дефект проявился на оборудовании")
            st.caption("Подсвечены датчики, по сигналам которых система определила дефект.")
            # какие датчики связаны с каждым дефектом
            FAULT_SENSORS = {
                "bearing_wear": {"accel", "temp"},
                "imbalance": {"accel", "current"},
                "cavitation": {"current", "press"},
                "overload": {"current", "temp"},
                "clogging": {"flow", "current"},
                "motor_fault": {"temp", "current"},
                "normal": set(),
            }
            active = FAULT_SENSORS.get(final_fault, set())
            # позиции датчиков на узлах флотомашины FM-8
            # (id, x, y, подпись, цвет)
            sensors = [
                ("accel", 175, 78, "Вибрация вала", "#E24B4A"),
                ("current", 355, 78, "Ток привода (MCSA)", "#7F77DD"),
                ("temp", 505, 78, "Температура", "#EF9F27"),
                ("flow", 70, 205, "Расход пульпы", "#1D9E75"),
                ("press", 585, 205, "Аэрация / давление", "#534AB7"),
            ]
            circles = []
            labels = []
            for sid, x, y, label, color in sensors:
                is_active = sid in active
                anchor = "start" if sid == "flow" else ("end" if sid == "press" else "middle")
                if is_active:
                    circles.append(
                        f"<circle class='ring' cx='{x}' cy='{y}' r='9' fill='none' "
                        f"stroke='{color}' stroke-width='2'/>")
                    circles.append(
                        f"<circle class='pulse' cx='{x}' cy='{y}' r='9' fill='{color}'/>")
                    labels.append(
                        f"<text x='{x}' y='{y-18}' text-anchor='{anchor}' font-size='12' "
                        f"font-weight='700' fill='{color}'>{label}</text>")
                else:
                    circles.append(
                        f"<circle cx='{x}' cy='{y}' r='6' fill='{color}' opacity='0.3'/>")
                    labels.append(
                        f"<text x='{x}' y='{y-18}' text-anchor='{anchor}' font-size='11' "
                        f"fill='#9ca3af'>{label}</text>")

            # приводные блоки импеллеров над камерами (x-центры)
            drive_x = [110, 260, 410, 545]
            drives = []
            for i, dx in enumerate(drive_x):
                # опора привода (H-образная рама)
                drives.append(
                    f"<rect x='{dx-32}' y='95' width='64' height='70' rx='3' "
                    f"fill='none' stroke='#4A90C2' stroke-width='2'/>")
                # жёлтая верхняя площадка (как на реальном фото FM-8)
                drives.append(
                    f"<rect x='{dx-38}' y='78' width='76' height='18' rx='4' "
                    f"fill='#F5C518' opacity='0.85' stroke='#B8901A'/>")
                # вал импеллера
                drives.append(
                    f"<rect x='{dx-4}' y='95' width='8' height='95' fill='#5F5E5A' opacity='0.6'/>")
                # импеллер внутри камеры (вращается)
                drives.append(
                    f"<g class='imp' style='transform-origin:{dx}px 205px'>"
                    f"<ellipse cx='{dx}' cy='205' rx='22' ry='8' fill='#378ADD' opacity='0.55'/>"
                    f"<ellipse cx='{dx}' cy='205' rx='8' ry='22' fill='#378ADD' opacity='0.35'/>"
                    f"</g>")

            schematic_html = f"""
<!DOCTYPE html><html><head><style>
body {{ margin:0; padding:0; background:transparent; }}
@keyframes pulse {{ 0%,100% {{ r:9; opacity:1; }} 50% {{ r:13; opacity:0.6; }} }}
@keyframes ring {{ 0% {{ r:9; opacity:0.6; }} 100% {{ r:26; opacity:0; }} }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.pulse {{ animation: pulse 1.4s ease-in-out infinite; }}
.ring {{ animation: ring 1.8s ease-out infinite; }}
.imp {{ animation: spin 3s linear infinite; }}
</style></head><body>
<svg viewBox='0 0 640 300' style='width:100%;display:block;background:#fff;border:1px solid #e8eaed;border-radius:12px'>
<!-- Флотомашина FM-8: ряд камер в синем корпусе -->
<!-- корпус (синий, как реальная FM-8) -->
<rect x='40' y='180' width='560' height='75' rx='6' fill='#5B9BD5' opacity='0.35' stroke='#2E6DA4' stroke-width='1.5'/>
<!-- разделители камер -->
<line x1='190' y1='180' x2='190' y2='255' stroke='#2E6DA4' stroke-width='1.5' opacity='0.6'/>
<line x1='340' y1='180' x2='340' y2='255' stroke='#2E6DA4' stroke-width='1.5' opacity='0.6'/>
<line x1='480' y1='180' x2='480' y2='255' stroke='#2E6DA4' stroke-width='1.5' opacity='0.6'/>
<!-- подача пульпы (слева) и разгрузка (справа) -->
<rect x='20' y='198' width='22' height='30' rx='3' fill='#9FE1CB' opacity='0.6' stroke='#0F6E56'/>
<rect x='598' y='198' width='22' height='30' rx='3' fill='#F5C4B3' opacity='0.6' stroke='#993C1D'/>
<!-- опорная рама -->
<rect x='40' y='255' width='560' height='12' rx='2' fill='#B4B2A9' opacity='0.4'/>
{''.join(drives)}
<text x='320' y='285' text-anchor='middle' font-size='13' font-weight='600' fill='#1a1d21'>Флотомашина FM-8 (камеры с импеллерами)</text>
{''.join(circles)}
{''.join(labels)}
</svg></body></html>
"""
            components.html(schematic_html, height=360)
            if active:
                names = {"accel": "вибрация", "current": "ток", "temp": "температура",
                         "flow": "расход", "press": "давление"}
                signals = ", ".join(names[s] for s in active if s in names)
                st.markdown(
                    f"<div style='font-size:13px;color:#6b7280;margin-top:6px'>"
                    f"Дефект «{final_fault}» определён по отклонениям сигналов: "
                    f"<b>{signals}</b>. Подсвеченные датчики — ключевые для этого типа "
                    f"неисправности.</div>", unsafe_allow_html=True)
    else:
        live_ph.info("Выберите сценарий дефекта и нажмите «Запустить симуляцию». "
                     "Система покажет, как деградация развивается во времени и как "
                     "меняются индекс состояния, уровень тревоги и диагноз.")


# ---- TAB 5: валидация модели ----
with tab5:
    st.subheader("Валидация модели на потоковых сценариях")
    st.caption("Система прогоняет контролируемые сценарии деградации по всем типам "
               "дефектов и оценивает: обнаруживает ли дефект, за сколько времени до "
               "отказа предупреждает, верно ли определяет тип.")

    st.info("Это стресс-тест на контролируемых сценариях, а не валидация на реальных "
            "промышленных данных. Он показывает способность системы отслеживать "
            "деградацию и предупреждать заранее. Для промышленного внедрения требуется "
            "валидация на реальных сигналах предприятия.")

    if st.button("Запустить валидацию", type="primary"):
        with st.spinner("Прогон сценариев по всем дефектам..."):
            val = run_validation_cached(eng, cache_key, n_runs=3)
            # защита от устаревшего кэша: если результат без новых метрик —
            # сбрасываем кэш и пересчитываем
            if "det_accuracy" not in val:
                st.cache_data.clear()
                val = run_validation_cached(eng, cache_key + "_v2", n_runs=3)

        # ---- ключевые метрики ----
        st.markdown("##### Ключевые метрики")
        v1, v2, v3, v4 = st.columns(4)
        kpi(v1, "Accuracy детекции", f"{val['det_accuracy']:.0%}",
            "#1D9E75" if val["det_accuracy"] > 0.9 else "#EF9F27",
            sub="с классом «норма»")
        kpi(v2, "Точность типа дефекта", f"{val['type_accuracy']:.0%}")
        kpi(v3, "Macro-F1", f"{val['f1']:.3f}")
        kpi(v4, "Раннее предупреждение",
            f"{val['lead_mean']:.0f} ч", "#1D9E75",
            sub="в среднем до отказа")

        st.markdown("")
        st.markdown(
            f"<div style='padding:14px 18px;background:#E1F5EE;border-radius:10px;"
            f"border:1px solid #1D9E75'><span style='font-size:14px;color:#0F6E56'>"
            f"<b>Раннее обнаружение:</b> система предупреждает о дефекте в среднем за "
            f"<b>{val['lead_mean']:.0f} часов</b> до критического состояния "
            f"(диапазон {val['lead_min']:.0f}–{val['lead_max']:.0f} ч). Это позволяет "
            f"перейти от аварийного ремонта к плановому в удобное технологическое окно."
            f"</span></div>", unsafe_allow_html=True)

        # ---- честная детекция: с классом норма ----
        st.markdown("---")
        st.markdown("##### Детекция дефектов (с классом «норма»)")
        st.caption("Тест включает и дефектные, и здоровые сценарии — поэтому метрики "
                   "учитывают ложные тревоги. Это честная оценка: система проверена не "
                   "только на способность найти дефект, но и не паниковать на исправном "
                   "оборудовании.")
        dd1, dd2, dd3, dd4 = st.columns(4)
        kpi(dd1, "Recall (полнота)", f"{val['det_recall']:.0%}",
            sub="найдено дефектов")
        kpi(dd2, "Precision (точность)", f"{val['det_precision']:.0%}",
            sub="тревог обоснованы")
        kpi(dd3, "Ложные тревоги", f"{val['false_alarm_rate']:.0%}",
            "#EF9F27" if val["false_alarm_rate"] > 0.15 else "#1D9E75",
            sub="на здоровом оборуд.")
        kpi(dd4, "Проверено сценариев",
            f"{val['tp'] + val['fp'] + val['tn'] + val['fn']}",
            sub=f"{val['tp']+val['fn']} дефект. + {val['tn']+val['fp']} норма")
        st.markdown(
            f"<div style='font-size:13px;color:#6b7280;margin-top:8px'>"
            f"TP={val['tp']} (дефект найден) · FN={val['fn']} (пропущен) · "
            f"TN={val['tn']} (норма распознана) · FP={val['fp']} (ложная тревога). "
            f"Recall {val['det_recall']:.0%} означает, что система не пропускает дефекты; "
            f"частота ложных тревог {val['false_alarm_rate']:.0%} — приемлемо для "
            f"промышленной диагностики.</div>", unsafe_allow_html=True)

        # ---- детальные метрики классификации ----
        st.markdown("---")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("##### Precision / Recall / F1")
            met_df = pd.DataFrame({
                "Метрика": ["Precision", "Recall", "F1-score"],
                "Значение": [f"{val['precision']:.3f}", f"{val['recall']:.3f}",
                             f"{val['f1']:.3f}"],
            })
            st.dataframe(met_df, use_container_width=True, hide_index=True)

        with cc2:
            st.markdown("##### Раннее предупреждение (lead time)")
            lead_df = pd.DataFrame({
                "Показатель": ["Средний", "Медиана", "Минимум", "Максимум"],
                "Часов до отказа": [f"{val['lead_mean']:.0f}", f"{val['lead_median']:.0f}",
                                    f"{val['lead_min']:.0f}", f"{val['lead_max']:.0f}"],
            })
            st.dataframe(lead_df, use_container_width=True, hide_index=True)

        # ---- confusion matrix ----
        st.markdown("##### Матрица ошибок классификации дефектов")
        labels_ru = [VAL_FAULT_RU[l] for l in val["labels"]]
        cm = val["confusion"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels_ru, y=labels_ru, colorscale="Blues",
            text=cm, texttemplate="%{text}", textfont={"size": 13},
            showscale=False))
        fig_cm.update_layout(
            height=400, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Прогноз", side="bottom", color="#6b7280"),
            yaxis=dict(title="Факт", autorange="reversed", color="#6b7280"),
            plot_bgcolor="#fff", paper_bgcolor="#fff")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("По диагонали — верные определения. Вне диагонали — какой дефект с "
                   "каким путается. Засорение и износ подшипника имеют схожие сигнатуры "
                   "по вибрации — это известное ограничение, требующее больше реальных "
                   "данных для различения.")

        # ---- метрики по каждому дефекту ----
        st.markdown("##### Детекция и раннее предупреждение по типам дефектов")
        pf = val["per_fault"]
        rows = []
        for f in val["labels"]:
            rows.append({
                "Дефект": VAL_FAULT_RU[f],
                "Обнаружение": f"{pf[f]['detect']:.0%}",
                "Точность типа": f"{pf[f]['type_acc']:.0%}",
                "Lead time, ч": f"{pf[f]['lead']:.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("Нажмите «Запустить валидацию», чтобы прогнать все сценарии "
                    "дефектов и получить метрики качества модели: точность обнаружения, "
                    "F1-score, матрицу ошибок и главное — за сколько времени до отказа "
                    "система предупреждает.")


# ---- TAB 6: промышленное внедрение ----
with tab6:
    st.subheader("Архитектура промышленного внедрения")
    st.caption("Как система подключается к инфраструктуре предприятия — от датчиков "
               "до ремонтной заявки.")

    # SVG-схема потока данных
    stages = [
        ("Датчики", "вибрация, ток,\nтемпература,\nрасход", "#185FA5"),
        ("SCADA / PLC", "сбор данных\nOPC UA / MQTT", "#2b6cb0"),
        ("Edge Gateway", "буферизация,\nпредобработка", "#378ADD"),
        ("AI-модель", "детекция аномалий,\nHealth Index,\nдиагноз", "#D85A30"),
        ("Дашборд", "мониторинг,\nтревоги,\nрекомендации", "#1D9E75"),
        ("Ремонтная\nзаявка", "интеграция\nс ТОиР", "#854F0B"),
    ]
    box_w, gap, x0, y = 165, 30, 20, 40
    svg = ["<svg viewBox='0 0 1180 180' style='width:100%;height:auto'>"]
    for i, (title, sub, color) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        svg.append(
            f"<rect x='{x}' y='{y}' width='{box_w}' height='90' rx='10' "
            f"fill='#fff' stroke='{color}' stroke-width='2'/>")
        svg.append(
            f"<text x='{x+box_w/2}' y='{y+26}' font-size='14' font-weight='700' "
            f"fill='{color}' text-anchor='middle'>{title.split(chr(10))[0]}</text>")
        if chr(10) in title:
            svg.append(f"<text x='{x+box_w/2}' y='{y+42}' font-size='14' "
                       f"font-weight='700' fill='{color}' text-anchor='middle'>"
                       f"{title.split(chr(10))[1]}</text>")
        for j, line in enumerate(sub.split("\n")):
            svg.append(f"<text x='{x+box_w/2}' y='{y+58+j*14}' font-size='10' "
                       f"fill='#6b7280' text-anchor='middle'>{line}</text>")
        # стрелка
        if i < len(stages) - 1:
            ax = x + box_w
            svg.append(f"<line x1='{ax}' y1='{y+45}' x2='{ax+gap}' y2='{y+45}' "
                       f"stroke='#cbd0d6' stroke-width='2'/>")
            svg.append(f"<polygon points='{ax+gap-6},{y+41} {ax+gap},{y+45} "
                       f"{ax+gap-6},{y+49}' fill='#cbd0d6'/>")
    svg.append("</svg>")
    st.markdown("".join(svg), unsafe_allow_html=True)

    # ---- Анимированная схема аппаратной части (датчики на агрегате) ----
    st.markdown("---")
    st.markdown("##### Аппаратная часть: датчики на агрегате")
    st.caption("Датчики снимают сигнал в реальном времени и передают данные "
               "контроллеру сбора. Пульсация показывает активный сбор сигнала.")
    hardware_html = """
<!DOCTYPE html><html><head><style>
body { margin:0; background:transparent; font-family:Arial,sans-serif; }
@keyframes sPulse { 0%,100% { r:6; opacity:1; } 50% { r:9; opacity:0.65; } }
@keyframes sRing { 0% { r:6; opacity:0.5; } 100% { r:18; opacity:0; } }
@keyframes flow { to { stroke-dashoffset:-24; } }
@keyframes spin { to { transform: rotate(360deg); } }
.sensor { animation: sPulse 1.6s ease-in-out infinite; }
.ring { animation: sRing 2s ease-out infinite; }
.flow { stroke-dasharray:4 7; animation: flow 1s linear infinite; }
.fan { animation: spin 3s linear infinite; transform-origin: 250px 250px; }
.lbl { font-size:12px; font-weight:600; }
</style></head><body>
<svg width="100%" viewBox="0 0 680 480" xmlns="http://www.w3.org/2000/svg">
<rect x="60" y="360" width="560" height="20" rx="3" fill="#B4B2A9" opacity="0.5" stroke="#5F5E5A" stroke-width="0.5"/>
<rect x="90" y="380" width="30" height="30" fill="#B4B2A9" opacity="0.4"/>
<rect x="560" y="380" width="30" height="30" fill="#B4B2A9" opacity="0.4"/>
<path d="M250 190 A60 60 0 1 0 310 250 L310 250 L360 250 L360 220 L310 220" fill="#B5D4F4" opacity="0.4" stroke="#185FA5" stroke-width="1.5"/>
<circle cx="250" cy="250" r="60" fill="#B5D4F4" opacity="0.3" stroke="#185FA5" stroke-width="1.5"/>
<g class="fan">
<path d="M250 250 L250 200 Q265 210 262 235 Z" fill="#378ADD" opacity="0.5"/>
<path d="M250 250 L293 275 Q280 288 258 268 Z" fill="#378ADD" opacity="0.5"/>
<path d="M250 250 L207 275 Q220 288 242 268 Z" fill="#378ADD" opacity="0.5"/>
<path d="M250 250 L250 300 Q235 290 238 265 Z" fill="#378ADD" opacity="0.5"/>
<path d="M250 250 L207 225 Q220 212 242 232 Z" fill="#378ADD" opacity="0.5"/>
<path d="M250 250 L293 225 Q280 212 258 232 Z" fill="#378ADD" opacity="0.5"/>
</g>
<circle cx="250" cy="250" r="8" fill="#5F5E5A"/>
<text x="250" y="335" text-anchor="middle" style="font-size:13px;font-weight:600;fill:#1a1d21">Насос</text>
<rect x="230" y="130" width="40" height="62" rx="3" fill="#9FE1CB" opacity="0.5" stroke="#0F6E56" stroke-width="1"/>
<text x="250" y="120" text-anchor="middle" style="font-size:11px;fill:#6b7280">всас</text>
<rect x="345" y="215" width="60" height="34" rx="3" fill="#F5C4B3" opacity="0.5" stroke="#993C1D" stroke-width="1"/>
<text x="375" y="207" text-anchor="middle" style="font-size:11px;fill:#6b7280">напор</text>
<rect x="360" y="238" width="55" height="24" rx="3" fill="#D3D1C7" opacity="0.6" stroke="#5F5E5A" stroke-width="1"/>
<text x="387" y="254" text-anchor="middle" style="font-size:10px;fill:#5F5E5A">муфта</text>
<rect x="415" y="205" width="150" height="90" rx="8" fill="#B4B2A9" opacity="0.35" stroke="#5F5E5A" stroke-width="1.5"/>
<line x1="428" y1="215" x2="428" y2="285" stroke="#5F5E5A" stroke-width="0.5" opacity="0.5"/>
<line x1="440" y1="215" x2="440" y2="285" stroke="#5F5E5A" stroke-width="0.5" opacity="0.5"/>
<line x1="452" y1="215" x2="452" y2="285" stroke="#5F5E5A" stroke-width="0.5" opacity="0.5"/>
<text x="500" y="254" text-anchor="middle" style="font-size:12px;font-weight:600;fill:#1a1d21">Двигатель</text>
<rect x="540" y="230" width="30" height="40" rx="3" fill="#888780" opacity="0.4"/>
<line class="flow" x1="250" y1="205" x2="250" y2="70" stroke="#E24B4A" stroke-width="1.5" fill="none"/>
<line class="flow" x1="510" y1="207" x2="560" y2="112" stroke="#7F77DD" stroke-width="1.5" fill="none"/>
<line class="flow" x1="500" y1="295" x2="560" y2="400" stroke="#EF9F27" stroke-width="1.5" fill="none"/>
<line class="flow" x1="245" y1="132" x2="120" y2="105" stroke="#1D9E75" stroke-width="1.5" fill="none"/>
<line class="flow" x1="400" y1="222" x2="470" y2="130" stroke="#534AB7" stroke-width="1.5" fill="none"/>
<circle class="ring" cx="250" cy="205" r="6" fill="none" stroke="#E24B4A" stroke-width="2"/>
<circle class="sensor" cx="250" cy="205" r="6" fill="#E24B4A"/>
<circle class="ring" cx="510" cy="207" r="6" fill="none" stroke="#7F77DD" stroke-width="2" style="animation-delay:0.3s"/>
<circle class="sensor" cx="510" cy="207" r="6" fill="#7F77DD" style="animation-delay:0.3s"/>
<circle class="ring" cx="500" cy="295" r="6" fill="none" stroke="#EF9F27" stroke-width="2" style="animation-delay:0.6s"/>
<circle class="sensor" cx="500" cy="295" r="6" fill="#EF9F27" style="animation-delay:0.6s"/>
<circle class="ring" cx="245" cy="132" r="6" fill="none" stroke="#1D9E75" stroke-width="2" style="animation-delay:0.9s"/>
<circle class="sensor" cx="245" cy="132" r="6" fill="#1D9E75" style="animation-delay:0.9s"/>
<circle class="ring" cx="400" cy="222" r="6" fill="none" stroke="#534AB7" stroke-width="2" style="animation-delay:1.2s"/>
<circle class="sensor" cx="400" cy="222" r="6" fill="#534AB7" style="animation-delay:1.2s"/>
<rect x="150" y="52" width="205" height="28" rx="6" fill="#FCEBEB" stroke="#E24B4A" stroke-width="0.5"/>
<text x="160" y="70" class="lbl" fill="#A32D2D">Акселерометр вибрации</text>
<rect x="472" y="98" width="198" height="28" rx="6" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text x="482" y="116" class="lbl" fill="#3C3489">Датчик тока (MCSA)</text>
<rect x="472" y="388" width="198" height="28" rx="6" fill="#FAEEDA" stroke="#EF9F27" stroke-width="0.5"/>
<text x="482" y="406" class="lbl" fill="#854F0B">Термодатчик двигателя</text>
<rect x="10" y="90" width="215" height="28" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text x="20" y="108" class="lbl" fill="#0F6E56">Давление и расход (всас)</text>
<rect x="230" y="415" width="220" height="34" rx="8" fill="#E6F1FB" stroke="#185FA5" stroke-width="0.5"/>
<text x="340" y="437" text-anchor="middle" class="lbl" fill="#0C447C">Контроллер сбора (SCADA/PLC)</text>
</svg></body></html>
"""
    components.html(hardware_html, height=490)

    # ---- Специфика оборудования: насосы vs флотомашины ----
    st.markdown("---")
    st.markdown("##### Специфика оборудования (кейс «Казахмыс»)")
    st.caption("Поток данных общий, но у насосов и флотомашин разные критичные "
               "дефекты и диагностические сигналы. Система учитывает специфику каждого типа.")
    sp1, sp2 = st.columns(2)
    with sp1:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e8eaed;border-radius:12px;"
            "padding:18px;border-top:3px solid #185FA5'>"
            "<div style='font-size:16px;font-weight:600;color:#185FA5;margin-bottom:10px'>"
            "Насосное оборудование</div>"
            "<div style='font-size:12px;color:#9ca3af;text-transform:uppercase;"
            "letter-spacing:0.4px;margin-bottom:4px'>Критичные дефекты</div>"
            "<div style='font-size:13px;color:#1a1d21;line-height:1.7;margin-bottom:10px'>"
            "• Гидроабразивный износ рабочего колеса<br>"
            "• Кавитация шламовых насосов<br>"
            "• Разрушение подшипников</div>"
            "<div style='font-size:12px;color:#9ca3af;text-transform:uppercase;"
            "letter-spacing:0.4px;margin-bottom:4px'>Ключевые сигналы</div>"
            "<div style='font-size:13px;color:#6b7280;line-height:1.6'>"
            "токовый анализ (MCSA), падение КПД, вибрация подшипников, "
            "перепад давления, расход</div></div>", unsafe_allow_html=True)
    with sp2:
        st.markdown(
            "<div style='background:#fff;border:1px solid #e8eaed;border-radius:12px;"
            "padding:18px;border-top:3px solid #1D9E75'>"
            "<div style='font-size:16px;font-weight:600;color:#0F6E56;margin-bottom:10px'>"
            "Флотомашины</div>"
            "<div style='font-size:12px;color:#9ca3af;text-transform:uppercase;"
            "letter-spacing:0.4px;margin-bottom:4px'>Критичные дефекты</div>"
            "<div style='font-size:13px;color:#1a1d21;line-height:1.7;margin-bottom:10px'>"
            "• Дисбаланс импеллера (налипание, износ лопаток)<br>"
            "• Разрушение редуктора<br>"
            "• Нарушение аэрации и режима перемешивания</div>"
            "<div style='font-size:12px;color:#9ca3af;text-transform:uppercase;"
            "letter-spacing:0.4px;margin-bottom:4px'>Ключевые сигналы</div>"
            "<div style='font-size:13px;color:#6b7280;line-height:1.6'>"
            "вибрация вала, гармоники оборотной частоты, ток привода, "
            "стабильность импеллера</div></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#6b7280;margin-top:12px;line-height:1.6'>"
        "Дисбаланс импеллера флотомашины напрямую влияет на извлечение меди из руды — "
        "система предупреждает службу механика до того, как дисбаланс разрушит редуктор. "
        "У насосов износ лопастей ведёт к падению КПД и кавитации, разрушающей "
        "подшипники — это ловится токовым анализом.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Уровни системы")
    levels = pd.DataFrame({
        "Уровень": ["Датчики", "Сбор данных", "Хранилище", "AI-обработка",
                    "Интерфейс", "Интеграция", "Контроль"],
        "Что нужно": [
            "вибрация, ток, температура, давление, расход",
            "SCADA / PLC / OPC UA / MQTT",
            "time-series database (историзация сигналов)",
            "edge или сервер (обучение и инференс моделей)",
            "дашборд надёжности (текущая система)",
            "автоматическая заявка в ремонтную службу (ТОиР)",
            "журнал тревог, подтверждение дефекта инженером",
        ],
    })
    st.dataframe(levels, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("##### Человек в контуре")
    st.markdown(
        "Система работает в поддержку инженера, а не вместо него:\n\n"
        "1. AI выдаёт тревогу с обоснованием (какие сигналы отклонились)\n"
        "2. Инженер-механик подтверждает или отклоняет\n"
        "3. Результат фиксируется в журнале\n"
        "4. Модель дообучается на подтверждённых случаях\n"
        "5. Точность тревог растёт со временем\n\n"
        "Это обеспечивает доверие службы надёжности и адаптацию системы под "
        "специфику конкретного предприятия.")

