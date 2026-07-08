"""
diagnostics.py — рекомендации (ТЗ 12) и экономический эффект (ТЗ 13)
"""
import numpy as np

# ТЗ 12: дефект -> рекомендация ремонтной службе
RECOMMENDATIONS = {
    "normal": "Продолжить эксплуатацию. Отклонений не выявлено.",
    "bearing_wear": "Провести вибродиагностику подшипникового узла, проверить смазку и температуру. Запланировать замену подшипника в ближайшее окно ТО.",
    "imbalance": "Проверить балансировку рабочего колеса / импеллера и крепление ротора.",
    "misalignment": "Проверить центровку вала и состояние муфты.",
    "cavitation": "Проверить давление и подпор на входе, режим работы. Исключить работу вне рабочей точки.",
    "overload": "Проверить токовую нагрузку двигателя и механические заедания. Снизить нагрузку до диагностики.",
    "clogging": "Проверить проточную часть на засорение и износ крыльчатки, оценить падение КПД.",
    "motor_fault": "Проверить электродвигатель: обмотки, изоляцию, токовый профиль. Привлечь электрослужбу.",
    "seal_failure": "Проверить состояние уплотнений, наличие утечек.",
    "unknown": "Отклонение обнаружено, тип не определён уверенно. Усилить мониторинг и провести осмотр.",
}


def recommendation_for(fault):
    return RECOMMENDATIONS.get(fault, RECOMMENDATIONS["unknown"])


def build_explanation(row):
    """Объяснимость (ТЗ 10.4): почему такой вывод."""
    reasons = []
    if row.get("vibration_rms_trend", 0) > 0.5:
        reasons.append(f"рост вибрации (тренд {row['vibration_rms_trend']:+.1f} мм/с)")
    if row.get("current_a_trend", 0) > 5:
        reasons.append(f"аномалия тока ({row['current_a_trend']:+.1f} А)")
    if row.get("temperature_trend", 0) > 3:
        reasons.append(f"рост температуры ({row['temperature_trend']:+.1f} °C)")
    if row.get("flow_rate_trend", 0) < -10:
        reasons.append(f"падение расхода ({row['flow_rate_trend']:+.1f} м³/ч)")
    if row.get("current_imbalance", 0) > 0.1:
        reasons.append(f"перекос фаз тока ({row['current_imbalance']:.0%})")
    if not reasons:
        reasons.append("незначительные отклонения по нескольким сигналам")
    return "Основание: " + ", ".join(reasons) + "."


# ТЗ 13: экономический эффект
def economic_impact(n_critical, n_warning, cost_per_hour=1500,
                    avg_downtime_h=8, deployment_cost=0):
    """
    Эффект = предотвращённые часы простоя × стоимость часа
             - стоимость внедрения.
    """
    prevented_units = n_critical + n_warning
    hours_saved = prevented_units * avg_downtime_h
    gross = hours_saved * cost_per_hour
    net = gross - deployment_cost
    return dict(prevented_units=prevented_units, hours_saved=hours_saved,
                gross=gross, net=net)


if __name__ == "__main__":
    print(recommendation_for("bearing_wear"))
    print(economic_impact(3, 5))
