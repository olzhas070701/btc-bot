# btc_bot.py
# -*- coding: utf-8 -*-
"""
BTC bot (OKX) — сценарии / фильтры / уровни пользователя.

Что делает:
1) Берёт данные с OKX по инструменту BTC-USDT-SWAP (перп).
2) Пишет ВСЁ в archive/ (история) и в latest/ (актуальные файлы).
3) Умеет работать по ТВОИМ уровням из latest/btc_levels_data.csv (приоритет),
   если их нет — строит авто-уровни по H1.
4) Даёт статусы:
   - НЕТ СДЕЛКИ (нет импульса / середина диапазона / нет отказа)
   - ЖДЁМ СВИП (нет снятия ликвидности у уровня)
   - НЕДЕЙСТВИТЕЛЬНО (запрет сделки без ретеста)
   - СДЕЛКА (есть свип + отказ + ретест)
5) Пишет “КОГО НАКАЖУТ, ЕСЛИ ВОЙТИ”.
6) Генерирует текст “СЦЕНАРИЙ / ПРОГНОЗ” (не ордер!), “ГДЕ ЛИКВИДНОСТЬ”.

ВАЖНО про графики:
- matplotlib может отсутствовать в GitHub Actions.
- Поэтому визуализация включается ТОЛЬКО если matplotlib установлен.
  Если нет — бот не падает, а просто пропускает PNG.

Файлы на выходе (в latest/ и в archive/...):
- btc_levels_data.csv
- btc_setups_data.csv
- btc_attention_points_data.csv
- btc_patterns_data.csv
- btc_ohlcv_1h.csv
- btc_ohlcv_1d.csv
- btc_scenario_report.txt  (сценарий/ликвидность/вывод)
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import requests
import pandas as pd
import numpy as np
import pytz

# ------------------- Настройки -------------------
SYMBOL_FUTURES = "BTC-USDT-SWAP"
TIMEFRAMES = {"5m": "5m", "1h": "1H", "1d": "1D"}
ATR_PERIOD = 14
LIMIT = 200

ARCHIVE_ROOT = "archive"
LATEST_ROOT = "latest"

BASE_FILES = {
    "levels": "btc_levels_data",
    "patterns": "btc_patterns_data",
    "ohlcv_1h": "btc_ohlcv_1h",
    "ohlcv_1d": "btc_ohlcv_1d",
    "setups": "btc_setups_data",
    "attention": "btc_attention_points_data",
    "scenario": "btc_scenario_report",
    "viz_dir": "btc_viz",
}

# --- Параметры фильтров/логики ---
RETEST_ATR_TOL = 0.15        # допуск к уровню как доля ATR(1H)
RETEST_WINDOW_5M = 12        # сколько свечей M5 ждём ретест (12*5m=60 минут)
IMPULSE_ATR_MULT = 0.7       # импульс >= 0.7*ATR(1H)
IMPULSE_WINDOW_5M = 6        # окно импульса (6*5m=30 минут)
MID_RANGE_BAND = 0.20        # зона середины диапазона (20% от range вокруг mid)

# Визуализация (не обязательна)
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Логирование (в консоль — видно в Actions)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)


# ------------------- Утилиты сохранения -------------------
def get_archive_path(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    week_num = dt.isocalendar()[1]
    day = dt.strftime("%d")
    # archive/YYYY/MM/week_XX/DD/
    return os.path.join(ARCHIVE_ROOT, year, month, f"week_{week_num:02d}", f"{day}")


def ensure_dirs():
    os.makedirs(ARCHIVE_ROOT, exist_ok=True)
    os.makedirs(LATEST_ROOT, exist_ok=True)


def save_to_archive(df: pd.DataFrame, base_filename: str, date_str: str, header=True, also_latest=True) -> str:
    """Пишет в архив (append по дням) и обновляет latest/ (перезапись)."""
    archive_dir = get_archive_path(date_str)
    os.makedirs(archive_dir, exist_ok=True)
    day = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").strftime("%d")

    # 1) архив
    file_path = os.path.join(archive_dir, f"{day}_{base_filename}.csv")
    mode = "a" if os.path.exists(file_path) else "w"
    df.to_csv(file_path, mode=mode, header=header and mode == "w", index=False)

    # 2) latest
    if also_latest:
        latest_path = os.path.join(LATEST_ROOT, f"{base_filename}.csv")
        df.to_csv(latest_path, index=False)

    return file_path


def save_text_report(text: str, date_str: str, base_filename: str):
    """Текстовый отчёт: archive + latest."""
    archive_dir = get_archive_path(date_str)
    os.makedirs(archive_dir, exist_ok=True)
    day = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").strftime("%d")

    # archive
    p1 = os.path.join(archive_dir, f"{day}_{base_filename}.txt")
    with open(p1, "w", encoding="utf-8") as f:
        f.write(text)

    # latest
    p2 = os.path.join(LATEST_ROOT, f"{base_filename}.txt")
    with open(p2, "w", encoding="utf-8") as f:
        f.write(text)


# ------------------- OKX API -------------------
def fetch_ohlcv(inst_id: str, bar: str, limit: int = LIMIT) -> pd.DataFrame:
    """
    OKX candles:
    data rows: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    ts is milliseconds.
    """
    def _request(url: str):
        resp = requests.get(url, timeout=15)
        data_json = resp.json()
        if data_json.get("code") not in (None, "0", 0):
            raise ValueError(f"API error: {data_json.get('msg') or data_json}")
        return data_json.get("data", [])

    # latest endpoint
    url_latest = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    try:
        data = _request(url_latest)
    except Exception as e:
        logging.warning("Latest candles endpoint failed, fallback to history: %s", e)
        url_hist = f"https://www.okx.com/api/v5/market/history-candles?instId={inst_id}&bar={bar}&limit={limit}"
        data = _request(url_hist)

    df = pd.DataFrame(data)
    if df.empty:
        return df

    cols = ["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"]
    if df.shape[1] < len(cols):
        cols = cols[: df.shape[1]]
    df.columns = cols

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ts ms -> UTC naive datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True).dt.tz_convert(None)

    # OKX newest-first -> reverse
    df = df.iloc[::-1].reset_index(drop=True)
    return df


# ------------------- Индикаторы / уровни -------------------
def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["high_low"] = d["high"] - d["low"]
    d["high_close"] = (d["high"] - d["close"].shift()).abs()
    d["low_close"] = (d["low"] - d["close"].shift()).abs()
    d["tr"] = d[["high_low", "high_close", "low_close"]].max(axis=1)
    d["atr"] = d["tr"].rolling(window=period, min_periods=1).mean()
    return d


def determine_trend(df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df_1h.empty or df_1d.empty:
        return None, None
    current_price = float(df_1h["close"].iloc[-1])
    last_daily_close = float(df_1d["close"].iloc[-1])
    local_trend = "long" if current_price > last_daily_close else "short"

    if len(df_1d) >= 6:
        global_trend = "long" if float(df_1d["close"].iloc[-1]) > float(df_1d["close"].iloc[-5]) else "short"
    else:
        global_trend = None
    return local_trend, global_trend


def find_key_levels_auto(df_1h: pd.DataFrame) -> List[Tuple[str, float, datetime]]:
    """Авто-уровни (fallback)."""
    levels: List[Tuple[str, float, datetime]] = []
    if df_1h.empty:
        return levels

    highs = df_1h["high"].rolling(window=20, min_periods=1).max()
    lows = df_1h["low"].rolling(window=20, min_periods=1).min()

    for i in range(1, len(df_1h) - 1):
        if df_1h["high"].iloc[i] == highs.iloc[i]:
            levels.append(("resistance", float(df_1h["high"].iloc[i]), df_1h["timestamp"].iloc[i]))
        if df_1h["low"].iloc[i] == lows.iloc[i]:
            levels.append(("support", float(df_1h["low"].iloc[i]), df_1h["timestamp"].iloc[i]))

    # убираем дубли близкие (грубо)
    levels_sorted = sorted(levels, key=lambda x: x[2])
    return levels_sorted[-200:]


def load_user_levels_from_latest() -> List[Tuple[str, float, datetime]]:
    """
    Читает уровни пользователя из latest/btc_levels_data.csv
    Ожидаемые колонки: type, price, timestamp
    """
    path = os.path.join(LATEST_ROOT, f"{BASE_FILES['levels']}.csv")
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    if df.empty or "type" not in df.columns or "price" not in df.columns:
        return []

    out: List[Tuple[str, float, datetime]] = []
    for row in df.itertuples(index=False):
        t = str(getattr(row, "type"))
        p = float(getattr(row, "price"))
        ts_raw = getattr(row, "timestamp", None)

        try:
            ts = pd.to_datetime(ts_raw)
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert(None)
            ts_dt = ts.to_pydatetime()
        except Exception:
            ts_dt = datetime.utcnow()

        # нормализуем type
        t_low = t.strip().lower()
        if "sup" in t_low or "поддерж" in t_low:
            t_norm = "support"
        elif "res" in t_low or "сопр" in t_low:
            t_norm = "resistance"
        else:
            t_norm = t_low  # как есть

        out.append((t_norm, p, ts_dt))

    return out[-400:]


def _nearest_levels(levels: List[Tuple[str, float, datetime]], price: float):
    if not levels or price is None or np.isnan(price):
        return None, None
    supports = [lvl for lvl in levels if lvl[0] == "support" and float(lvl[1]) <= float(price)]
    resistances = [lvl for lvl in levels if lvl[0] == "resistance" and float(lvl[1]) >= float(price)]
    nearest_sup = max(supports, key=lambda x: x[1]) if supports else None
    nearest_res = min(resistances, key=lambda x: x[1]) if resistances else None
    return nearest_sup, nearest_res


def _calc_mid_range(df_1h: pd.DataFrame, lookback: int = 48):
    if df_1h.empty:
        return None
    d = df_1h.tail(lookback)
    hi = float(d["high"].max())
    lo = float(d["low"].min())
    mid = (hi + lo) / 2.0
    rng = hi - lo
    return {"high": hi, "low": lo, "mid": mid, "range": rng}


def _impulse_ok(df_5m: pd.DataFrame, atr_1h: float) -> bool:
    if df_5m.empty or atr_1h is None or np.isnan(atr_1h) or len(df_5m) < IMPULSE_WINDOW_5M + 1:
        return False
    w = df_5m.tail(IMPULSE_WINDOW_5M)
    move = float(w["high"].max() - w["low"].min())
    return move >= float(atr_1h) * IMPULSE_ATR_MULT


def _wick_rejection(bar: pd.Series) -> Tuple[bool, str]:
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    top_wick = h - max(o, c)
    bot_wick = min(o, c) - l

    # значимый хвост: >40% диапазона и больше тела
    if top_wick / rng > 0.4 and top_wick > body:
        return True, "upper_wick"
    if bot_wick / rng > 0.4 and bot_wick > body:
        return True, "lower_wick"
    return False, ""


def _who_gets_punished(status_ru: str, reason_ru: str, direction: Optional[str]) -> str:
    if status_ru == "СДЕЛКА" and direction:
        return "накажут покупателей" if direction == "short" else "накажут продавцов"

    if "СЕРЕДИНА" in reason_ru:
        return "накажут тех, кто торгует середину диапазона"
    if "НЕТ ИМПУЛЬСА" in reason_ru:
        return "накажут нетерпеливых (рынок вялый — без импульса)"
    if "ЖДЁМ СВИП" in status_ru or "СНЯТИЕ ЛИКВИДНОСТИ" in reason_ru:
        return "накажут тех, кто входит ДО снятия ликвидности"
    if "НЕТ ОТКАЗА" in reason_ru or "ХВОСТ" in reason_ru:
        return "накажут тех, кто входит без подтверждения (без отказа)"
    if "НЕТ РЕТЕСТА" in reason_ru:
        return "накажут тех, кто догоняет цену без ретеста"
    return "не ясно; лучше подождать подтверждение"


# ------------------- Паттерны (на H1) -------------------
def check_patterns_1h(df_1h: pd.DataFrame, levels: List[Tuple[str, float, datetime]]):
    """Простейшие паттерны относительно уровней (H1)."""
    signals = []
    if df_1h.shape[0] < 2 or not levels:
        return signals

    last_bar = df_1h.iloc[-1]
    prev_bar = df_1h.iloc[-2]

    for level_type, level_price, _ in levels:
        lp = float(level_price)

        # ложный пробой (ЛП)
        if (
            level_type == "resistance"
            and float(prev_bar["high"]) > lp
            and float(last_bar["close"]) < lp
        ) or (
            level_type == "support"
            and float(prev_bar["low"]) < lp
            and float(last_bar["close"]) > lp
        ):
            signals.append(("ложный пробой (ЛП)", lp, last_bar["timestamp"]))

        # пробой
        if level_type == "resistance" and float(last_bar["close"]) > lp and float(prev_bar["close"]) < lp:
            signals.append(("пробой вверх", lp, last_bar["timestamp"]))
        if level_type == "support" and float(last_bar["close"]) < lp and float(prev_bar["close"]) > lp:
            signals.append(("пробой вниз", lp, last_bar["timestamp"]))

    return signals


# ------------------- Сетапы (M5 фильтры + уровни) -------------------
def assess_setups(df_5m: pd.DataFrame, df_1h: pd.DataFrame, key_levels: List[Tuple[str, float, datetime]]) -> Tuple[List[Dict], List[Dict]]:
    if df_5m.empty or df_1h.empty:
        return [], []

    price = float(df_5m["close"].iloc[-1])
    atr_1h = float(df_1h["atr"].iloc[-1]) if "atr" in df_1h.columns else np.nan
    rng = _calc_mid_range(df_1h)
    nearest_sup, nearest_res = _nearest_levels(key_levels, price)

    rows: List[Dict] = []
    attention: List[Dict] = []

    def _near_level(level_price: float) -> bool:
        tol = float(atr_1h) * RETEST_ATR_TOL if atr_1h and not np.isnan(atr_1h) else 50.0
        return abs(price - level_price) <= tol

    # 1) NO TRADE: нет импульса
    if not _impulse_ok(df_5m, atr_1h):
        status_ru = "НЕТ СДЕЛКИ"
        reason_ru = "НЕТ ИМПУЛЬСА (рынок вялый/нет расширения диапазона)"
        direction = None
        rows.append({
            "timestamp": df_5m["timestamp"].iloc[-1],
            "inst": SYMBOL_FUTURES,
            "price": price,
            "status": status_ru,
            "reason": reason_ru,
            "direction": direction,
            "level": np.nan,
            "atr_1h": atr_1h,
            "who_punished": _who_gets_punished(status_ru, reason_ru, direction),
        })
        return rows, attention

    # 2) NO TRADE: середина диапазона
    if rng and rng["range"] > 0:
        dist_to_mid = abs(price - rng["mid"])
        if dist_to_mid <= rng["range"] * MID_RANGE_BAND:
            status_ru = "НЕТ СДЕЛКИ"
            reason_ru = "СЕРЕДИНА ДИАПАЗОНА (лучшее место для наказания за нетерпение)"
            direction = None
            rows.append({
                "timestamp": df_5m["timestamp"].iloc[-1],
                "inst": SYMBOL_FUTURES,
                "price": price,
                "status": status_ru,
                "reason": reason_ru,
                "direction": direction,
                "level": np.nan,
                "atr_1h": atr_1h,
                "who_punished": _who_gets_punished(status_ru, reason_ru, direction),
            })
            return rows, attention

    # 3) NO TRADE: нет отказа на M5
    last_bar = df_5m.iloc[-1]
    has_rej, rej_type = _wick_rejection(last_bar)
    if not has_rej:
        status_ru = "НЕТ СДЕЛКИ"
        reason_ru = "НЕТ ОТКАЗА (нет хвоста/поглощения на M5)"
        direction = None

        if nearest_sup and _near_level(float(nearest_sup[1])):
            attention.append({
                "timestamp": df_5m["timestamp"].iloc[-1],
                "inst": SYMBOL_FUTURES,
                "price": price,
                "type": "ТОЧКА ВНИМАНИЯ",
                "level": float(nearest_sup[1]),
                "comment": "Подходим к поддержке, но отказа пока нет — ждём реакцию/хвост.",
                "atr_1h": atr_1h,
            })
        if nearest_res and _near_level(float(nearest_res[1])):
            attention.append({
                "timestamp": df_5m["timestamp"].iloc[-1],
                "inst": SYMBOL_FUTURES,
                "price": price,
                "type": "ТОЧКА ВНИМАНИЯ",
                "level": float(nearest_res[1]),
                "comment": "Подходим к сопротивлению, но отказа пока нет — ждём реакцию/хвост.",
                "atr_1h": atr_1h,
            })

        rows.append({
            "timestamp": df_5m["timestamp"].iloc[-1],
            "inst": SYMBOL_FUTURES,
            "price": price,
            "status": status_ru,
            "reason": reason_ru,
            "direction": direction,
            "level": np.nan,
            "atr_1h": atr_1h,
            "who_punished": _who_gets_punished(status_ru, reason_ru, direction),
        })
        return rows, attention

    # 4) Логика свип/ретест вокруг ближайшего уровня
    tol = float(atr_1h) * RETEST_ATR_TOL if atr_1h and not np.isnan(atr_1h) else 50.0
    window = df_5m.tail(RETEST_WINDOW_5M)

    def _touched_level(level_price: float) -> bool:
        return bool(((window["low"] <= level_price + tol) & (window["high"] >= level_price - tol)).any())

    # ШОРТ: отказ сверху + работаем от сопротивления
    if rej_type == "upper_wick" and nearest_res:
        lvl = float(nearest_res[1])
        swept = bool((window["high"] > lvl + tol).any())  # вынесли выше
        retested = _touched_level(lvl)                   # вернулись к уровню

        if not swept:
            status_ru = "ЖДЁМ СВИП"
            reason_ru = "НЕТ СНЯТИЯ ЛИКВИДНОСТИ (не вынесли выше уровня)"
        elif swept and not retested:
            status_ru = "НЕДЕЙСТВИТЕЛЬНО"
            reason_ru = "ЗАПРЕТ СДЕЛКИ БЕЗ РЕТЕСТА (цена улетела после свипа)"
        else:
            status_ru = "СДЕЛКА"
            reason_ru = "ЕСТЬ СВИП + ОТКАЗ + РЕТЕСТ (валидный шорт)"

        direction = "short"
        rows.append({
            "timestamp": df_5m["timestamp"].iloc[-1],
            "inst": SYMBOL_FUTURES,
            "price": price,
            "status": status_ru,
            "reason": reason_ru,
            "direction": direction,
            "level": lvl,
            "atr_1h": atr_1h,
            "who_punished": _who_gets_punished(status_ru, reason_ru, direction),
        })

        if status_ru == "ЖДЁМ СВИП" and _near_level(lvl):
            attention.append({
                "timestamp": df_5m["timestamp"].iloc[-1],
                "inst": SYMBOL_FUTURES,
                "price": price,
                "type": "ЖДЁМ",
                "level": lvl,
                "comment": "Рядом сопротивление. Ждём вынос (свип) и отказ для шорта.",
                "atr_1h": atr_1h,
            })

        return rows, attention

    # ЛОНГ: отказ снизу + работаем от поддержки
    if rej_type == "lower_wick" and nearest_sup:
        lvl = float(nearest_sup[1])
        swept = bool((window["low"] < lvl - tol).any())   # вынесли ниже
        retested = _touched_level(lvl)

        if not swept:
            status_ru = "ЖДЁМ СВИП"
            reason_ru = "НЕТ СНЯТИЯ ЛИКВИДНОСТИ (не вынесли ниже уровня)"
        elif swept and not retested:
            status_ru = "НЕДЕЙСТВИТЕЛЬНО"
            reason_ru = "ЗАПРЕТ СДЕЛКИ БЕЗ РЕТЕСТА (цена улетела после свипа)"
        else:
            status_ru = "СДЕЛКА"
            reason_ru = "ЕСТЬ СВИП + ОТКАЗ + РЕТЕСТ (валидный лонг)"

        direction = "long"
        rows.append({
            "timestamp": df_5m["timestamp"].iloc[-1],
            "inst": SYMBOL_FUTURES,
            "price": price,
            "status": status_ru,
            "reason": reason_ru,
            "direction": direction,
            "level": lvl,
            "atr_1h": atr_1h,
            "who_punished": _who_gets_punished(status_ru, reason_ru, direction),
        })

        if status_ru == "ЖДЁМ СВИП" and _near_level(lvl):
            attention.append({
                "timestamp": df_5m["timestamp"].iloc[-1],
                "inst": SYMBOL_FUTURES,
                "price": price,
                "type": "ЖДЁМ",
                "level": lvl,
                "comment": "Рядом поддержка. Ждём вынос (свип) и отказ для лонга.",
                "atr_1h": atr_1h,
            })

        return rows, attention

    # Есть хвост, но уровни не подходят / далеко
    status_ru = "НЕТ СДЕЛКИ"
    reason_ru = "НЕТ БЛИЖАЙШЕГО УРОВНЯ ДЛЯ РАБОТЫ (уровни далеко/непонятно)"
    direction = None
    rows.append({
        "timestamp": df_5m["timestamp"].iloc[-1],
        "inst": SYMBOL_FUTURES,
        "price": price,
        "status": status_ru,
        "reason": reason_ru,
        "direction": direction,
        "level": np.nan,
        "atr_1h": atr_1h,
        "who_punished": _who_gets_punished(status_ru, reason_ru, direction),
    })
    return rows, attention


# ------------------- Сценарий / Ликвидность (текст) -------------------
def _format_level(x: float) -> str:
    try:
        return f"{float(x):,.0f}".replace(",", " ")
    except Exception:
        return str(x)


def infer_liquidity_pools(levels: List[Tuple[str, float, datetime]], price: float) -> Dict[str, List[float]]:
    """Грубая логика: собираем уровни ниже и выше цены."""
    supports = sorted([float(l[1]) for l in levels if l[0] == "support"], reverse=True)
    resist = sorted([float(l[1]) for l in levels if l[0] == "resistance"])
    below = [x for x in supports if x <= price][:5]
    above = [x for x in resist if x >= price][:5]
    return {"below": below, "above": above}


def build_scenario_text_ru(
    now_str: str,
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
    key_levels: List[Tuple[str, float, datetime]],
    setups_df: pd.DataFrame,
) -> str:
    price = float(df_5m["close"].iloc[-1])
    atr_1h = float(df_1h["atr"].iloc[-1]) if "atr" in df_1h.columns else np.nan
    rng = _calc_mid_range(df_1h)
    pools = infer_liquidity_pools(key_levels, price)

    # Состояние рынка
    is_impulse = _impulse_ok(df_5m, atr_1h)
    state = "ВЯЛЫЙ РЫНОК" if not is_impulse else "РЫНОК ДАЛ ИМПУЛЬС"
    forbid = "⛔ Сделки запрещены до появления импульса" if not is_impulse else "✅ Можно рассматривать сценарии ТОЛЬКО по правилам"

    # Последний статус бота
    last_line = ""
    if not setups_df.empty:
        last = setups_df.iloc[-1].to_dict()
        last_line = (
            f"Последнее обновление: {now_str}\n"
            f"Статус: {last.get('status','')}\n"
            f"Причина: {last.get('reason','')}\n"
            f"Кого накажут, если войти: {last.get('who_punished','')}\n"
        )
    else:
        last_line = f"Последнее обновление: {now_str}\nСтатус: нет данных\n"

    # Где ликвидность
    liq_text = []
    liq_text.append("ГДЕ СЕЙЧАС ЛИКВИДНОСТЬ")
    liq_text.append("Снизу (очевидные зоны):")
    if pools["below"]:
        for x in pools["below"]:
            liq_text.append(f"- {_format_level(x)}")
    else:
        liq_text.append("- нет уровней ниже (проверь btc_levels_data)")

    liq_text.append("\nСверху (очевидные зоны):")
    if pools["above"]:
        for x in pools["above"]:
            liq_text.append(f"- {_format_level(x)}")
    else:
        liq_text.append("- нет уровней выше (проверь btc_levels_data)")

    # План сценариев (НЕ ордера)
    # Берём ближайшую поддержку/сопротивление от текущей цены
    nearest_sup, nearest_res = _nearest_levels(key_levels, price)
    sup_lvl = float(nearest_sup[1]) if nearest_sup else None
    res_lvl = float(nearest_res[1]) if nearest_res else None

    plan = []
    plan.append("СЦЕНАРИЙ (НЕ ордер, а план действий)")
    plan.append("")
    plan.append(f"Текущая цена: {_format_level(price)} | ATR(1H): {_format_level(atr_1h) if not np.isnan(atr_1h) else 'нет'}")

    if rng and rng["range"] > 0:
        plan.append(f"Диапазон (H1 ~48 свечей): low {_format_level(rng['low'])} / mid {_format_level(rng['mid'])} / high {_format_level(rng['high'])}")

    plan.append("")
    plan.append("Состояние: " + state)
    plan.append(forbid)
    plan.append("")

    # Ждём (лонг)
    if sup_lvl is not None:
        plan.append("ДОПУСТИМЫЙ ПЛАН ЛОНГА (ТОЛЬКО ПОСЛЕ СВИПА + ОТКАЗА)")
        plan.append(f"ЖДЁМ:\n- вынос ниже {_format_level(sup_lvl)}\n- быстрый возврат выше уровня\n- хвост снизу на M5/M15\n- рынок НЕ может закрепиться ниже")
        plan.append("Если будет медленное сползание без реакции → НЕТ СДЕЛКИ.")
        plan.append("")

    # Ждём (шорт)
    if res_lvl is not None:
        plan.append("ДОПУСТИМЫЙ ПЛАН ШОРТА (ТОЛЬКО ПОСЛЕ СВИПА + ОТКАЗА)")
        plan.append(f"ЖДЁМ:\n- вынос выше {_format_level(res_lvl)}\n- быстрый возврат ниже уровня\n- хвост сверху на M5/M15\n- рынок НЕ может удержаться выше")
        plan.append("Если рынок просто ползёт у уровня без свипа → ЖДЁМ.")
        plan.append("")

    return "\n".join([
        f"BTCUSDT (OKX {SYMBOL_FUTURES})",
        "",
        f"{state}",
        forbid,
        "",
        last_line.strip(),
        "",
        "\n".join(liq_text),
        "",
        "\n".join(plan),
        "",
        "ПРИМЕЧАНИЕ: это не финансовый совет. Это фильтр/сценарии по твоим правилам (свип → отказ → ретест)."
    ])


# ------------------- Визуализация (если есть matplotlib) -------------------
def make_viz(df_5m: pd.DataFrame, level: Optional[float], status: str, reason: str, out_path: str):
    if not MATPLOTLIB_AVAILABLE:
        return
    if df_5m.empty:
        return

    d = df_5m.tail(120).copy()  # ~10 часов M5
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    for i, row in enumerate(d.itertuples(index=False)):
        o = float(getattr(row, "open"))
        h = float(getattr(row, "high"))
        l = float(getattr(row, "low"))
        c = float(getattr(row, "close"))
        ax.vlines(i, l, h, linewidth=1)
        y0 = min(o, c)
        y1 = max(o, c)
        ax.vlines(i, y0, y1, linewidth=4)

    if level is not None and not (isinstance(level, float) and np.isnan(level)):
        ax.axhline(level, linestyle="--", linewidth=1)
        ax.text(0.01, 0.98, f"Уровень: {level:.0f}", transform=ax.transAxes, va="top")

    ax.set_title(f"BTC M5 | {status} | {reason}")
    ax.set_xlabel("Последние свечи (M5)")
    ax.set_ylabel("Цена")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_viz_if_possible(df_5m: pd.DataFrame, status: str, reason: str, level: Optional[float], date_str: str):
    if not MATPLOTLIB_AVAILABLE:
        return
    archive_dir = get_archive_path(date_str)
    os.makedirs(archive_dir, exist_ok=True)
    day = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").strftime("%d")
    viz_dir = os.path.join(archive_dir, BASE_FILES["viz_dir"])
    os.makedirs(viz_dir, exist_ok=True)

    file_path = os.path.join(viz_dir, f"{day}_{SYMBOL_FUTURES}.png")
    make_viz(df_5m, level=level, status=status, reason=reason, out_path=file_path)

    # latest копия
    latest_viz_dir = os.path.join(LATEST_ROOT, BASE_FILES["viz_dir"])
    os.makedirs(latest_viz_dir, exist_ok=True)
    latest_path = os.path.join(latest_viz_dir, f"{SYMBOL_FUTURES}.png")
    try:
        # копируем байтами
        with open(file_path, "rb") as src, open(latest_path, "wb") as dst:
            dst.write(src.read())
    except Exception:
        pass


# ------------------- MAIN -------------------
def main():
    try:
        ensure_dirs()

        # Время Алматы
        almaty_tz = pytz.timezone("Asia/Almaty")
        almaty_time = datetime.now(almaty_tz)
        date_str = almaty_time.strftime("%Y-%m-%d %H:%M:%S")
        logging.info("Старт: %s (Asia/Almaty)", date_str)

        # Данные
        df_5m = fetch_ohlcv(SYMBOL_FUTURES, TIMEFRAMES["5m"], limit=300)
        df_1h = fetch_ohlcv(SYMBOL_FUTURES, TIMEFRAMES["1h"], limit=250)
        df_1d = fetch_ohlcv(SYMBOL_FUTURES, TIMEFRAMES["1d"], limit=200)

        if df_5m.empty or df_1h.empty or df_1d.empty:
            logging.warning("OHLCV пустые. Прерывание.")
            return

        df_1h = calculate_atr(df_1h)
        df_1d = calculate_atr(df_1d)

        local_trend, global_trend = determine_trend(df_1h, df_1d)

        # 1) Авто-уровни (fallback)
        auto_levels = find_key_levels_auto(df_1h)

        # 2) Всегда сохраняем авто-уровни в архив+latest как базу (чтобы файл был)
        if auto_levels:
            df_levels_auto = pd.DataFrame(auto_levels, columns=["type", "price", "timestamp"])
            save_to_archive(df_levels_auto, BASE_FILES["levels"], date_str, header=True)

        # 3) Читаем ТВОИ уровни из latest (если ты их заменил руками — они приоритет)
        user_levels = load_user_levels_from_latest()

        # 4) Выбираем уровни
        key_levels = user_levels if user_levels else auto_levels

        # Паттерны (H1) относительно уровней
        patterns = check_patterns_1h(df_1h, key_levels)
        if patterns:
            df_patterns = pd.DataFrame(patterns, columns=["pattern", "level_price", "timestamp"])
            save_to_archive(df_patterns, BASE_FILES["patterns"], date_str, header=True)

        # Сетапы / точки внимания
        setups_rows, attention_rows = assess_setups(df_5m=df_5m, df_1h=df_1h, key_levels=key_levels)
        df_setups = pd.DataFrame(setups_rows)
        df_attention = pd.DataFrame(attention_rows)

        if not df_setups.empty:
            save_to_archive(df_setups, BASE_FILES["setups"], date_str, header=True)
        if not df_attention.empty:
            save_to_archive(df_attention, BASE_FILES["attention"], date_str, header=True)

        # Сохраняем OHLCV
        save_to_archive(df_1h, BASE_FILES["ohlcv_1h"], date_str, header=True)
        save_to_archive(df_1d, BASE_FILES["ohlcv_1d"], date_str, header=True)

        # Визуализация (если есть matplotlib)
        if not df_setups.empty:
            last = df_setups.iloc[-1].to_dict()
            save_viz_if_possible(
                df_5m=df_5m,
                status=str(last.get("status", "")),
                reason=str(last.get("reason", "")),
                level=last.get("level", None),
                date_str=date_str
            )

        # Текст сценария (как ты просил: состояние/ждём/где ликвидность/не ордер)
        scenario_text = build_scenario_text_ru(
            now_str=date_str,
            df_5m=df_5m,
            df_1h=df_1h,
            df_1d=df_1d,
            key_levels=key_levels,
            setups_df=df_setups
        )
        save_text_report(scenario_text, date_str, BASE_FILES["scenario"])

        # Логи
        logging.info("Инструмент: %s", SYMBOL_FUTURES)
        logging.info("Локальный тренд: %s | Глобальный тренд: %s", local_trend, global_trend)
        logging.info("Уровни: %s (используем %s)", len(key_levels), "ПОЛЬЗОВАТЕЛЬСКИЕ" if user_levels else "АВТО")
        if not df_setups.empty:
            logging.info("Последний статус: %s | %s", df_setups["status"].iloc[-1], df_setups["reason"].iloc[-1])
        else:
            logging.info("Статус: нет записей setups")

        logging.info("Готово. Файлы обновлены в latest/ и записаны в archive/")

    except Exception as e:
        logging.exception("Ошибка в main: %s", e)


if __name__ == "__main__":
    main()

