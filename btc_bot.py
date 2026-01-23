import requests
import pandas as pd
import os
from datetime import datetime
import pytz
import numpy as np
import logging
import sys
from typing import Optional, Tuple, List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Настройки
BASE_FILES = {
    "levels": "btc_levels_data",
    "patterns": "btc_patterns_data",
    "ohlcv_1h": "btc_ohlcv_1h",
    "ohlcv_1d": "btc_ohlcv_1d"
}
"""BTC bot (OKX).

Fixes and upgrades:
1) Use OKX perpetual swap instrument: BTC-USDT-SWAP (spot symbol caused price mismatch).
2) Add a strict but lightweight trade filter logic:
   - "Запрет сделки без ретеста": mark setup as INVALID if price breaks a key level
     and does not return to it (retest) within a limited window.
   - "NO TRADE" scenarios: no impulse / no rejection / price is in the middle of range.

This is NOT a full trading system. It's a data + signal helper aligned with the user's
rules: trade only after liquidity sweep + rejection + retest, otherwise stand aside.
"""

SYMBOL_FUTURES = "BTC-USDT-SWAP"
TIMEFRAMES = {"5m": "5m", "1h": "1H", "1d": "1D"}
ATR_PERIOD = 14
ROUND_LEVELS = [1000, 500, 100]
LIMIT = 100
ARCHIVE_ROOT = "archive"

# New outputs
BASE_FILES["setups"] = "btc_setups_data"              # сделки/валидаторы
BASE_FILES["attention"] = "btc_attention_points_data"  # точки внимания (не сделки)
BASE_FILES["viz"] = "btc_viz"                          # изображения (пояснения)

# --- Heuristic parameters (tune if needed) ---
# Retest tolerance relative to ATR(1H)
RETEST_ATR_TOL = 0.15
# How many 5m candles we give the market to retest after a break
RETEST_WINDOW_5M = 12   # 12*5m = 60 minutes
# Impulse definition: move >= IMPULSE_ATR_MULT * ATR(1H) over last IMPULSE_WINDOW_5M candles
IMPULSE_ATR_MULT = 0.7
IMPULSE_WINDOW_5M = 6   # 30 minutes
# Middle-of-range filter: if price is inside this % band around range mid -> NO TRADE
MID_RANGE_BAND = 0.20

# Логирование (вывод в консоль — видно в Actions)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)

def get_archive_path(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    year = dt.strftime('%Y')
    month = dt.strftime('%m')
    week_num = dt.isocalendar()[1]
    day = dt.strftime('%d')
    # папка: archive/YYYY/MM/week_XX/DD/
    return os.path.join(ARCHIVE_ROOT, year, month, f"week_{week_num:02d}", f"{day}")

def save_to_archive(df, base_filename, date_str, header=True):
    archive_dir = get_archive_path(date_str)
    os.makedirs(archive_dir, exist_ok=True)
    day = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime('%d')
    file_path = os.path.join(archive_dir, f"{day}_{base_filename}.csv")
    mode = 'a' if os.path.exists(file_path) else 'w'
    df.to_csv(file_path, mode=mode, header=header and mode == 'w', index=False)
    return file_path

def fetch_ohlcv(inst_id, bar, limit=LIMIT):
    """Fetch candles from OKX.

    We prefer the fast *latest* endpoint (/market/candles) for timely data.
    If it fails (rare), we fall back to /market/history-candles.

    OKX returns arrays with columns:
      [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    ts is in milliseconds.
    """

    def _request(url: str):
        resp = requests.get(url, timeout=15)
        data_json = resp.json()
        if data_json.get('code') not in (None, '0', 0):
            raise ValueError(f"API error: {data_json.get('msg') or data_json}")
        return data_json.get('data', [])

    # 1) Latest candles (faster and usually more up-to-date)
    url_latest = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    try:
        data = _request(url_latest)
    except Exception as e:
        logging.warning('Latest candles endpoint failed, fallback to history: %s', e)
        url_hist = f"https://www.okx.com/api/v5/market/history-candles?instId={inst_id}&bar={bar}&limit={limit}"
        data = _request(url_hist)

    df = pd.DataFrame(data)
    if df.empty:
        return df

    cols = [
        'timestamp', 'open', 'high', 'low', 'close',
        'volume', 'volCcy', 'volCcyQuote', 'confirm'
    ]
    if df.shape[1] < len(cols):
        cols = cols[:df.shape[1]]
    df.columns = cols

    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # timestamp in ms -> datetime (UTC naive)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True).dt.tz_convert(None)

    # OKX returns newest-first; reverse to chronological
    df = df.iloc[::-1].reset_index(drop=True)
    return df


def calculate_atr(df, period=ATR_PERIOD):
    if df.empty:
        return df
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = (df['high'] - df['close'].shift()).abs()
    df['low_close'] = (df['low'] - df['close'].shift()).abs()
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period, min_periods=1).mean()
    return df

def determine_trend(df_1h, df_1d):
    if df_1h.empty or df_1d.empty:
        return None, None
    current_price = df_1h['close'].iloc[-1]
    last_daily_close = df_1d['close'].iloc[-1]
    local_trend = 'long' if current_price > last_daily_close else 'short'
    # глобальный тренд: сравнение последнего дневного закрытия с закрытием 5 дней назад
    if len(df_1d) >= 6:
        global_trend = 'long' if df_1d['close'].iloc[-1] > df_1d['close'].iloc[-5] else 'short'
    else:
        global_trend = None
    return local_trend, global_trend

def find_key_levels(df):
    levels = []
    if df.empty:
        return levels
    highs = df['high'].rolling(window=20, min_periods=1).max()
    lows = df['low'].rolling(window=20, min_periods=1).min()
    for i in range(1, len(df)-1):
        if df['high'].iloc[i] == highs.iloc[i]:
            levels.append(('resistance', float(df['high'].iloc[i]), df['timestamp'].iloc[i]))
        if df['low'].iloc[i] == lows.iloc[i]:
            levels.append(('support', float(df['low'].iloc[i]), df['timestamp'].iloc[i]))
    return levels

def find_round_levels(price, round_levels=ROUND_LEVELS):
    if price is None or np.isnan(price):
        return None
    for level in round_levels:
        rounded = round(price / level) * level
        if abs(price - rounded) < level * 0.1:
            return rounded
    return None

def check_patterns(df, levels):
    """Detect simple price-action patterns relative to key levels (1H).

    Returns list of tuples: (pattern, level_price, timestamp)
    """
    signals = []
    if df.shape[0] < 2:
        return signals

    last_bar = df.iloc[-1]
    prev_bar = df.iloc[-2]

    for level_type, level_price, _ in levels:
        # false breakout (LP)
        if (
            level_type == 'resistance'
            and prev_bar['high'] > level_price
            and last_bar['close'] < level_price
        ) or (
            level_type == 'support'
            and prev_bar['low'] < level_price
            and last_bar['close'] > level_price
        ):
            signals.append(('false_breakout', float(level_price), last_bar['timestamp']))

        # breakout
        if level_type == 'resistance' and last_bar['close'] > level_price and prev_bar['close'] < level_price:
            signals.append(('breakout', float(level_price), last_bar['timestamp']))

        # bounce from support
        atr_val = float(last_bar.get('atr', 0)) if 'atr' in df.columns else 0.0
        if level_type == 'support' and atr_val and abs(float(last_bar['low']) - float(level_price)) < atr_val * 0.1 and float(last_bar['close']) > float(level_price):
            signals.append(('bounce', float(level_price), last_bar['timestamp']))

    return signals
def _nearest_levels(levels, price: float):
    """Pick nearest support below and resistance above current price.

    levels: list of tuples (type, price, timestamp)
    """
    if not levels or price is None or np.isnan(price):
        return None, None
    supports = [lvl for lvl in levels if lvl[0] == 'support' and float(lvl[1]) <= float(price)]
    resistances = [lvl for lvl in levels if lvl[0] == 'resistance' and float(lvl[1]) >= float(price)]
    nearest_sup = max(supports, key=lambda x: x[1]) if supports else None
    nearest_res = min(resistances, key=lambda x: x[1]) if resistances else None
    return nearest_sup, nearest_res


def _calc_mid_range(df_1h, lookback=48):
    """Define an intraday range using recent 1H candles."""
    if df_1h.empty:
        return None
    d = df_1h.tail(lookback)
    hi = float(d['high'].max())
    lo = float(d['low'].min())
    mid = (hi + lo) / 2.0
    rng = hi - lo
    return {'high': hi, 'low': lo, 'mid': mid, 'range': rng}


def _impulse_ok(df_5m, atr_1h: float):
    """Impulse filter: need a decent move in a short window."""
    if df_5m.empty or atr_1h is None or np.isnan(atr_1h) or len(df_5m) < IMPULSE_WINDOW_5M + 1:
        return False
    w = df_5m.tail(IMPULSE_WINDOW_5M)
    move = float(w['high'].max() - w['low'].min())
    return move >= float(atr_1h) * IMPULSE_ATR_MULT


def _wick_rejection(bar) -> Tuple[bool, str]:
    """Detect rejection wick on a single candle."""
    o, h, l, c = float(bar['open']), float(bar['high']), float(bar['low']), float(bar['close'])
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    top_wick = h - max(o, c)
    bot_wick = min(o, c) - l
    # "Хвост" считаем значимым, если > 40% диапазона и больше тела
    if top_wick / rng > 0.4 and top_wick > body:
        return True, 'upper_wick'
    if bot_wick / rng > 0.4 and bot_wick > body:
        return True, 'lower_wick'
    return False, ''


def _who_gets_punished(status_ru: str, reason_ru: str, direction: Optional[str]) -> str:
    """Human-friendly heuristic: 'кого накажут, если войти сейчас'."""
    # Если сетап валиден — наказывают толпу по сторону ложного движения
    if status_ru == "СДЕЛКА" and direction:
        return "накажут покупателей" if direction == "short" else "накажут продавцов"

    # Универсальные ответы для фильтров
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

    # По умолчанию
    return "не ясно; лучше подождать подтверждение"


def assess_setups(df_5m: pd.DataFrame, df_1h: pd.DataFrame, key_levels) -> Tuple[List[Dict], List[Dict]]:
    """Оценка сетапа по твоему алгоритму.

    Возвращает:
      1) setups_rows — строки по статусу (СДЕЛКА / НЕТ СДЕЛКИ / НЕДЕЙСТВИТЕЛЬНО / ЖДЁМ СВИП)
      2) attention_rows — 'ТОЧКИ ВНИМАНИЯ' (когда уровень рядом, но условий ещё нет)

    Важно: вывод строго на русском (без английских терминов).
    """
    if df_5m.empty or df_1h.empty:
        return [], []

    price = float(df_5m['close'].iloc[-1])
    atr_1h = float(df_1h['atr'].iloc[-1]) if 'atr' in df_1h.columns else np.nan
    rng = _calc_mid_range(df_1h)
    nearest_sup, nearest_res = _nearest_levels(key_levels, price)

    rows: List[Dict] = []
    attention: List[Dict] = []

    # helper: уровень рядом?
    def _near_level(level_price: float) -> bool:
        if atr_1h and not np.isnan(atr_1h):
            tol = float(atr_1h) * RETEST_ATR_TOL
        else:
            tol = 50.0
        return abs(price - level_price) <= tol
    # Basic NO_TRADE filters
    if not _impulse_ok(df_5m, atr_1h):
        status_ru = "НЕТ СДЕЛКИ"
        reason_ru = "НЕТ ИМПУЛЬСА (рынок ползёт/нет расширения диапазона)"
        direction = None
        rows.append({
            'timestamp': df_5m['timestamp'].iloc[-1],
            'inst': SYMBOL_FUTURES,
            'price': price,
            'status': status_ru,
            'reason': reason_ru,
            'direction': direction,
            'level': np.nan,
            'atr_1h': atr_1h,
            'who_punished': _who_gets_punished(status_ru, reason_ru, direction),
        })
        return rows, attention

    if rng and rng['range'] > 0:
        dist_to_mid = abs(price - rng['mid'])
        if dist_to_mid <= rng['range'] * MID_RANGE_BAND:
            status_ru = "НЕТ СДЕЛКИ"
            reason_ru = "СЕРЕДИНА ДИАПАЗОНА (лучшее место для наказания за нетерпение)"
            direction = None
            rows.append({
                'timestamp': df_5m['timestamp'].iloc[-1],
                'inst': SYMBOL_FUTURES,
                'price': price,
                'status': status_ru,
                'reason': reason_ru,
                'direction': direction,
                'level': np.nan,
                'atr_1h': atr_1h,
                'who_punished': _who_gets_punished(status_ru, reason_ru, direction),
            })
            return rows, attention

    last_bar = df_5m.iloc[-1]
    has_rej, rej_type = _wick_rejection(last_bar)
    if not has_rej:
        status_ru = "НЕТ СДЕЛКИ"
        reason_ru = "НЕТ ОТКАЗА (нет хвоста/поглощения на M5)"
        direction = None

        # если при этом мы рядом с уровнем — это точка внимания
        if nearest_sup and _near_level(float(nearest_sup[1])):
            attention.append({
                'timestamp': df_5m['timestamp'].iloc[-1],
                'inst': SYMBOL_FUTURES,
                'price': price,
                'type': 'ВНИМАНИЕ',
                'level': float(nearest_sup[1]),
                'comment': 'Подходим к поддержке, но отказа пока нет — ждём реакцию/хвост.',
                'atr_1h': atr_1h,
            })
        if nearest_res and _near_level(float(nearest_res[1])):
            attention.append({
                'timestamp': df_5m['timestamp'].iloc[-1],
                'inst': SYMBOL_FUTURES,
                'price': price,
                'type': 'ВНИМАНИЕ',
                'level': float(nearest_res[1]),
                'comment': 'Подходим к сопротивлению, но отказа пока нет — ждём реакцию/хвост.',
                'atr_1h': atr_1h,
            })

        rows.append({
            'timestamp': df_5m['timestamp'].iloc[-1],
            'inst': SYMBOL_FUTURES,
            'price': price,
            'status': status_ru,
            'reason': reason_ru,
            'direction': direction,
            'level': np.nan,
            'atr_1h': atr_1h,
            'who_punished': _who_gets_punished(status_ru, reason_ru, direction),
        })
        return rows, attention

    # Retest logic around nearest key level, depending on rejection direction
    tol = float(atr_1h) * RETEST_ATR_TOL if atr_1h and not np.isnan(atr_1h) else 50.0

    window = df_5m.tail(RETEST_WINDOW_5M)

    def _touched_level(level_price: float) -> bool:
        return bool(((window['low'] <= level_price + tol) & (window['high'] >= level_price - tol)).any())

    # If we rejected from above -> prefer short around resistance
    if rej_type == 'upper_wick' and nearest_res:
        lvl = float(nearest_res[1])
        # Break/sweep: price went above level in window
        swept = bool((window['high'] > lvl + tol).any())
        retested = _touched_level(lvl)
        if not swept:
            status_ru = "ЖДЁМ СВИП"
            reason_ru = "НЕТ СНЯТИЯ ЛИКВИДНОСТИ (не вынесли выше уровня)"
        elif swept and not retested:
            status_ru = "НЕДЕЙСТВИТЕЛЬНО"
            reason_ru = "ЗАПРЕТ СДЕЛКИ БЕЗ РЕТЕСТА (цена улетела после свипа)"
        else:
            status_ru = "СДЕЛКА"
            reason_ru = "ЕСТЬ СВИП + ОТКАЗ + РЕТЕСТ (валидный шорт)"

        direction = 'short'
        rows.append({
            'timestamp': df_5m['timestamp'].iloc[-1],
            'inst': SYMBOL_FUTURES,
            'price': price,
            'status': status_ru,
            'reason': reason_ru,
            'direction': direction,
            'level': lvl,
            'atr_1h': atr_1h,
            'who_punished': _who_gets_punished(status_ru, reason_ru, direction),
        })
        # точка внимания, если уровень рядом, но ещё нет свипа
        if status_ru == "ЖДЁМ СВИП" and _near_level(lvl):
            attention.append({
                'timestamp': df_5m['timestamp'].iloc[-1],
                'inst': SYMBOL_FUTURES,
                'price': price,
                'type': 'ОЖИДАНИЕ',
                'level': lvl,
                'comment': 'Рядом сопротивление. Ждём вынос (свип) и отказ для шорта.',
                'atr_1h': atr_1h,
            })
        return rows, attention

    # If we rejected from below -> prefer long around support
    if rej_type == 'lower_wick' and nearest_sup:
        lvl = float(nearest_sup[1])
        swept = bool((window['low'] < lvl - tol).any())
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

        direction = 'long'
        rows.append({
            'timestamp': df_5m['timestamp'].iloc[-1],
            'inst': SYMBOL_FUTURES,
            'price': price,
            'status': status_ru,
            'reason': reason_ru,
            'direction': direction,
            'level': lvl,
            'atr_1h': atr_1h,
            'who_punished': _who_gets_punished(status_ru, reason_ru, direction),
        })
        if status_ru == "ЖДЁМ СВИП" and _near_level(lvl):
            attention.append({
                'timestamp': df_5m['timestamp'].iloc[-1],
                'inst': SYMBOL_FUTURES,
                'price': price,
                'type': 'ОЖИДАНИЕ',
                'level': lvl,
                'comment': 'Рядом поддержка. Ждём вынос (свип) и отказ для лонга.',
                'atr_1h': atr_1h,
            })
        return rows, attention

    # Rejection exists but no suitable level on the right side -> stand aside
    status_ru = "НЕТ СДЕЛКИ"
    reason_ru = "НЕТ БЛИЖАЙШЕГО УРОВНЯ ДЛЯ РАБОТЫ (уровни далеко/непонятно)"
    direction = None
    rows.append({
        'timestamp': df_5m['timestamp'].iloc[-1],
        'inst': SYMBOL_FUTURES,
        'price': price,
        'status': status_ru,
        'reason': reason_ru,
        'direction': direction,
        'level': np.nan,
        'atr_1h': atr_1h,
        'who_punished': _who_gets_punished(status_ru, reason_ru, direction),
    })
    return rows, attention


def assess_setups_rus(df_5m: pd.DataFrame, df_1h: pd.DataFrame, key_levels) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Обёртка: возвращает + задания для визуализации."""
    setups_rows, attention_rows = assess_setups(df_5m=df_5m, df_1h=df_1h, key_levels=key_levels)

    viz_tasks: List[Dict] = []
    if setups_rows:
        last = setups_rows[-1]
        viz_tasks.append({
            'df_5m': df_5m,
            'level': last.get('level', None),
            'status': str(last.get('status', '')),
            'reason': str(last.get('reason', '')),
            'inst': str(last.get('inst', SYMBOL_FUTURES)),
        })
    return setups_rows, attention_rows, viz_tasks


def save_viz(task: Dict, date_str: str):
    """Сохраняет визуализацию в архив."""
    archive_dir = get_archive_path(date_str)
    os.makedirs(archive_dir, exist_ok=True)
    day = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime('%d')
    viz_dir = os.path.join(archive_dir, BASE_FILES['viz'])
    os.makedirs(viz_dir, exist_ok=True)

    inst = task.get('inst', SYMBOL_FUTURES)
    status = str(task.get('status', ''))
    reason = str(task.get('reason', ''))
    level = task.get('level', None)

    safe_inst = str(inst).replace('/', '-').replace(':', '-')
    file_path = os.path.join(viz_dir, f"{day}_{safe_inst}.png")
    make_viz(task['df_5m'], level=level, status=status, reason=reason, out_path=file_path)
    return file_path


def make_viz(df_5m: pd.DataFrame, level: Optional[float], status: str, reason: str, out_path: str):
    """Сохраняет PNG с последними свечами M5 и пояснением."""
    if df_5m.empty:
        return

    d = df_5m.tail(120).copy()  # последние ~10 часов
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    x = np.arange(len(d))
    # Простейший свечной график (без внешних библиотек)
    for i, row in enumerate(d.itertuples(index=False)):
        o = float(getattr(row, 'open'))
        h = float(getattr(row, 'high'))
        l = float(getattr(row, 'low'))
        c = float(getattr(row, 'close'))
        ax.vlines(i, l, h, linewidth=1)
        # тело
        y0 = min(o, c)
        y1 = max(o, c)
        ax.vlines(i, y0, y1, linewidth=4)

    if level is not None and not (isinstance(level, float) and np.isnan(level)):
        ax.axhline(level, linestyle='--', linewidth=1)
        ax.text(0.01, 0.98, f"Уровень: {level:.0f}", transform=ax.transAxes, va='top')

    ax.set_title(f"BTC M5 | {status} | {reason}")
    ax.set_xlabel("Последние свечи (M5)")
    ax.set_ylabel("Цена")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def get_reserve_move(df_1h, df_1d, atr):
    if df_1h.empty or df_1d.empty or atr is None or np.isnan(atr):
        return "Нет данных для оценки запаса хода"
    price_move = abs(df_1h['close'].iloc[-1] - df_1d['close'].iloc[-1])
    if price_move > atr * 0.75:
        return "Запас хода исчерпан (>75% ATR), предпочтение контртрендовым сделкам"
    return "Запас хода нормальный"

def main():
    try:
        # Часовой пояс Алматы (официальный IANA)
        # Важно: фиксированные Etc/GMT-* легко дают смещение, поэтому используем Asia/Almaty.
        almaty_tz = pytz.timezone('Asia/Almaty')
        almaty_time = datetime.now(almaty_tz)
        date_str = almaty_time.strftime('%Y-%m-%d %H:%M:%S')
        logging.info("Начало работы скрипта: %s", date_str)

        # 5m is required for strict filters (impulse / rejection / retest)
        df_5m = fetch_ohlcv(SYMBOL_FUTURES, "5m", limit=200)
        df_1h = fetch_ohlcv(SYMBOL_FUTURES, TIMEFRAMES["1h"])
        df_1d = fetch_ohlcv(SYMBOL_FUTURES, TIMEFRAMES["1d"])

        if df_5m.empty or df_1h.empty or df_1d.empty:
            logging.warning("OHLCV данные пусты. Прерывание.")
            return

        df_1h = calculate_atr(df_1h)
        df_1d = calculate_atr(df_1d)

        local_trend, global_trend = determine_trend(df_1h, df_1d)
        key_levels = find_key_levels(df_1h)
        round_level = find_round_levels(df_1h['close'].iloc[-1])
        patterns = check_patterns(df_1h, key_levels)
        atr_daily = df_1d['atr'].iloc[-1] if 'atr' in df_1d.columns else np.nan
        reserve_status = get_reserve_move(df_1h, df_1d, atr_daily)


        # --- Smart trade filters (НЕТ СДЕЛКИ / НЕДЕЙСТВИТЕЛЬНО / СДЕЛКА) ---
        setups_rows, attention_rows, viz_tasks = assess_setups_rus(df_5m=df_5m, df_1h=df_1h, key_levels=key_levels)
        df_setups = pd.DataFrame(setups_rows)
        df_attention = pd.DataFrame(attention_rows)

        # Сохраняем
        if key_levels:
            df_levels = pd.DataFrame(key_levels, columns=["type", "price", "timestamp"])
            save_to_archive(df_levels, BASE_FILES["levels"], date_str)
        if patterns:
            df_patterns = pd.DataFrame(patterns, columns=["pattern", "level_price", "timestamp"])
            save_to_archive(df_patterns, BASE_FILES["patterns"], date_str)

        if not df_setups.empty:
            save_to_archive(df_setups, BASE_FILES["setups"], date_str, header=True)

        if not df_attention.empty:
            save_to_archive(df_attention, BASE_FILES["attention"], date_str, header=True)

        # Визуальные пояснения (скрин/схема по последним свечам)
        for task in viz_tasks:
            try:
                save_viz(task=task, date_str=date_str)
            except Exception as e:
                logging.warning("Не удалось сохранить визуализацию: %s", e)

        save_to_archive(df_1h, BASE_FILES["ohlcv_1h"], date_str, header=True)
        save_to_archive(df_1d, BASE_FILES["ohlcv_1d"], date_str, header=True)

        # Лог для Actions
        logging.info("Локальный тренд: %s, Глобальный тренд: %s", local_trend, global_trend)
        logging.info("Ключевые уровни (последние 5): %s", key_levels[-5:] if key_levels else "нет")
        logging.info("Круглый уровень: %s", round_level)
        logging.info("Найденные паттерны: %s", patterns)
        logging.info("Запас хода: %s", reserve_status)
    except Exception as e:
        logging.exception("Ошибка в main: %s", e)


if __name__ == "__main__":
    main()
