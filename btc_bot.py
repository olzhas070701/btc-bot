import requests
import pandas as pd
import os
from datetime import datetime
import pytz
import numpy as np
import logging
import sys

# Настройки
BASE_FILES = {
    "levels": "btc_levels_data",
    "patterns": "btc_patterns_data",
    "ohlcv_1h": "btc_ohlcv_1h",
    "ohlcv_1d": "btc_ohlcv_1d"
}
SYMBOL_FUTURES = "BTC-USD-SWAP"
TIMEFRAMES = {"1h": "1H", "1d": "1D"}
ATR_PERIOD = 14
ROUND_LEVELS = [1000, 500, 100]
LIMIT = 100
ARCHIVE_ROOT = "archive"

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
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst_id}&bar={bar}&limit={limit}"
    resp = requests.get(url, timeout=15)
    data_json = resp.json()
    if data_json.get("code") not in (None, "0", 0):
        raise ValueError(f"API error: {data_json.get('msg') or data_json}")
    data = data_json.get("data", [])
    # OKX возвращает массивы; приведём к DF
    df = pd.DataFrame(data)
    # Если пришёл пустой ответ
    if df.empty:
        return df
    # Назначаем колонки в соответствии с OKX (если их меньше — обработаем осторожно)
    cols = ["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"]
    # Если колонок меньше, дополним
    if df.shape[1] < len(cols):
        # просто переименуем существующие столбцы
        cols = cols[:df.shape[1]]
    df.columns = cols
    # Приводим типы
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # timestamp в ms -> datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df[::-1].reset_index(drop=True)
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
    signals = []
    if df.shape[0] < 2:
        return signals
    last_bar = df.iloc[-1]
    prev_bar = df.iloc[-2]
    for level_type, level_price, _ in levels:
        # false breakout
        if (prev_bar['high'] > level_price and last_bar['close'] < level_price and level_type == 'resistance') or \
           (prev_bar['low'] < level_price and last_bar['close'] > level_price and level_type == 'support'):
            signals.append(('false_breakout', level_price, last_bar['timestamp']))
        # breakout
        if last_bar['close'] > level_price and prev_bar['close'] < level_price and level_type == 'resistance':
            signals.append(('breakout', level_price, last_bar['timestamp']))
        # bounce
        atr_val = last_bar.get('atr', 0) if 'atr' in last_bar.index else 0
        if level_type == 'support' and not np.isnan(atr_val) and abs(last_bar['low'] - level_price) < atr_val * 0.1 and last_bar['close'] > level_price:
            signals.append(('bounce', level_price, last_bar['timestamp']))
    return signals

def get_reserve_move(df_1h, df_1d, atr):
    if df_1h.empty or df_1d.empty or atr is None or np.isnan(atr):
        return "Нет данных для оценки запаса хода"
    price_move = abs(df_1h['close'].iloc[-1] - df_1d['close'].iloc[-1])
    if price_move > atr * 0.75:
        return "Запас хода исчерпан (>75% ATR), предпочтение контртрендовым сделкам"
    return "Запас хода нормальный"

def main():
    try:
        almaty_tz = pytz.timezone('Asia/Almaty')
        almaty_time = datetime.now(almaty_tz)
        date_str = almaty_time.strftime('%Y-%m-%d %H:%M:%S')
        logging.info("Начало работы скрипта: %s", date_str)

        df_1h = fetch_ohlcv(SYMBOL_FUTURES, TIMEFRAMES["1h"])
        df_1d = fetch_ohlcv(SYMBOL_FUTURES, TIMEFRAMES["1d"])

        if df_1h.empty or df_1d.empty:
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

        # Сохраняем
        if key_levels:
            df_levels = pd.DataFrame(key_levels, columns=["type", "price", "timestamp"])
            save_to_archive(df_levels, BASE_FILES["levels"], date_str)
        if patterns:
            df_patterns = pd.DataFrame(patterns, columns=["pattern", "level_price", "timestamp"])
            save_to_archive(df_patterns, BASE_FILES["patterns"], date_str)

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
