import os
import math
import time
import json
import csv
import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional

import requests
import pandas as pd

# =========================
# CONFIG
# =========================
OKX_BASE = "https://www.okx.com"
INST_ID = "BTC-USDT-SWAP"          # OKX Perpetual Swap
TZ = "Asia/Almaty"

# Data windows
FETCH_LIMIT_5M = 600              # ~50h on 5m
FETCH_LIMIT_1H = 400              # ~16d on 1h
FETCH_LIMIT_1D = 120              # ~4 months

ATR_PERIOD = 14

# Impulse requirements (Kravchenko style: сначала импульс)
IMPULSE_ATR_MULT = 0.8            # impulse if move >= 0.8*ATR(H1)
IMPULSE_LOOKBACK_5M = 12          # 12x5m = 60 minutes

# Sweep / Retest logic (fractions of ATR(H1))
SWEEP_ATR_FRAC = 0.15             # sweep depth threshold
RETEST_ATR_FRAC = 0.10            # retest zone thickness around level
LEVEL_CLUSTER_ATR_FRAC = 0.12     # merge near levels within 0.12*ATR
RETEST_WINDOW_5M = 12             # must retest within next 60 minutes

# NO_TRADE: "middle of range"
MID_NO_TRADE_ATR_FRAC = 0.35

# RR target (1k1 = 1:1). If you want 2k1 -> set to 2.0
RR_TARGET = float(os.getenv("RR_TARGET", "1.0"))

ARCHIVE_DIR = "archive"

HTTP_TIMEOUT = 12
RETRIES = 4


# =========================
# TIME / IO HELPERS
# =========================
def _now_almaty() -> pd.Timestamp:
    # ✅ tz-aware now; no tz_localize
    return pd.Timestamp.now(tz="UTC").tz_convert(TZ)


def _safe_request(url: str, params: Dict) -> Dict:
    last_err = None
    for i in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            js = r.json()
            return js
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (2 ** i))
    raise RuntimeError(f"Request failed after retries: {last_err}")


def _okx_get_candles(inst_id: str, bar: str, limit: int, use_history: bool) -> pd.DataFrame:
    """
    OKX endpoints:
      - /api/v5/market/candles         (более "живой", быстрее обновляется)
      - /api/v5/market/history-candles (иногда отстаёт, но для бэка)
    """
    path = "/api/v5/market/history-candles" if use_history else "/api/v5/market/candles"
    url = f"{OKX_BASE}{path}"

    js = _safe_request(url, {"instId": inst_id, "bar": bar, "limit": str(limit)})
    if js.get("code") != "0":
        raise RuntimeError(f"OKX error: {js}")

    data = js.get("data", [])
    if not data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    rows = []
    for item in data:
        # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        ts = int(item[0])
        o, h, l, c = map(float, item[1:5])
        vol = float(item[5])
        rows.append([ts, o, h, l, c, vol])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["time_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["time_almaty"] = df["time_utc"].dt.tz_convert(TZ)
    return df


def fetch_okx_candles(inst_id: str, bar: str, limit: int) -> pd.DataFrame:
    """
    ✅ Сначала берём "candles" (самые свежие),
    ✅ если пусто/ошибка — fallback на history-candles.
    """
    try:
        df = _okx_get_candles(inst_id, bar, limit, use_history=False)
        if len(df) > 0:
            return df
    except Exception:
        pass
    return _okx_get_candles(inst_id, bar, limit, use_history=True)


def upsert_csv(path: str, df: pd.DataFrame, key_col: str = "timestamp") -> None:
    """
    Merge with existing CSV: no duplicates, keep sorted.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df2 = df.copy()
    df2[key_col] = df2[key_col].astype("int64")

    if os.path.exists(path):
        old = pd.read_csv(path)
        if key_col in old.columns:
            old[key_col] = old[key_col].astype("int64")
        merged = pd.concat([old, df2], ignore_index=True)
        merged = merged.drop_duplicates(subset=[key_col], keep="last")
        merged = merged.sort_values(key_col)
        merged.to_csv(path, index=False)
    else:
        df2.to_csv(path, index=False)


# =========================
# INDICATORS / STRATEGY HELPERS
# =========================
def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def is_rejection_candle(o, h, l, c, side: str) -> bool:
    """
    SHORT: long upper wick + weak close (close <= mid)
    LONG : long lower wick + strong close (close >= mid)
    """
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    mid = (h + l) / 2

    # wick candle preferred
    if body / rng > 0.55:
        return False

    if side == "short":
        return (upper_wick / rng >= 0.45) and (c <= mid)
    else:
        return (lower_wick / rng >= 0.45) and (c >= mid)


def fractal_levels_h1(df_h1: pd.DataFrame, left: int = 2, right: int = 2) -> List[float]:
    highs = df_h1["high"].values
    lows = df_h1["low"].values

    levels: List[float] = []
    n = len(df_h1)

    for i in range(left, n - right):
        h = highs[i]
        if all(h > highs[i - j] for j in range(1, left + 1)) and all(h >= highs[i + j] for j in range(1, right + 1)):
            levels.append(float(h))

        lo = lows[i]
        if all(lo < lows[i - j] for j in range(1, left + 1)) and all(lo <= lows[i + j] for j in range(1, right + 1)):
            levels.append(float(lo))

    return levels


def cluster_levels(levels: List[float], atr: float, cluster_frac: float) -> List[float]:
    if not levels:
        return []
    tol = max(atr * cluster_frac, 1.0)

    levels_sorted = sorted(levels)
    clusters = [[levels_sorted[0]]]
    for x in levels_sorted[1:]:
        if abs(x - clusters[-1][-1]) <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    return [float(pd.Series(c).median()) for c in clusters]


def daily_levels(df_d1: pd.DataFrame) -> Dict[str, float]:
    """
    Yesterday OHLC -> simple pivots + yesterday high/low
    """
    if len(df_d1) < 2:
        return {}
    y = df_d1.iloc[-2]
    H, L, C = float(y["high"]), float(y["low"]), float(y["close"])
    P = (H + L + C) / 3.0
    R1 = 2 * P - L
    S1 = 2 * P - H

    return {
        "Y_HIGH": H,
        "Y_LOW": L,
        "Y_CLOSE": C,
        "PIVOT": float(P),
        "R1": float(R1),
        "S1": float(S1),
    }


def nearest_levels(levels: List[float], price: float, k: int = 8) -> List[float]:
    return sorted(levels, key=lambda x: abs(x - price))[:k]


@dataclass
class SetupResult:
    status: str  # VALID / INVALID / NO_TRADE
    side: str    # long/short/none
    reason: str
    level: Optional[float] = None
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr: Optional[float] = None
    atr_h1: Optional[float] = None


def evaluate_kravchenko_1k1(df_5m: pd.DataFrame, df_h1: pd.DataFrame, df_d1: pd.DataFrame) -> SetupResult:
    """
    Kravchenko 1k1:
      - NO_TRADE: нет импульса / середина диапазона / нет модели у уровня
      - INVALID: цена "улетела без ретеста" после sweep+rejection
      - VALID: sweep + rejection + retest к уровню
    """
    if len(df_5m) < 80 or len(df_h1) < 80 or len(df_d1) < 3:
        return SetupResult("NO_TRADE", "none", "not enough data")

    atr_series = wilder_atr(df_h1, ATR_PERIOD)
    atr = float(atr_series.iloc[-1])
    if not math.isfinite(atr) or atr <= 0:
        return SetupResult("NO_TRADE", "none", "atr not available")

    price = float(df_5m["close"].iloc[-1])

    # levels
    dlevels = daily_levels(df_d1)
    h1_fr = fractal_levels_h1(df_h1.tail(3 * 24))  # ~3 days
    all_levels = list(dlevels.values()) + h1_fr
    all_levels = cluster_levels(all_levels, atr, LEVEL_CLUSTER_ATR_FRAC)

    if not all_levels:
        return SetupResult("NO_TRADE", "none", "no levels", atr_h1=atr)

    near = nearest_levels(all_levels, price, k=8)
    nearest_dist = min(abs(price - lv) for lv in near)

    # NO_TRADE: middle
    if nearest_dist > atr * MID_NO_TRADE_ATR_FRAC:
        return SetupResult(
            "NO_TRADE", "none",
            f"middle of range: dist_to_level={nearest_dist:.0f} > {MID_NO_TRADE_ATR_FRAC:.2f}*ATR",
            atr_h1=atr
        )

    # Impulse check (last 60 minutes)
    w = df_5m.tail(IMPULSE_LOOKBACK_5M)
    impulse_move = float(w["high"].max() - w["low"].min())
    if impulse_move < atr * IMPULSE_ATR_MULT:
        return SetupResult(
            "NO_TRADE", "none",
            f"no impulse: move={impulse_move:.0f} < {IMPULSE_ATR_MULT:.2f}*ATR",
            atr_h1=atr
        )

    retest_tol = atr * RETEST_ATR_FRAC
    sweep_req = atr * SWEEP_ATR_FRAC

    scan = df_5m.tail(40).reset_index(drop=True)

    best: Optional[SetupResult] = None

    for lv in near:
        for i in range(0, len(scan) - 2):
            row = scan.iloc[i]
            o, h, l, c = map(float, [row["open"], row["high"], row["low"], row["close"]])

            # ---------- SHORT ----------
            if (h >= lv + sweep_req) and (c <= lv) and is_rejection_candle(o, h, l, c, "short"):
                post = scan.iloc[i + 1: i + 1 + RETEST_WINDOW_5M]
                hit = post[(post["high"] >= lv - retest_tol) & (post["low"] <= lv + retest_tol)]
                if hit.empty:
                    cand = SetupResult(
                        "INVALID", "short", "no retest after sweep (price flew away)",
                        level=float(lv), atr_h1=atr
                    )
                else:
                    sweep_high = float(h)
                    entry = float(lv)
                    sl = sweep_high + 0.10 * atr
                    risk = sl - entry
                    if risk <= 0:
                        continue
                    tp = entry - RR_TARGET * risk
                    cand = SetupResult(
                        "VALID", "short", "sweep+rejection+retest",
                        level=float(lv), entry=entry, sl=float(sl), tp=float(tp),
                        rr=float(RR_TARGET), atr_h1=atr
                    )

                best = _pick_better(best, cand, price)

            # ---------- LONG ----------
            if (l <= lv - sweep_req) and (c >= lv) and is_rejection_candle(o, h, l, c, "long"):
                post = scan.iloc[i + 1: i + 1 + RETEST_WINDOW_5M]
                hit = post[(post["high"] >= lv - retest_tol) & (post["low"] <= lv + retest_tol)]
                if hit.empty:
                    cand = SetupResult(
                        "INVALID", "long", "no retest after sweep (price flew away)",
                        level=float(lv), atr_h1=atr
                    )
                else:
                    sweep_low = float(l)
                    entry = float(lv)
                    sl = sweep_low - 0.10 * atr
                    risk = entry - sl
                    if risk <= 0:
                        continue
                    tp = entry + RR_TARGET * risk
                    cand = SetupResult(
                        "VALID", "long", "sweep+rejection+retest",
                        level=float(lv), entry=entry, sl=float(sl), tp=float(tp),
                        rr=float(RR_TARGET), atr_h1=atr
                    )

                best = _pick_better(best, cand, price)

    if best is None:
        return SetupResult("NO_TRADE", "none", "no sweep/rejection near levels", atr_h1=atr)

    return best


def _pick_better(best: Optional[SetupResult], cand: SetupResult, price: float) -> SetupResult:
    if best is None:
        return cand
    # prefer VALID over INVALID
    if best.status != "VALID" and cand.status == "VALID":
        return cand
    if best.status == cand.status:
        # prefer closer to current price
        if best.level is None:
            return cand
        if cand.level is None:
            return best
        if abs(price - cand.level) < abs(price - best.level):
            return cand
    return best


# =========================
# ARCHIVE STRUCTURE
# =========================
def build_paths(now: pd.Timestamp) -> Dict[str, str]:
    """
    archive/YYYY/MM/DD/
      - btc_5m.csv, btc_1h.csv, btc_1d.csv, setups.csv, errors.log
    archive/YYYY/MM/runs/YYYY-MM-DD_HHMM/
      - same files for each run (optional but удобно)
    """
    y = now.strftime("%Y")
    m = now.strftime("%m")
    d = now.strftime("%d")
    run_tag = now.strftime("%Y-%m-%d_%H%M")

    day_dir = os.path.join(ARCHIVE_DIR, y, m, d)
    runs_dir = os.path.join(ARCHIVE_DIR, y, m, "runs", run_tag)

    return {
        "day_dir": day_dir,
        "runs_dir": runs_dir,
        "day_5m": os.path.join(day_dir, "btc_5m.csv"),
        "day_1h": os.path.join(day_dir, "btc_1h.csv"),
        "day_1d": os.path.join(day_dir, "btc_1d.csv"),
        "day_setups": os.path.join(day_dir, "setups.csv"),
        "day_errors": os.path.join(day_dir, "errors.log"),

        "run_5m": os.path.join(runs_dir, "btc_5m.csv"),
        "run_1h": os.path.join(runs_dir, "btc_1h.csv"),
        "run_1d": os.path.join(runs_dir, "btc_1d.csv"),
        "run_setups": os.path.join(runs_dir, "setups.csv"),
        "run_errors": os.path.join(runs_dir, "errors.log"),
    }


def append_setup(path: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def main():
    now = _now_almaty()
    paths = build_paths(now)

    try:
        df5 = fetch_okx_candles(INST_ID, "5m", FETCH_LIMIT_5M)
        dfh = fetch_okx_candles(INST_ID, "1H", FETCH_LIMIT_1H)
        dfd = fetch_okx_candles(INST_ID, "1D", FETCH_LIMIT_1D)

        cols = ["timestamp", "time_almaty", "open", "high", "low", "close", "volume"]
        df5_out = df5[cols]
        dfh_out = dfh[cols]
        dfd_out = dfd[cols]

        # ✅ daily (grows during the day; no dupes)
        upsert_csv(paths["day_5m"], df5_out)
        upsert_csv(paths["day_1h"], dfh_out)
        upsert_csv(paths["day_1d"], dfd_out)

        # ✅ per-run snapshots (optional but useful)
        upsert_csv(paths["run_5m"], df5_out)
        upsert_csv(paths["run_1h"], dfh_out)
        upsert_csv(paths["run_1d"], dfd_out)

        # Strategy eval
        res = evaluate_kravchenko_1k1(df5, dfh, dfd)

        setup_row = {
            "time_almaty": str(now),
            "instId": INST_ID,
            "price": float(df5["close"].iloc[-1]) if len(df5) else None,
            "status": res.status,
            "side": res.side,
            "reason": res.reason,
            "level": res.level,
            "entry": res.entry,
            "sl": res.sl,
            "tp": res.tp,
            "rr": res.rr,
            "atr_h1": res.atr_h1,
        }

        append_setup(paths["day_setups"], setup_row)
        append_setup(paths["run_setups"], setup_row)

        print(json.dumps(setup_row, ensure_ascii=False))

    except Exception as e:
        err = f"\n[{now}] ERROR: {repr(e)}\n{traceback.format_exc()}\n"
        os.makedirs(os.path.dirname(paths["day_errors"]), exist_ok=True)
        with open(paths["day_errors"], "a", encoding="utf-8") as f:
            f.write(err)
        os.makedirs(os.path.dirname(paths["run_errors"]), exist_ok=True)
        with open(paths["run_errors"], "a", encoding="utf-8") as f:
            f.write(err)
        raise


if __name__ == "__main__":
    main()
