import os
import math
import time
import json
import csv
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd

# =========================
# CONFIG
# =========================
OKX_BASE = "https://www.okx.com"
INST_ID = "BTC-USDT-SWAP"  # ✅ Correct Perpetual Swap
TZ = "Asia/Almaty"

# Data cadence / windows
FETCH_LIMIT_5M = 300        # last ~25h on 5m
FETCH_LIMIT_1H = 240        # last 10d on 1h (enough for ATR + levels)
FETCH_LIMIT_1D = 60         # last 2 months

ATR_PERIOD = 14
IMPULSE_ATR_MULT = 0.8      # impulse if move >= 0.8*ATR(H1) over lookback
IMPULSE_LOOKBACK_5M = 12    # 12x5m = 60 minutes

# Level / retest tolerances (as fraction of ATR(H1))
SWEEP_ATR_FRAC = 0.15       # must pierce level by >= 0.15*ATR to count as sweep
RETEST_ATR_FRAC = 0.10      # retest zone thickness around level
LEVEL_CLUSTER_ATR_FRAC = 0.12  # merge nearby levels within 0.12*ATR

# Retest window: if no retest -> INVALID
RETEST_WINDOW_5M = 12       # 12 candles = 60 min

# "Middle of range" no-trade
MID_NO_TRADE_ATR_FRAC = 0.35

# Output
ARCHIVE_DIR = "archive"
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Requests
HTTP_TIMEOUT = 12
RETRIES = 4

# =========================
# HELPERS
# =========================

def _now_almaty() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(TZ)

def _safe_request(url: str, params: Dict) -> Dict:
    last_err = None
    for i in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (2 ** i))
    raise RuntimeError(f"Request failed after retries: {last_err}")

def fetch_okx_candles(inst_id: str, bar: str, limit: int) -> pd.DataFrame:
    """
    OKX candles:
    GET /api/v5/market/history-candles?instId=...&bar=...&limit=...
    returns newest->oldest; we reverse to chronological.
    """
    url = f"{OKX_BASE}/api/v5/market/history-candles"
    js = _safe_request(url, {"instId": inst_id, "bar": bar, "limit": str(limit)})
    if js.get("code") != "0":
        raise RuntimeError(f"OKX error: {js}")
    data = js.get("data", [])
    if not data:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    rows = []
    for item in data:
        # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        ts = int(item[0])
        o, h, l, c = map(float, item[1:5])
        vol = float(item[5])
        rows.append([ts, o, h, l, c, vol])
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # ✅ Proper timezone conversion to Asia/Almaty
    df["time_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["time_almaty"] = df["time_utc"].dt.tz_convert(TZ)
    return df

def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder ATR on OHLC dataframe (chronological).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def is_rejection_candle(o, h, l, c, side: str) -> bool:
    """
    Simple rejection logic:
    SHORT: long upper wick + weak close (close <= mid)
    LONG:  long lower wick + strong close (close >= mid)
    """
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    mid = (h + l) / 2

    # body should not be huge (we prefer wick signal)
    if body / rng > 0.55:
        return False

    if side == "short":
        return (upper_wick / rng >= 0.45) and (c <= mid)
    else:
        return (lower_wick / rng >= 0.45) and (c >= mid)

def fractal_levels_h1(df_h1: pd.DataFrame, left: int = 2, right: int = 2) -> List[float]:
    """
    Build "nearest left" levels using simple fractals on H1.
    """
    highs = df_h1["high"].values
    lows = df_h1["low"].values
    levels = []

    n = len(df_h1)
    for i in range(left, n - right):
        h = highs[i]
        if all(h > highs[i-j] for j in range(1, left+1)) and all(h >= highs[i+j] for j in range(1, right+1)):
            levels.append(float(h))
        lo = lows[i]
        if all(lo < lows[i-j] for j in range(1, left+1)) and all(lo <= lows[i+j] for j in range(1, right+1)):
            levels.append(float(lo))

    return levels

def cluster_levels(levels: List[float], atr: float, cluster_frac: float) -> List[float]:
    """
    Merge close levels into clusters; keep cluster median.
    """
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
    clustered = [float(pd.Series(c).median()) for c in clusters]
    return clustered

def daily_levels(df_d1: pd.DataFrame) -> Dict[str, float]:
    """
    Yesterday OHLC -> pivot levels
    """
    if len(df_d1) < 2:
        return {}
    y = df_d1.iloc[-2]
    H, L, C = y["high"], y["low"], y["close"]
    P = (H + L + C) / 3.0
    R1 = 2*P - L
    S1 = 2*P - H
    return {
        "Y_HIGH": float(H),
        "Y_LOW": float(L),
        "Y_CLOSE": float(C),
        "PIVOT": float(P),
        "R1": float(R1),
        "S1": float(S1),
    }

def nearest_levels(levels: List[float], price: float, k: int = 6) -> List[float]:
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

def evaluate_1k1_setups(df_5m: pd.DataFrame, df_h1: pd.DataFrame, df_d1: pd.DataFrame) -> SetupResult:
    """
    Core "Kravchenko 1k1" evaluator:
    - must have impulse
    - must be near a key level (daily or nearest-left)
    - must have sweep + rejection
    - must have retest; if no retest -> INVALID
    """
    if len(df_5m) < 50 or len(df_h1) < 50 or len(df_d1) < 3:
        return SetupResult("NO_TRADE", "none", "not enough data")

    # ATR on H1
    atr_series = wilder_atr(df_h1, ATR_PERIOD)
    atr = float(atr_series.iloc[-1])
    if not math.isfinite(atr) or atr <= 0:
        return SetupResult("NO_TRADE", "none", "atr not available")

    price = float(df_5m["close"].iloc[-1])

    # Build levels: daily + fractals
    dlevels = daily_levels(df_d1)
    h1_fr = fractal_levels_h1(df_h1.tail(3*24))  # last ~3 days
    all_levels = list(dlevels.values()) + h1_fr
    all_levels = cluster_levels(all_levels, atr, LEVEL_CLUSTER_ATR_FRAC)

    if not all_levels:
        return SetupResult("NO_TRADE", "none", "no levels")

    near = nearest_levels(all_levels, price, k=8)
    nearest_dist = min(abs(price - lv) for lv in near)

    # NO_TRADE: middle of range (far from levels)
    if nearest_dist > atr * MID_NO_TRADE_ATR_FRAC:
        return SetupResult("NO_TRADE", "none", f"middle of range: dist_to_level={nearest_dist:.0f} > {MID_NO_TRADE_ATR_FRAC:.2f}*ATR")

    # Impulse check on 5m: last 60 minutes move
    w = df_5m.tail(IMPULSE_LOOKBACK_5M)
    impulse_move = float(w["high"].max() - w["low"].min())
    if impulse_move < atr * IMPULSE_ATR_MULT:
        return SetupResult("NO_TRADE", "none", f"no impulse: move={impulse_move:.0f} < {IMPULSE_ATR_MULT:.2f}*ATR")

    # Find best candidate level among nearest
    # We will check sweep + rejection on last few candles
    retest_tol = atr * RETEST_ATR_FRAC
    sweep_req = atr * SWEEP_ATR_FRAC

    last = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]
    # We examine last N candles for a sweep event
    scan = df_5m.tail(30).reset_index(drop=True)

    best: Optional[SetupResult] = None

    for lv in near:
        # Sweep up (short) if candle high pierced level + sweep_req and close back below lv
        # Sweep down (long) if candle low pierced below level - sweep_req and close back above lv
        for i in range(len(scan)-RETEST_WINDOW_5M, len(scan)):  # check recent region
            row = scan.iloc[i]
            o,h,l,c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            # SHORT sweep
            if (h >= lv + sweep_req) and (c <= lv):
                if not is_rejection_candle(o,h,l,c,"short"):
                    continue
                # retest must occur after i within window
                post = scan.iloc[i+1: i+1+RETEST_WINDOW_5M]
                if post.empty:
                    continue
                # retest: price returns to within retest_tol of level
                hit = post[(post["high"] >= lv - retest_tol) & (post["low"] <= lv + retest_tol)]
                if hit.empty:
                    # "flew away without retest"
                    cand = SetupResult("INVALID","short","no retest after sweep", level=lv)
                else:
                    # entry at level (limit), SL above sweep high + buffer, TP 1R
                    sweep_high = float(row["high"])
                    entry = float(lv)
                    sl = sweep_high + (0.10 * atr)
                    risk = sl - entry
                    if risk <= 0:
                        continue
                    tp = entry - risk  # 1:1
                    rr = 1.0
                    cand = SetupResult("VALID","short","sweep+rejection+retest", level=lv, entry=entry, sl=sl, tp=tp, rr=rr)

                # choose the closest/cleanest
                if best is None:
                    best = cand
                else:
                    # prefer VALID over INVALID, and nearer level to price
                    if best.status != "VALID" and cand.status == "VALID":
                        best = cand
                    elif best.status == cand.status:
                        if abs(price - lv) < abs(price - (best.level or lv)):
                            best = cand

            # LONG sweep
            if (l <= lv - sweep_req) and (c >= lv):
                if not is_rejection_candle(o,h,l,c,"long"):
                    continue
                post = scan.iloc[i+1: i+1+RETEST_WINDOW_5M]
                if post.empty:
                    continue
                hit = post[(post["high"] >= lv - retest_tol) & (post["low"] <= lv + retest_tol)]
                if hit.empty:
                    cand = SetupResult("INVALID","long","no retest after sweep", level=lv)
                else:
                    sweep_low = float(row["low"])
                    entry = float(lv)
                    sl = sweep_low - (0.10 * atr)
                    risk = entry - sl
                    if risk <= 0:
                        continue
                    tp = entry + risk  # 1:1
                    rr = 1.0
                    cand = SetupResult("VALID","long","sweep+rejection+retest", level=lv, entry=entry, sl=sl, tp=tp, rr=rr)

                if best is None:
                    best = cand
                else:
                    if best.status != "VALID" and cand.status == "VALID":
                        best = cand
                    elif best.status == cand.status:
                        if abs(price - lv) < abs(price - (best.level or lv)):
                            best = cand

    if best is None:
        # if there was impulse but no rejection pattern near level
        return SetupResult("NO_TRADE", "none", "no valid sweep/rejection near levels")

    return best

def upsert_csv(path: str, df: pd.DataFrame, key_col: str = "timestamp") -> None:
    """
    Merge with existing csv to avoid duplicates and keep sorted.
    """
    if os.path.exists(path):
        old = pd.read_csv(path)
        # if timestamp types differ
        if key_col in old.columns:
            old[key_col] = old[key_col].astype("int64")
        df2 = df.copy()
        df2[key_col] = df2[key_col].astype("int64")
        merged = pd.concat([old, df2], ignore_index=True)
        merged = merged.drop_duplicates(subset=[key_col], keep="last")
        merged = merged.sort_values(key_col)
        merged.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)

def main():
    now = _now_almaty()
    day_tag = now.strftime("%Y-%m-%d")

    try:
        df5 = fetch_okx_candles(INST_ID, "5m", FETCH_LIMIT_5M)
        dfh = fetch_okx_candles(INST_ID, "1H", FETCH_LIMIT_1H)
        dfd = fetch_okx_candles(INST_ID, "1D", FETCH_LIMIT_1D)

        # Save data (no duplicates)
        upsert_csv(os.path.join(ARCHIVE_DIR, f"{day_tag}_btc_5m.csv"), df5[["timestamp","time_almaty","open","high","low","close","volume"]])
        upsert_csv(os.path.join(ARCHIVE_DIR, f"{day_tag}_btc_1h.csv"), dfh[["timestamp","time_almaty","open","high","low","close","volume"]])
        upsert_csv(os.path.join(ARCHIVE_DIR, f"{day_tag}_btc_1d.csv"), dfd[["timestamp","time_almaty","open","high","low","close","volume"]])

        # Evaluate strategy
        res = evaluate_1k1_setups(df5, dfh, dfd)

        # Log setups
        setup_row = {
            "time_almaty": str(now),
            "instId": INST_ID,
            "price": float(df5["close"].iloc[-1]),
            "status": res.status,
            "side": res.side,
            "reason": res.reason,
            "level": res.level,
            "entry": res.entry,
            "sl": res.sl,
            "tp": res.tp,
            "rr": res.rr,
        }
        setups_path = os.path.join(ARCHIVE_DIR, f"{day_tag}_btc_setups.csv")
        # append
        file_exists = os.path.exists(setups_path)
        with open(setups_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(setup_row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(setup_row)

        print(json.dumps(setup_row, ensure_ascii=False))

    except Exception as e:
        err_path = os.path.join(ARCHIVE_DIR, f"{day_tag}_errors.log")
        with open(err_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{now}] ERROR: {repr(e)}\n")
            f.write(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
