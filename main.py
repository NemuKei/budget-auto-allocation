import csv
import configparser
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import jpholiday

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

INPUT_DIR = Path('input')
OUTPUT_DIR = Path('output')
CONFIG_FILE = Path('settings.ini')

@dataclass
class Settings:
    fiscal_year: int
    capacity: int
    breakfast_price: int
    weight_prev_year: float
    weight_2_3m: float
    weight_recent: float

def load_settings() -> Settings:
    cp = configparser.ConfigParser()
    if CONFIG_FILE.exists():
        cp.read(CONFIG_FILE)
    s = cp['DEFAULT'] if 'DEFAULT' in cp else {}
    return Settings(
        fiscal_year=int(s.get('fiscal_year', datetime.now().year)),
        capacity=int(s.get('capacity', 100)),
        breakfast_price=int(s.get('breakfast_price', 2000)),
        weight_prev_year=float(s.get('weight_prev_year', 0.7)),
        weight_2_3m=float(s.get('weight_2_3m', 0.1)),
        weight_recent=float(s.get('weight_recent', 0.2)),
    )


def save_settings(settings: Settings):
    cp = configparser.ConfigParser()
    cp['DEFAULT'] = {
        'fiscal_year': settings.fiscal_year,
        'capacity': settings.capacity,
        'breakfast_price': settings.breakfast_price,
        'weight_prev_year': settings.weight_prev_year,
        'weight_2_3m': settings.weight_2_3m,
        'weight_recent': settings.weight_recent,
    }
    with CONFIG_FILE.open('w') as f:
        cp.write(f)


def is_pre_holiday(date: datetime) -> bool:
    """Return True if ``date`` should be treated as a pre-holiday.

    A day is considered pre-holiday when any of the following conditions are met:
    * The next day is a holiday.
    * It is Saturday and the following Monday is a holiday (3 day weekend).
    * The next day is a weekend/holiday and the day after that is a holiday
      (handles longer blocks such as year-end or Golden Week).
    """

    # direct next day check
    if jpholiday.is_holiday(date + timedelta(days=1)):
        return True

    # Saturday -> Monday holiday
    if date.weekday() == 5 and jpholiday.is_holiday(date + timedelta(days=2)):
        return True

    # weekend/holiday bridge to a holiday within two days
    if (
        (date + timedelta(days=1)).weekday() >= 5
        or jpholiday.is_holiday(date + timedelta(days=1))
    ) and jpholiday.is_holiday(date + timedelta(days=2)):
        return True

    return False


def pre_holiday_boost(date: datetime) -> float:
    """Return weight boost factor for pre-holiday dates.

    Previous versions applied additional smoothing based on surrounding
    holidays.  The new specification treats pre-holidays the same as
    Saturdays, so no extra boost is applied.
    """
    return 1.0


def is_holiday_or_weekend(d: datetime) -> bool:
    """Return True if ``d`` is a weekend or a holiday."""
    return d.weekday() >= 5 or jpholiday.is_holiday(d)


def holiday_block_info(date: datetime):
    """Return (index, length) if ``date`` belongs to a holiday/weekend block.

    A block is a consecutive stretch of holiday/weekend days of length three or
    more.  Adjacent weekends around long holidays are included in the block.
    """

    if not is_holiday_or_weekend(date):
        return None

    start = date
    while is_holiday_or_weekend(start - timedelta(days=1)):
        start -= timedelta(days=1)

    length = 0
    d = start
    while is_holiday_or_weekend(d):
        length += 1
        d += timedelta(days=1)

    if length < 3:
        return None

    index = (date - start).days + 1
    return index, length


def is_pre_block(date: datetime) -> bool:
    """Return True if ``date`` is the day before a long holiday/weekend block."""
    return holiday_block_info(date + timedelta(days=1)) is not None


def day_type(date: datetime) -> str:
    """Classify ``date`` into day types used for distribution."""
    # Explicit keys for year end and New Year
    if date.month == 12 and date.day == 31:
        return "NYE"
    if date.month == 1 and date.day == 1:
        return "NYD"

    info = holiday_block_info(date)
    if info:
        index, _ = info
        return f"holiday{index}"
    if is_pre_block(date):
        return "preholiday"
    if jpholiday.is_holiday(date) and date.weekday() != 5:
        return "holiday"
    if is_pre_holiday(date):
        return "pre"
    return "normal"


def day_key(date: datetime):
    """Return a tuple representing the weekday and day type used for ratios."""
    dtype = day_type(date)
    return (date.weekday(), dtype)


def weekday_dist(df: pd.DataFrame, col: str) -> dict:
    """Return distribution ratio per (weekday, special) tuple with smoothing."""
    if df.empty:
        logging.debug("weekday_dist: empty dataframe for %s", col)
        return {}

    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if df[col].sum() == 0:
        logging.debug("weekday_dist: empty or zero data for %s", col)
        return {}

    df['key'] = df['date'].apply(day_key)
    grouped = df.groupby('key')[col].sum()

    keys = set(grouped.index)
    # ensure base weekday keys exist for fallback
    for i in range(7):
        keys.add((i, 'normal'))

    base = 0.01  # smoothing factor so rare keys don't vanish
    dist = {k: grouped.get(k, 0.0) + base for k in keys}

    # blend with uniform distribution when history is sparse
    if len(df) < 30 or len(keys) < 5:
        uniform = {k: 1 / len(keys) for k in keys}
        for k in dist:
            dist[k] = dist[k] * 0.7 + uniform[k] * 0.3

    total = sum(dist.values())
    if total == 0:
        # fallback to uniform distribution
        return {k: 1 / len(keys) for k in keys}
    for k in dist:
        dist[k] /= total

    if max(dist.values()) - min(dist.values()) < 0.05:
        logging.warning("weekday_dist ratios nearly uniform for %s", col)

    return dist


def weighted_dist(dists, weights):
    result = {}
    for d, w in zip(dists, weights):
        for k, v in d.items():
            result[k] = result.get(k, 0) + v * w
    s = sum(result.values())
    if s > 0:
        for k in result:
            result[k] /= s
    return result


def day_weight(date: datetime, key) -> float:
    """Return additional weight factor for the date.

    Placeholder for future per-day weighting logic. Currently returns ``1.0``.
    """
    return 1.0

def apply_distribution(dates, ratios, total, step):
    """Distribute total to each date according to weekday/holiday ratios."""
    if not ratios:
        return pd.Series([0] * len(dates), index=dates)

    keys = [day_key(d) for d in dates]
    weights = np.array([
        ratios.get(k, ratios.get((k[0], 'normal'), 0))
        for k in keys
    ])

    if weights.sum() == 0:
        weights[:] = 1 / len(weights)
    else:
        weights /= weights.sum()

    logging.debug(
        "apply_distribution weights stats min=%.4f max=%.4f mean=%.4f",
        weights.min(), weights.max(), weights.mean(),
    )

    base = total * weights

    if base.sum() == 0:
        base[:] = total / len(base)

    floor = np.floor(base / step) * step
    diff = total - floor.sum()
    order = np.argsort(base - floor)[::-1]
    i = 0
    while abs(diff) >= step / 2:
        floor[order[i % len(order)]] += step if diff > 0 else -step
        diff += -step if diff > 0 else step
        i += 1

    return pd.Series(floor, index=dates)


def adjust_to_total(series: pd.Series, target: float, step: int):
    if series.isna().any() or np.isinf(series).any():
        raise ValueError('series contains invalid value')
    base = series.copy().astype(float)
    diff = target - base.sum()
    order = np.argsort(base - np.floor(base/step)*step)[::-1].to_numpy()
    rng = np.random.default_rng()
    rng.shuffle(order)
    i = 0
    while abs(diff) >= step/2:
        idx = series.index[order[i % len(order)]]
        base[idx] += step if diff > 0 else -step
        diff += -step if diff > 0 else step
        i += 1
        if i % len(order) == 0:
            rng.shuffle(order)
    return base


def adjust_to_total_with_cap(series: pd.Series, target: float, step: int, cap: float):
    """Adjust series to match target without exceeding ``cap`` per element."""
    base = series.copy().astype(float)
    diff = target - base.sum()
    order = np.argsort(base - np.floor(base/step)*step)[::-1].to_numpy()
    rng = np.random.default_rng()
    rng.shuffle(order)
    caps = np.full(len(base), cap)
    i = 0
    loops = 0
    while abs(diff) >= step/2 and loops < len(order) * 10:
        idx = series.index[order[i % len(order)]]
        if diff > 0:
            if base[idx] + step <= caps[order[i % len(order)]]:
                base[idx] += step
                diff -= step
        else:
            if base[idx] - step >= 0:
                base[idx] -= step
                diff += step
        i += 1
        loops += 1
        if i % len(order) == 0:
            rng.shuffle(order)
    return base


def uniform_allocation(dates, total, step):
    """Return a uniform distribution series with residual on the last day."""
    per_day = (total // step) // len(dates) * step
    base = pd.Series(per_day, index=dates)
    residual = total - base.sum()
    if len(base) > 0:
        base.iloc[-1] += residual
    return base


def weekday_ratio(df: pd.DataFrame, col: str) -> dict:
    """Return weekday ratio = weekday average / overall average."""
    if df.empty:
        return {i: 1.0 for i in range(7)}

    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['weekday'] = df['date'].dt.weekday
    overall = df[col].mean() or 1.0

    ratios = {}
    for i in range(7):
        vals = df.loc[df['weekday'] == i, col]
        ratios[i] = (vals.mean() / overall) if not vals.empty else 1.0
    return ratios


def blend_weekday_ratios(ratios_list, weights):
    """Blend multiple weekday ratio tables with weights."""
    result = {i: 0.0 for i in range(7)}
    total = sum(weights)
    if total == 0:
        return {i: 1.0 for i in range(7)}
    for r, w in zip(ratios_list, weights):
        for i in range(7):
            result[i] += r.get(i, 1.0) * w
    for i in result:
        result[i] /= total
    return result


def allocate_from_ratios(dates, ratios, total, step):
    """Allocate ``total`` across ``dates`` using precomputed ``ratios`` array."""
    weights = np.array(ratios)
    if weights.sum() == 0:
        weights[:] = 1 / len(weights)
    else:
        weights /= weights.sum()

    base = total * weights
    floor = np.floor(base / step) * step
    diff = total - floor.sum()
    order = np.argsort(base - floor)[::-1]
    i = 0
    while abs(diff) >= step / 2:
        floor[order[i % len(order)]] += step if diff > 0 else -step
        diff += -step if diff > 0 else step
        i += 1

    return pd.Series(floor, index=dates)


def compute_daily_ratios(dates, weekday_ratios):
    """Return daily ratio series with holiday adjustments."""
    ratios = [weekday_ratios.get(d.weekday(), 1.0) for d in dates]

    # pre-holiday days use Saturday ratio
    for i in range(len(dates) - 1):
        if not is_holiday_or_weekend(dates[i]) and is_holiday_or_weekend(dates[i + 1]):
            ratios[i] = weekday_ratios.get(5, 1.0)

    # consecutive holiday blocks
    i = 0
    while i < len(dates):
        if is_holiday_or_weekend(dates[i]):
            prev = weekday_ratios.get(5, 1.0)
            ratios[i] = prev
            j = i + 1
            while j < len(dates) and is_holiday_or_weekend(dates[j]):
                prev *= 0.9
                ratios[j] = prev
                j += 1
            if j < len(dates) and not is_holiday_or_weekend(dates[j]):
                ratios[j] = weekday_ratios.get(dates[j].weekday(), 1.0) * 0.9
            i = j
        else:
            i += 1

    # New Year special handling
    for idx, d in enumerate(dates):
        if d.month == 12 and d.day == 31:
            ratios[idx] = weekday_ratios.get(5, 1.0)
            base = ratios[idx]
            if idx + 1 < len(dates) and dates[idx + 1].month == 1 and dates[idx + 1].day == 1:
                ratios[idx + 1] = base * 0.9
                base = ratios[idx + 1]
                if idx + 2 < len(dates) and dates[idx + 2].month == 1 and dates[idx + 2].day == 2:
                    ratios[idx + 2] = base * 0.8
                    base = ratios[idx + 2]
                    if idx + 3 < len(dates) and dates[idx + 3].month == 1 and dates[idx + 3].day == 3:
                        ratios[idx + 3] = base * 0.8
    return pd.Series(ratios, index=dates)


def process(settings: Settings):
    monthly = pd.read_csv(INPUT_DIR / '月次予算.csv', encoding='utf-8-sig')
    daily = pd.read_csv(INPUT_DIR / '日別実績.csv', encoding='utf-8-sig')

    max_guests = daily['人数'].max()
    max_adr_hist = (daily['宿泊売上'] / daily['室数'].replace(0, np.nan)).max()
    max_revpar_hist = (daily['宿泊売上'] / settings.capacity).max()
    max_daily_hist = daily['宿泊売上'].max()
    daily_rev_cap = round(max_daily_hist * 1.05 / 100) * 100
    logging.info("daily revenue cap set to %.0f", daily_rev_cap)

    numeric_cols_m = ['年', '月', '室数', '人数', '宿泊売上', '朝食売上', '料飲その他売上', 'その他売上', '総合計', '喫食数']
    for c in numeric_cols_m:
        monthly[c] = pd.to_numeric(monthly[c], errors='coerce')

    numeric_cols_d = ['室数', '人数', '宿泊売上', '喫食数']
    for c in numeric_cols_d:
        daily[c] = pd.to_numeric(daily[c], errors='coerce')

    daily['date'] = pd.to_datetime(daily['日付'], errors='coerce')
    daily = daily.dropna(subset=['date'])

    year_start = datetime(settings.fiscal_year, 6, 1)
    output_book = {}
    ratio_rows = []

    latest_date = daily['date'].max()
    recent_start = latest_date - pd.DateOffset(days=30)
    recent_end = latest_date - pd.DateOffset(days=1)
    range_recent_mask = (daily['date'] >= recent_start) & (daily['date'] <= recent_end)
    ratio_recent_base = weekday_ratio(daily.loc[range_recent_mask], '宿泊売上')

    range_23_start = latest_date - pd.DateOffset(days=90)
    range_23_end = latest_date - pd.DateOffset(days=30)
    range_23_mask = (daily['date'] >= range_23_start) & (daily['date'] < range_23_end)
    ratio_23_base = weekday_ratio(daily.loc[range_23_mask], '宿泊売上')

    logging.info(
        "weekday dist base ranges recent=%s to %s, 2-3m=%s to %s",
        recent_start.date(), recent_end.date(),
        range_23_start.date(), (range_23_end - pd.Timedelta(days=1)).date(),
    )

    for _, row in monthly.iterrows():
        year = int(row['年'])
        month = int(row['月'])
        logging.info("processing %d-%02d", year, month)
        start = datetime(year, month, 1)
        end = (start + pd.offsets.MonthEnd()).to_pydatetime()

        prev_start1 = start - pd.DateOffset(years=1)
        prev_end1 = end - pd.DateOffset(years=1)
        mask_prev1 = (daily['date'] >= prev_start1) & (daily['date'] <= prev_end1)
        mask_prev2 = pd.Series(False, index=daily.index)

        ratio_prev = {}
        prev_label = None
        if not daily.loc[mask_prev1].empty:
            ratio_prev = weekday_ratio(daily.loc[mask_prev1], '宿泊売上')
            prev_label = f"{prev_start1.date()} to {prev_end1.date()}"
            logging.info("same month last year used for %d-%02d: %s", year, month, prev_label)
        else:
            prev_start2 = start - pd.DateOffset(years=2)
            prev_end2 = end - pd.DateOffset(years=2)
            mask_prev2 = (daily['date'] >= prev_start2) & (daily['date'] <= prev_end2)
            if not daily.loc[mask_prev2].empty:
                ratio_prev = weekday_ratio(daily.loc[mask_prev2], '宿泊売上')
                prev_label = f"{prev_start2.date()} to {prev_end2.date()} (2y)"
                logging.info("same month fallback 2yrs for %d-%02d: %s", year, month, prev_label)
            else:
                logging.info("same month data missing for %d-%02d", year, month)

        mask_prev = mask_prev1 if not daily.loc[mask_prev1].empty else mask_prev2
        mask_recent = (daily['date'] >= start - pd.DateOffset(days=30)) & (daily['date'] < start)

        ratios_list = []
        weights = []
        labels = []
        if ratio_prev:
            ratios_list.append(ratio_prev)
            weights.append(settings.weight_prev_year)
            labels.append('prev')
        if ratio_23_base:
            ratios_list.append(ratio_23_base)
            weights.append(settings.weight_2_3m)
            labels.append('2-3m')
        if ratio_recent_base:
            ratios_list.append(ratio_recent_base)
            weights.append(settings.weight_recent)
            labels.append('recent')

        if not weights:
            weekday_ratios = {i: 1.0 for i in range(7)}
            logging.warning("no historical data for weekday distribution %d-%02d", year, month)
        else:
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            weekday_ratios = blend_weekday_ratios(ratios_list, weights)
            weight_info = {l: round(w, 3) for l, w in zip(labels, weights)}
            logging.info("distribution weights for %d-%02d: %s", year, month, weight_info)

        logging.debug("weekday ratios %s", weekday_ratios)
        for wd, rt in weekday_ratios.items():
            ratio_rows.append({'年': year, '月': month, '曜日': wd, '係数': rt})
        dates = pd.date_range(start, end)
        df = pd.DataFrame({'date': dates})
        daily_ratios = compute_daily_ratios(dates, weekday_ratios)
        df['宿泊売上'] = allocate_from_ratios(dates, daily_ratios.values, row['宿泊売上'], 100).values
        df['室数'] = allocate_from_ratios(dates, daily_ratios.values, row['室数'], 1).values
        df['人数'] = allocate_from_ratios(dates, daily_ratios.values, row['人数'], 1).values

        # ensure no zero allocations for key metrics
        for col, step, total_val in [
            ('宿泊売上', 100, row['宿泊売上']),
            ('室数', 1, row['室数']),
            ('人数', 1, row['人数']),
        ]:
            zeros = df[col] <= 0
            if zeros.any():
                logging.info('zero allocation adjusted for %s in %d-%02d', col, year, month)
                min_val = step if df.loc[~zeros, col].empty else df.loc[~zeros, col].min()
                df.loc[zeros, col] = min_val
                df[col] = adjust_to_total(df[col], total_val, step)

        # dynamic room upper bound based on history
        mask_recent3 = (daily['date'] >= start - pd.DateOffset(months=3)) & (daily['date'] < start)
        max_prev = daily.loc[mask_prev, '室数'].max()
        max_recent = daily.loc[mask_recent3, '室数'].max()
        valid_vals = [v for v in [max_prev, max_recent] if not np.isnan(v)]
        room_upper_bound = max(valid_vals) if valid_vals else settings.capacity
        room_upper_bound = min(room_upper_bound, settings.capacity)
        before_sum = df['室数'].sum()
        df['室数'] = df['室数'].clip(upper=room_upper_bound)
        if not np.isclose(before_sum, df['室数'].sum()):
            logging.warning('room count clipped for %d-%02d', year, month)
        df['室数'] = adjust_to_total_with_cap(df['室数'], row['室数'], 1, room_upper_bound)
        df['室数'] = adjust_to_total(df['室数'], row['室数'], 1)

        # minimum revenue threshold based on historical lows
        prev_min = daily.loc[mask_prev, '宿泊売上'].min()
        recent_min = daily.loc[mask_recent, '宿泊売上'].min()
        floor_candidates = [v for v in [prev_min, recent_min] if not np.isnan(v)]
        if floor_candidates:
            revenue_floor = np.floor(min(floor_candidates) * 0.9 / 100) * 100
            before = df['宿泊売上'].copy()
            df['宿泊売上'] = df['宿泊売上'].clip(lower=revenue_floor)
            if not before.equals(df['宿泊売上']):
                logging.info('revenue floor %.0f applied for %d-%02d', revenue_floor, year, month)
            df['宿泊売上'] = adjust_to_total(df['宿泊売上'], row['宿泊売上'], 100)

        # guest count bounds based on history
        guest_prev_max = daily.loc[mask_prev, '人数'].max()
        guest_recent_max = daily.loc[mask_recent3, '人数'].max()
        guest_prev_min = daily.loc[mask_prev, '人数'].min()
        guest_recent_min = daily.loc[mask_recent3, '人数'].min()
        upper_candidates = [v for v in [guest_prev_max, guest_recent_max] if not np.isnan(v)]
        lower_candidates = [v for v in [guest_prev_min, guest_recent_min] if not np.isnan(v)]
        guest_upper = max(upper_candidates) * 1.05 if upper_candidates else max_guests
        guest_upper = min(guest_upper, max_guests)
        guest_lower = min(lower_candidates) * 0.95 if lower_candidates else 0
        before_guests = df['人数'].copy()
        df['人数'] = df['人数'].clip(lower=guest_lower, upper=guest_upper)
        df['人数'] = adjust_to_total_with_cap(df['人数'], row['人数'], 1, guest_upper)
        if not before_guests.equals(df['人数']):
            logging.warning('guest count clipped for %d-%02d', year, month)
        df['人数'] = df['人数'].round().astype(int)
        for c in ['宿泊売上', '室数', '人数']:
            if df[c].sum() == 0:
                logging.warning('all zeros allocated for %s in %d-%02d', c, year, month)

        person_ratio = df['人数'] / df['人数'].sum() if df['人数'].sum() else 0
        breakfast_count = row['喫食数'] if not np.isnan(row['喫食数']) else row['朝食売上']/settings.breakfast_price

        # round counts to integer while keeping the monthly total
        base_count = (person_ratio * breakfast_count).round()
        df['喫食数'] = adjust_to_total(base_count, breakfast_count, 1).round().astype(int)

        # breakfast revenue should also be integer (100 yen step)
        base_revenue = (df['喫食数'] * settings.breakfast_price).round()
        df['朝食売上'] = adjust_to_total(base_revenue, row['朝食売上'], 100).round().astype(int)

        fb_other = uniform_allocation(dates, row['料飲その他売上'], 100)
        other = uniform_allocation(dates, row['その他売上'], 100)
        df['料飲その他売上'] = fb_other.values
        df['その他売上'] = other.values

        # metric validation
        max_adr = max_adr_hist
        max_revpar = max_revpar_hist
        for idx, r in df.iterrows():
            rooms = max(r['室数'], 1)
            adr = r['宿泊売上'] / rooms
            dor = r['人数'] / rooms
            revpar = r['宿泊売上'] / settings.capacity
            if adr > max_adr * 1.1:
                df.at[idx, '宿泊売上'] = rooms * max_adr * 1.1
                logging.warning('ADR clipped on %s', r['date'])
            if dor > 2.5:
                df.at[idx, '人数'] = rooms * 2.5
                logging.warning('DOR high on %s', r['date'])
            if dor < 1.0 and r['人数'] > 0:
                df.at[idx, '人数'] = rooms * 1.0
                logging.warning('DOR low on %s', r['date'])
            if revpar > max_revpar * 1.1:
                df.at[idx, '宿泊売上'] = settings.capacity * max_revpar * 1.1
                logging.warning('RevPAR clipped on %s', r['date'])
            if r['宿泊売上'] == 0:
                logging.warning('zero revenue on %s', r['date'])
            if df.at[idx, '宿泊売上'] > daily_rev_cap:
                df.at[idx, '宿泊売上'] = daily_rev_cap
                logging.warning('daily revenue clipped to cap on %s', r['date'])

        # scale down if revenue sum exceeds monthly budget
        total_revenue = df['宿泊売上'].sum()
        if total_revenue > row['宿泊売上']:
            scale = row['宿泊売上'] / total_revenue
            df['宿泊売上'] = (df['宿泊売上'] * scale).round()

        # ensure monthly totals match budget after all adjustments
        for col, step, total_val in [
            ('宿泊売上', 100, row['宿泊売上']),
            ('室数', 1, row['室数']),
            ('人数', 1, row['人数']),
        ]:
            if col == '人数':
                df[col] = adjust_to_total_with_cap(df[col], total_val, step, guest_upper).round().astype(int)
            elif col == '室数':
                df[col] = adjust_to_total_with_cap(df[col], total_val, step, room_upper_bound)
                df[col] = adjust_to_total(df[col], total_val, step).round().astype(int)
            else:
                df[col] = adjust_to_total(df[col], total_val, step)

        # occupancy cap handling
        cap_val = min(room_upper_bound, settings.capacity)
        occ_before = df['室数'] / settings.capacity
        df['室数'] = df['室数'].clip(upper=cap_val)
        df['室数'] = adjust_to_total_with_cap(df['室数'], row['室数'], 1, cap_val)
        df['室数'] = adjust_to_total(df['室数'], row['室数'], 1)
        df['人数'] = adjust_to_total_with_cap(df['人数'], row['人数'], 1, guest_upper)
        df['人数'] = adjust_to_total(df['人数'], row['人数'], 1)
        occ_after = df['室数'] / settings.capacity
        deviation = (occ_after - occ_before).abs() / occ_before.replace(0, np.nan)
        for idx in df.index[deviation > 0.1]:
            logging.warning('OCC clipped >10%% on %s', df.at[idx, 'date'])

        df['総合計'] = df[['宿泊売上','朝食売上','料飲その他売上','その他売上']].sum(axis=1)
        total_diff = row['総合計'] - df['総合計'].sum()
        if abs(total_diff) >= 1:
            df.loc[df.index[-1],'その他売上'] += total_diff
            df.loc[df.index[-1],'総合計'] += total_diff

        # KPI calculations
        rooms_safe = np.where(df['室数'] == 0, 1, df['室数'])
        df['ADR'] = (df['宿泊売上'] / rooms_safe).round().astype(int)
        df['DOR'] = (df['人数'] / rooms_safe).round(2)
        df['RevPAR'] = (df['宿泊売上'] / settings.capacity).round().astype(int)
        df['OCC'] = np.minimum(df['室数'] / settings.capacity, 1.0).round(2)

        # add date related columns before computing the summary row
        weekday_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
        df['曜日'] = df['date'].dt.weekday.map(weekday_map)
        df['祝日名'] = df['date'].apply(lambda d: jpholiday.is_holiday_name(d) or '')
        # reorder so that date related columns appear right after date and KPIs at the end
        kpi_cols = ['ADR', 'DOR', 'RevPAR', 'OCC']
        col_order = ['date', '曜日', '祝日名'] + [c for c in df.columns if c not in ['date', '曜日', '祝日名'] + kpi_cols] + kpi_cols
        df = df[col_order]
        df['date'] = df['date'].dt.strftime('%Y/%-m/%-d')

        summary = df.sum(numeric_only=True)
        r_total = summary.get('宿泊売上', 0)
        room_total = summary.get('室数', 0)
        guest_total = summary.get('人数', 0)
        days = len(df)
        summary['ADR'] = round(r_total / room_total) if room_total else 0
        summary['DOR'] = round(guest_total / room_total, 2) if room_total else 0
        summary['RevPAR'] = round(r_total / (settings.capacity * days)) if days else 0
        summary['OCC'] = round(room_total / (settings.capacity * days), 2) if days else 0
        df.loc['合計'] = summary
        output_book[f'{year}-{month:02d}'] = df
    path = OUTPUT_DIR / f'日別予算_{settings.fiscal_year}.xlsx'
    with pd.ExcelWriter(path) as writer:
        for name, data in output_book.items():
            data.to_excel(writer, sheet_name=name, index=False)
        if ratio_rows:
            pd.DataFrame(ratio_rows).to_excel(writer, sheet_name='係数', index=False)
    logging.info("saved output to %s", path)


def run_gui():
    settings = load_settings()
    root = tk.Tk()
    root.title('日別予算自動生成')

    tk.Label(root, text='年度').grid(row=0, column=0)
    fiscal_var = tk.IntVar(value=settings.fiscal_year)
    tk.Entry(root, textvariable=fiscal_var).grid(row=0, column=1)

    tk.Label(root, text='室数キャパシティ').grid(row=1, column=0)
    cap_var = tk.IntVar(value=settings.capacity)
    tk.Entry(root, textvariable=cap_var).grid(row=1, column=1)

    tk.Label(root, text='朝食単価').grid(row=2, column=0)
    price_var = tk.IntVar(value=settings.breakfast_price)
    tk.Entry(root, textvariable=price_var).grid(row=2, column=1)

    tk.Label(root, text='前年同月ウェイト').grid(row=3, column=0)
    w1_var = tk.DoubleVar(value=settings.weight_prev_year)
    tk.Entry(root, textvariable=w1_var).grid(row=3, column=1)

    tk.Label(root, text='2-3ヶ月前ウェイト').grid(row=4, column=0)
    w2_var = tk.DoubleVar(value=settings.weight_2_3m)
    tk.Entry(root, textvariable=w2_var).grid(row=4, column=1)

    tk.Label(root, text='直近1ヶ月ウェイト').grid(row=5, column=0)
    w3_var = tk.DoubleVar(value=settings.weight_recent)
    tk.Entry(root, textvariable=w3_var).grid(row=5, column=1)

    def on_run():
        s = Settings(
            fiscal_year=fiscal_var.get(),
            capacity=cap_var.get(),
            breakfast_price=price_var.get(),
            weight_prev_year=w1_var.get(),
            weight_2_3m=w2_var.get(),
            weight_recent=w3_var.get(),
        )
        save_settings(s)
        try:
            process(s)
            messagebox.showinfo('完了', '処理が完了しました')
        except Exception as e:
            logging.exception('error during process')
            messagebox.showerror('エラー', str(e))

    tk.Button(root, text='実行', command=on_run).grid(row=6, column=0, columnspan=2)
    root.mainloop()

if __name__ == '__main__':
    run_gui()
