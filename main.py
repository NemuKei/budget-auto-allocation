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
    # Pre-holiday dates share the Saturday coefficient for distribution
    if dtype in ('pre', 'preholiday'):
        return (5, 'normal')
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
    base = 0.01  # smoothing factor so rare keys don't vanish
    dist = {k: grouped.get(k, 0.0) + base for k in keys}

    total = sum(dist.values())
    if total == 0:
        # fallback to uniform distribution
        return {k: 1 / len(keys) for k in keys}
    for k in dist:
        dist[k] /= total
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

    The new rules do not apply smoothing based on day type here; all
    adjustments are handled in ``apply_distribution``.
    """
    return 1.0

def apply_distribution(dates, ratios, total, step):
    """Distribute total to each date according to weekday/holiday ratios."""
    if not ratios:
        return pd.Series([0] * len(dates), index=dates)

    # count how many times each key appears in the target dates
    keys = [day_key(d) for d in dates]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1

    # check if ratios sufficiently cover the keys; otherwise fall back to uniform
    ratio_sum = sum(ratios.get(k, 0) for k in counts)
    missing = [k for k in counts if ratios.get(k, 0) == 0]
    if ratio_sum == 0 or len(missing) > len(counts) / 2:
        logging.info("apply_distribution fallback to uniform distribution")
        ratios = {k: 1 / len(counts) for k in counts}
    else:
        logging.debug("apply_distribution using normal distribution")

    base_sat = ratios.get((5, 'normal'), 0)

    def single_ratio(date, key):
        dtype = day_type(date)
        info = holiday_block_info(date)
        # consecutive holiday/weekend block adjustment
        if info and info[1] >= 3:
            idx, length = info
            if idx == 1:
                return base_sat * 1.1
            elif idx == length:
                return base_sat * 0.6
            else:
                return base_sat * 0.9
        # pre-holiday dates share Saturday ratio
        if dtype in ('pre', 'preholiday'):
            return base_sat
        return ratios.get(key, 0)

    weights = np.array([
        single_ratio(d, k) * day_weight(d, k) / counts.get(k, 1)
        for d, k in zip(dates, keys)
    ])

    if weights.sum() == 0:
        weights[:] = 1 / len(weights)
    else:
        weights /= weights.sum()

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
    order = np.argsort(base - np.floor(base/step)*step)[::-1]
    i = 0
    while abs(diff) >= step/2:
        idx = series.index[order[i%len(order)]]
        base[idx] += step if diff>0 else -step
        diff += -step if diff>0 else step
        i += 1
    return base


def adjust_to_total_with_cap(series: pd.Series, target: float, step: int, cap: float):
    """Adjust series to match target without exceeding ``cap`` per element."""
    base = series.copy().astype(float)
    diff = target - base.sum()
    order = np.argsort(base - np.floor(base/step)*step)[::-1]
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
    return base


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

    for _, row in monthly.iterrows():
        year = int(row['年'])
        month = int(row['月'])
        logging.info("processing %d-%02d", year, month)
        start = datetime(year, month, 1)
        end = (start + pd.offsets.MonthEnd()).to_pydatetime()
        mask_prev = (daily['date'] >= start - pd.DateOffset(years=1)) & (daily['date'] <= end - pd.DateOffset(years=1))
        mask_23 = (daily['date'] >= start - pd.DateOffset(days=90)) & (daily['date'] <= start - pd.DateOffset(days=30))
        mask_recent = (daily['date'] >= start - pd.DateOffset(days=30)) & (daily['date'] < start)
        dists = [
            weekday_dist(daily.loc[mask_prev], '宿泊売上'),
            weekday_dist(daily.loc[mask_23], '宿泊売上'),
            weekday_dist(daily.loc[mask_recent], '宿泊売上'),
        ]
        ratios = weighted_dist(dists, [settings.weight_prev_year, settings.weight_2_3m, settings.weight_recent])
        logging.debug("ratios %s", ratios)
        dates = pd.date_range(start, end)
        df = pd.DataFrame({'date': dates})
        df['宿泊売上'] = apply_distribution(dates, ratios, row['宿泊売上'], 100).values
        df['室数'] = apply_distribution(dates, ratios, row['室数'], 1).values
        df['人数'] = apply_distribution(dates, ratios, row['人数'], 1).values

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
        before_rooms = df['室数'].copy()
        df['室数'] = df['室数'].clip(upper=room_upper_bound)
        if not before_rooms.equals(df['室数']):
            logging.warning('room count clipped for %d-%02d', year, month)
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

        # enforce guest maximums
        df['人数'] = df['人数'].clip(upper=max_guests)
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

        fb_other = apply_distribution(dates, {('any','any'):1/len(dates)}, row['料飲その他売上'], 100)
        other = apply_distribution(dates, {('any','any'):1/len(dates)}, row['その他売上'], 100)
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

        # ensure monthly totals match budget after all adjustments
        for col, step, total_val in [
            ('宿泊売上', 100, row['宿泊売上']),
            ('室数', 1, row['室数']),
            ('人数', 1, row['人数']),
        ]:
            df[col] = adjust_to_total(df[col], total_val, step)

        # occupancy cap handling
        cap_val = min(room_upper_bound, settings.capacity)
        occ_before = df['室数'] / settings.capacity
        df['室数'] = df['室数'].clip(upper=cap_val)
        df['室数'] = adjust_to_total_with_cap(df['室数'], row['室数'], 1, cap_val)
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
        summary['ADR'] = round(r_total / room_total) if room_total else 0
        summary['DOR'] = round(guest_total / room_total, 2) if room_total else 0
        summary['RevPAR'] = round(r_total / settings.capacity)
        summary['OCC'] = round(room_total / settings.capacity, 2)
        df.loc['合計'] = summary
        output_book[f'{year}-{month:02d}'] = df
    path = OUTPUT_DIR / f'日別予算_{settings.fiscal_year}.xlsx'
    with pd.ExcelWriter(path) as writer:
        for name, data in output_book.items():
            data.to_excel(writer, sheet_name=name, index=False)
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
