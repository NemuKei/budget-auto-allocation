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
    next_day = date + timedelta(days=1)
    return date.weekday() < 4 and jpholiday.is_holiday(next_day)


def day_key(date: datetime):
    special = 'holiday' if jpholiday.is_holiday(date) else 'pre' if is_pre_holiday(date) else 'normal'
    return (date.weekday(), special)


def weekday_dist(df: pd.DataFrame, col: str) -> dict:
    """Return distribution ratio per (weekday, special) tuple."""
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
    total = grouped.sum()
    if total == 0:
        return {}
    dist = (grouped / total).to_dict()
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


def apply_distribution(dates, ratios, total, step):
    """Distribute total to each date according to weekday/holiday ratios."""
    if not ratios:
        return pd.Series([0] * len(dates), index=dates)

    # count how many times each key appears in the target dates
    keys = [day_key(d) for d in dates]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1

    base = np.array([
        total * ratios.get(k, 0) / counts.get(k, 1)
        for k in keys
    ])

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


def process(settings: Settings):
    monthly = pd.read_csv(INPUT_DIR / '月次予算.csv', encoding='utf-8-sig')
    daily = pd.read_csv(INPUT_DIR / '日別実績.csv', encoding='utf-8-sig')

    max_guests = daily['人数'].max()

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

        # enforce capacity and historical guest maximums
        df['室数'] = df['室数'].clip(upper=settings.capacity)
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

        df['総合計'] = df[['宿泊売上','朝食売上','料飲その他売上','その他売上']].sum(axis=1)
        total_diff = row['総合計'] - df['総合計'].sum()
        if abs(total_diff) >= 1:
            df.loc[df.index[-1],'その他売上'] += total_diff
            df.loc[df.index[-1],'総合計'] += total_diff

        # add date related columns before computing the summary row
        weekday_map = {0:'月', 1:'火', 2:'水', 3:'木', 4:'金', 5:'土', 6:'日'}
        df['曜日'] = df['date'].dt.weekday.map(weekday_map)
        df['祝日'] = df['date'].apply(lambda d: '祝' if pd.notnull(d) and jpholiday.is_holiday(d) else '')
        df['date'] = df['date'].dt.strftime('%Y/%-m/%-d')

        df.loc['合計'] = df.sum(numeric_only=True)
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
