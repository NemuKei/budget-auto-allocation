# 日別予算自動割当ツール

月次予算と日別実績をもとに、ホテル向けの日別予算（売上・室数・人数など）を自動配分する Python ツールです。GUI から設定値を調整して実行できます。

## 対象ファイル

- 実行スクリプト: `budget-auto-allocation.py`
- 設定ファイル: `settings.ini`
- 入力フォルダ: `input/`
- 出力フォルダ: `output/`

## 主な処理

- 月次予算を日次に配分（宿泊売上・室数・人数・朝食売上・その他売上）
- 曜日傾向（前年同月 / 直近 2-3 か月 / 直近 1 か月）を重み付きで平滑化
- 祝日・連休・GW・年末年始・3月後半の補正
- DOR / ADR / RevPAR / OCC の算出
- 月次合計値との一致調整

## 動作環境

- Python 3.10 以上（推奨）
- Windows（Tkinter GUI 利用）

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 実行方法

```bash
python budget-auto-allocation.py
```

GUI の「実行」ボタン押下で処理が開始されます。

## 入力仕様

文字コードは `UTF-8 with BOM (utf-8-sig)` を想定しています。

### `input/月次予算.csv`

必要列:

- `年`
- `月`
- `室数`
- `人数`
- `宿泊売上`
- `朝食売上`
- `料飲その他売上`
- `その他売上`
- `総合計`
- `喫食数`

### `input/日別実績.csv`

必要列:

- `日付`（日付として解釈可能な値）
- `室数`
- `人数`
- `宿泊売上`
- `喫食数`

## 設定ファイル（`settings.ini`）

`[DEFAULT]` セクションに以下を設定します。

- `fiscal_year`
- `capacity`
- `breakfast_price`
- `weight_prev_year`
- `weight_2_3m`
- `weight_recent`
- `dor_min`
- `dor_max`（0 以下で上限なし）

## 出力

- `output/日別予算_<年度>.xlsx`
- 月別シート + 平滑化係数シート（`平滑化係数`）

## 補足

- `main.spec` は PyInstaller 用のひな形ですが、現状の実行スクリプト名と差異がある場合は `Analysis([...])` の対象を実ファイルに合わせてください。
