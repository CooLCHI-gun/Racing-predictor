# 🏇 HKJC 賽馬預測系統

基於機器學習（XGBoost + LightGBM 集成模型）的香港賽馬會賽馬預測系統，具備完整的 Web 儀表板、實時賠率更新、Telegram 通知及回測分析功能。

---

## 系統架構

```
Racing-predictor/
├── config.py                 # 全局設定
├── run.py                    # 主程式入口
├── requirements.txt
├── data/
│   ├── raw/                  # 原始抓取數據
│   ├── processed/            # 特徵數據 (ELO, 統計)
│   ├── models/               # 訓練模型
│   └── predictions/          # 預測歷史
├── scraper/                  # 數據抓取模組
│   ├── racecard.py           # 賽事表
│   ├── horse_profile.py      # 馬匹過往表現
│   ├── odds.py               # 即時賠率
│   └── weather.py            # 天氣 / 場地狀況
├── features/                 # 特徵工程
│   ├── elo.py                # ELO 評分系統
│   ├── jockey_trainer.py     # 騎師 / 練馬師統計
│   ├── draw.py               # 檔位統計
│   └── builder.py            # 特徵建構管線
├── model/
│   ├── trainer.py            # XGBoost + LightGBM 訓練
│   ├── predictor.py          # 預測引擎
│   └── backtest.py           # 回測 & ROI 追蹤
├── web/
│   ├── app.py                # Flask 應用
│   ├── templates/            # Jinja2 模板 (繁體中文)
│   └── static/               # CSS / JS
├── notifier/
│   └── telegram.py           # Telegram 通知
└── scheduler/
    └── tasks.py              # APScheduler 排程任務
```

---

## 數據流架構

```
HKJC 網站 / HKO API
        │
        ▼
   [scraper/]          抓取賽事表、馬匹資料、賠率、天氣
        │
        ▼
   [features/]         ELO評分 + 騎師/練馬師統計 + 檔位分析
        │
        ▼
   [model/]            XGBoost + LightGBM 集成預測
        │
    ┌───┴───┐
    ▼       ▼
[web/app]  [notifier/]   Flask 儀表板 + Telegram 通知
    │
    ▼
[scheduler/]             APScheduler 自動化排程
```

---

## 快速安裝

### 1. 系統需求

- Python 3.9+
- pip

### 2. 安裝依賴

```bash
cd Racing-predictor
pip install -r requirements.txt
```

### 3. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env 填入 Telegram Bot Token (可選)
```

---

## 執行方式

### 開發模式（模擬數據，無需真實 HKJC 數據）

```bash
# 確保 config.py 中 DEMO_MODE = True
python run.py --mode web --debug
```

瀏覽器開啟：http://localhost:5000

### 各種執行模式

```bash
# Web 儀表板 + 自動排程
python run.py --mode web

# 一鍵診斷（適合 Railway log 排障）
python run.py --mode debug

# 一次性輪詢（適合 Railway Cron 每5分鐘）
python run.py --mode tick

# 單一 cron command（Railway 只可設一條指令時）
python run.py --mode cron

# 每日維護（回測 + 優化 + Telegram 摘要）
python run.py --mode maintenance

# 僅顯示今日預測（命令行）
python run.py --mode predict

# 回測成績報告
python run.py --mode backtest

# 抓取今日賽事表
python run.py --mode fetch

# 重新訓練模型（使用合成數據）
python run.py --mode train

# 強制使用模擬模式
python run.py --mode demo

# 自定義端口
python run.py --mode web --port 8080

# 不啟動排程器
python run.py --mode web --no-scheduler
```

---

## 功能說明

### 🤖 預測模型

| 特徵類別 | 特徵 |
|---------|------|
| ELO 評分 | elo_rating, elo_vs_field, elo_delta |
| 騎師統計 | 近30日勝率、到位率、賽場/距離勝率 |
| 練馬師統計 | 近30日勝率、騎師練馬師組合勝率 |
| 檔位分析 | 各距離/賽場檔位優劣勢 |
| 馬匹近況 | 近6場加權表現分、距離適應性、場地適應 |
| 評分趨勢 | 官方評分近期走勢 |
| 市場信息 | 賠率隱含概率、EV（期望值） |

**集成策略**: XGBoost (50%) + LightGBM (50%)
**目標變量**: 是否前三名到位（二分類）

### 📊 ELO 評分

採用棋類競技的 ELO 評分改良版：
- 每場賽事後根據名次更新所有參賽馬匹評分
- K係數 = 32（可在 config.py 調整）
- 新馬預設評分 1500

### 💰 值博推介

當模型估算概率 > 賠率隱含概率 × 110% 時標記為「值博」：
```
優勢 = (模型概率 - 賠率概率) / 賠率概率 > 10%
```

### 📱 Telegram 通知格式

```
🏇 第X場 - 沙田 14:00
📍 距離: 1400m | 班次: Class 3 | 場地: 草地
⛅ 場地狀況: GOOD

🎯 頭三名預測 (不分先後):
1️⃣ 馬號3 [GOLDEN SIXTY] - 信心度 72% ⭐值博
2️⃣ 馬號7 [ROMANTIC WARRIOR] - 信心度 65%
3️⃣ 馬號1 [SIGHT SUCCESS] - 信心度 58%

💰 三T建議: 3/7/1
📊 EV最佳: 馬號3 (模型35% vs 賠率隱含28%)

⚠️ 僅供研究參考，不構成投注建議
```

---

## 生產部署（真實 HKJC 數據）

1. 修改 `config.py`：`DEMO_MODE = False`
2. 設定 `.env` 中的 Telegram 憑據
3. 確保服務器可連接 HKJC 網站
4. 使用 PM2 或 systemd 維持後台運行

### Railway 部署（推薦）

本專案的正確入口是 `run.py`，不是 `src.main`。

- Web + 排程（定期 Telegram）啟動指令：`python run.py --mode web`
- Railway 單一 cron 指令（tick + 定時維護）: `python run.py --mode cron`
- Railway Cron 每5分鐘輪詢指令：`python run.py --mode tick`
- 快速診斷指令（分類顯示非賽日/無場次/網絡失敗）：`python run.py --mode debug`
- 驗證 Telegram 憑據是否正確：`python run.py --mode telegram-test`

Railway 設定重點：

1. 使用 GitHub 連接此 repo，自動部署。
2. Start Command 使用 `python run.py --mode web`（或使用 Procfile）。
3. 於 Variables 設定：
    - `TELEGRAM_TOKEN`
    - `TELEGRAM_CHAT_ID`
    - `MESSAGE_STYLE=pro`（`pro` 專業 / `casual` 口語）
    - `DEMO_MODE=false`
    - `REAL_DATA_ONLY=true`
    - `ALLOW_SYNTHETIC_TRAINING=false`
    - `FLASK_DEBUG=false`
    - `TICK_SENT_STATE_FILE=/data/tick_notified.json`（建議配合 Railway Volume）
    - `CRON_STATE_FILE=/data/cron_state.json`
    - `CRON_MAINTENANCE_HOUR=21`
    - `CRON_MAINTENANCE_MINUTE=50`
    - `CRON_MAINTENANCE_WINDOW_MINS=10`
    - `MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED=true`
    - `MAINTENANCE_REPORT_DIR=/data/reports`
    - `DAILY_RETRAIN_ENABLED=true`
4. 服務需長駐（不可休眠），否則 APScheduler 不會在休眠期間執行。
5. Scheduler 內建時區為 `Asia/Hong_Kong`，會按 HKT 定時推送。

Railway Cron 建議任務：

【方案A：Railway 只可設一條 command】

1. 每5分鐘執行: `python run.py --mode cron`
2. 程式每次都會跑 tick
3. 每日到指定時間視窗（預設 21:50-21:59 HKT）只會跑一次 maintenance
4. maintenance 只在有新結算賽果時才推 Telegram 摘要
5. 每日輸出分析報告 JSON（預設 `data/reports/maintenance_YYYY-MM-DD.json`）
6. 若設定允許，maintenance 會按新增結算場數觸發每日再訓練

預設推送時間（HKT）：

1. 賽前通知：每場開跑前 30 分鐘（`PRE_RACE_NOTIFY_MINS=30`）
2. 賽後結果處理：20:30（記錄結果與回測）
3. 每日總結：21:00
4. maintenance 摘要：21:50 起的視窗內（預設 10 分鐘，只發一次）

若使用 `--mode cron`（單一 command）並每 5 分鐘觸發：

1. 每次都會跑 tick（只在賽前視窗內發賽前訊息）
2. 到 maintenance 視窗才會跑 maintenance，並按設定決定是否推送

【方案B：Railway 可設多條 command】

1. 每5分鐘（賽日）: `python run.py --mode tick`
2. 每日賽後維護摘要: `python run.py --mode maintenance`

防止 tick 重啟後重複推送：

1. 建立 Railway Volume 並掛載到 `/data`
2. 設定 `TICK_SENT_STATE_FILE=/data/tick_notified.json`
3. 這樣重啟/重新部署後仍可保留當日已推送記錄

```bash
# 使用 gunicorn 生產服務
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 "web.app:create_app()"
```

**注意**：HKJC 網站可能有反爬蟲措施，生產環境需考慮：
- 設定合理的請求間隔
- 使用住宅代理或在香港境內部署
- 遵守 HKJC 網站使用條款

---

## 免責聲明

> **本系統僅供學術研究及技術展示用途。所有預測結果不構成任何投注建議。**
>
> - 本系統與香港賽馬會（HKJC）並無任何關連
> - 賽馬投注涉及財務風險，請量力而為
> - 本系統不對任何因使用本系統而產生的損失負責
> - 香港法例規定，非法賭博屬違法行為
>
> **Responsible Gambling**: 如有需要，請致電賭博輔導熱線 1834 633

---

## 授權條款

MIT License - 僅供研究用途
