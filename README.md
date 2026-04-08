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

# 一鍵診斷（適合排程/部署 log 排障）
python run.py --mode debug

# 一次性輪詢（適合外部 Cron 每 5 分鐘）
python run.py --mode tick

# 單一 cron command（平台僅可設一條排程指令時）
python run.py --mode cron

# 每日維護（回測 + 優化 + Telegram 摘要）
python run.py --mode maintenance

# 套用資料留存政策（清理超過保留期的 artifact）
python run.py --mode retention-cleanup

# 一次性歷史來源溯源遷移（C/D + 原因）
python run.py --mode migrate-provenance

# 立即強制發送全場賽前預測到 Telegram
python run.py --mode send-all-previews-now

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

### GitHub Actions 排程（推薦）

本專案的正確入口是 `run.py`，不是 `src.main`。

已內建 workflow：`.github/workflows/hkjc-cron.yml`

資料來源策略（2026-04 更新）：

1. 即時賠率：優先 `https://info.cld.hkjc.com/graphql/base/`（必要時由 Playwright 截取頁面同源 GraphQL 請求）。
2. 賽後結果：優先 GraphQL，若遇 `WHITELIST_ERROR` 或欄位變更，會自動回退到 `localresults` HTML 解析。
    - 主要 URL：`https://racing.hkjc.com/zh-hk/local/information/localresults`
    - 後備 URL：`https://racing.hkjc.com/en-us/local/information/localresults`
3. 建議保留一次成功 HAR（需包含 `entries` 與 response body）作為 schema 變更時的校驗樣本。

此 workflow 最適合目前架構，原因：

1. 直接使用 `python run.py --mode cron`，不需改動既有排程邏輯。
2. 每 5 分鐘自動執行一次（HKT 09:00-22:59），同時涵蓋 tick 與 maintenance 視窗。
3. 內建 cache 保留 `tick_notified.json` 與 `cron_state.json`，避免重複推送。
4. 支援手動觸發並切換 mode（`cron/tick/maintenance/backtest/optimize/debug/telegram-test`）。

GitHub Actions 設定步驟：

1. Push 本專案到 GitHub。
2. 到 repository 的 **Settings → Secrets and variables → Actions**。
3. 建立 Secrets：
    - `TELEGRAM_TOKEN`
    - `TELEGRAM_CHAT_ID`
4. 建立 Variables（可選，未設定會用預設值）：
    - `MESSAGE_STYLE=pro`（`pro` 專業 / `casual` 口語）
    - `TELEGRAM_LANGUAGE_GUARD=true`（偵測到英文馬名即阻擋 Telegram 發送）
    - `DEMO_MODE=false`
    - `REAL_DATA_ONLY=true`
    - `ALLOW_SYNTHETIC_TRAINING=false`
    - `FLASK_DEBUG=false`
    - `CRON_MAINTENANCE_HOUR=21`
    - `CRON_MAINTENANCE_MINUTE=50`
    - `CRON_MAINTENANCE_WINDOW_MINS=10`
    - `MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED=false`（建議；確保每日 maintenance 摘要固定推送）
    - `DAILY_RETRAIN_ENABLED=true`
    - `TELEGRAM_ALERT_ON_FAILURE=true`（cron/maintenance 致命錯誤即時告警）
    - `TELEGRAM_ALERT_COOLDOWN_MINS=30`（同一錯誤告警冷卻時間，分鐘）
    - `TELEGRAM_ALERT_STATE_FILE=data/predictions/alert_state.json`（告警去重狀態檔）
    - `ENABLE_DAILY_ALERT_NOISE_SUMMARY=true`（maintenance 推送每日抑制噪音摘要）
    - `ALERT_NOISE_TOP_N=5`（每日噪音摘要最多列出幾個 fingerprint）
    - `TRAINING_RETENTION_MANIFEST_FILE=data/training_retention_manifest.json`
5. 到 **Actions** 頁面啟用 workflow。

### 一次部署，之後全自動（建議配置）

目標：只需部署一次，之後由 GitHub Actions 自動按時執行 `tick + maintenance + 報告推送`。

1. Secrets 最少要有：`TELEGRAM_TOKEN` 與 `TELEGRAM_CHAT_ID`（若你目前只用舊名 `TELEGRAM_BOT_TOKEN`，請同值再新增一個 `TELEGRAM_TOKEN`）。
2. Variables 建議設定：`MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED=false`（確保每日 maintenance 摘要固定推送，不會因為無新結算賽果而跳過）。
3. 已內建每日 heartbeat（預設 12:00 HKT）；如要改時間，可直接修改 workflow 內 `HEARTBEAT_*` env。
4. Push 後 workflow 會在 HKT 09:00-22:59 每 5 分鐘自動執行 `cron`。
5. `cron` 會自動執行賽前推送（tick）、每日 heartbeat（固定時段一次）以及 maintenance（視窗內一次）。
6. 你只需要看 Telegram 與 Actions log，不需每天手動執行 command。

GitHub Actions 手動模式說明：

1. `maintenance`：發送每日維護摘要（受 `MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED` 影響）。
2. `backtest`：執行回測並發送「回測模式摘要」到 Telegram。
3. `optimize`：執行優化並發送「優化模式摘要」到 Telegram（若無已結算真實賽果，也會發送未執行原因）。

GitHub Actions 常見陷阱（收不到 Telegram）：

1. Secret 名稱必須是 `TELEGRAM_TOKEN` 與 `TELEGRAM_CHAT_ID`。
2. Workflow 讀取的是 `TELEGRAM_TOKEN`；若你仍使用舊名 `TELEGRAM_BOT_TOKEN`，請同步建立 `TELEGRAM_TOKEN`（值相同）。
3. `maintenance` 在無新結算賽果時可能會跳過通知（可把 `MAINTENANCE_NOTIFY_ONLY_ON_NEW_SETTLED` 設為 `false` 以強制每日發送）。

預設推送時間（HKT）：

1. 賽前通知：每場開跑前 30 分鐘（`PRE_RACE_NOTIFY_MINS=30`）
2. maintenance 摘要：21:50 起的視窗內（預設 10 分鐘，只發一次）
3. heartbeat 健康檢查：12:00 起的視窗內（預設 10 分鐘，只發一次）

maintenance 輸出（`data/reports/`）：

1. `maintenance_YYYY-MM-DD.json`：當日回測摘要、優化結果、再訓練結果、最近賽日預測 vs 實際對照
2. `walkforward_YYYY-MM-DD.json`：walk-forward 滾動驗證報告（訓練窗 / 測試窗）
3. `authenticity_audit_YYYY-MM-DD.json`：每場資料來源溯源與可信等級（A/B/C/D）
4. `provenance_migration_YYYY-MM-DD.json`：一次性歷史來源回填遷移結果（C/D 分佈與規則）

資料留存政策（90 天 artifact）：

1. 設定檔：`data/retention_policy.json`
2. 預設啟用於 `maintenance`（可用 `ENABLE_RETENTION_CLEANUP=false` 關閉）
3. 亦可手動執行：`python run.py --mode retention-cleanup`
4. 只清理報告與 debug artifact，不會刪除 `data/models/` 與 `data/predictions/history.json`

訓練資料保留清單（machine-readable）：

1. 設定檔：`data/training_retention_manifest.json`
2. maintenance 會自動檢查：必要檔案是否存在、A/B/C/D 分級數量、是否達到最小可訓練樣本門檻
3. 預設只允許 A/B（高品質來源）進入每日增量再訓練，C/D 保留於歷史但不進訓練集
4. 檢查結果會寫入 `maintenance_YYYY-MM-DD.json` 的 `training_retention` 與 `retrain.quality_gate`

失敗自動 Telegram 通知：

1. 啟用 `TELEGRAM_ALERT_ON_FAILURE=true` 後，`run.py --mode cron` 或其他模式若發生致命錯誤，會自動推送告警
2. 告警內容包含 mode、錯誤類型與精簡 traceback，方便你直接從通知定位問題
3. 已內建「同錯誤去重 + 冷卻時間」：以 mode + exception 類型 + message 產生 fingerprint，在 `TELEGRAM_ALERT_COOLDOWN_MINS` 內相同 fingerprint 不重複推送
4. 去重狀態會持久化到 `TELEGRAM_ALERT_STATE_FILE`，即使程序重啟仍可避免短時間重複轟炸
5. 冷卻結束後下一次告警會附上 `suppressed_since_last_sent` 與抑制時間區間，方便觀察告警噪音量
6. maintenance 會主動推送「每日抑制統計摘要」（top noisy fingerprints），即使當日無新結算賽果，只要有噪音抑制也會推送

策略檔（`data/models/`）：

1. `strategy_profile.json`：全局門檻（min_confidence / max_win_odds）
2. `strategy_profile_segments.json`：分場地/分距離門檻（ST/HV + sprint/mile/middle/long）

若你只想手動測試 Telegram，可在 Actions 手動執行並選 `telegram-test`。

---

## 免責聲明（法律與合規）

1. 本專案為個人研究與技術驗證用途，非香港賽馬會官方產品、服務或合作項目。
2. 本專案不提供投注建議、投資建議或任何形式的獲利保證，所有輸出僅供研究參考。
3. 使用者需自行確認並遵守香港賽馬會網站條款、robots 規範、適用法律與平台政策；如有疑義，應先停止使用並諮詢法律專業意見。
4. 專案內資料可能來自公開頁面或介面，資料正確性、完整性與即時性不作任何明示或默示保證。
5. 因使用、誤用或依賴本專案輸出所造成之任何損失、責任或爭議，開發者與貢獻者不承擔責任。
6. 若資料來源方提出限制、下架或停止要求，應立即停用相關抓取、儲存與傳播流程。
7. 本專案不鼓勵或引導任何非法賭博行為，使用者須自行遵守香港及所在地法例。
8. 理性投注及負責任博彩：如有需要，請致電賭博輔導熱線 1834 633。

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

## 授權條款

MIT License - 僅供研究用途
