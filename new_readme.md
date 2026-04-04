# HKJC Predictor 實戰操作手冊

本文件給你一套可直接操作的流程：
1. 寫入最近 100 場真實賽果到 backtest 歷史
2. 修正既有錯誤賽後結果
3. 重新回測與優化策略
4. 說明 Telegram 在系統中的實際用途
5. **每次即時投注前的標準操作清單**

---

## 零、即時數據模組狀態（已確認可用）

| 模組 | 端點 | 狀態 |
|------|------|------|
| 賽事表 (racecard) | racing.hkjc.com/local/information/racecard | ✅ 正常 |
| 場地狀況 (going) | racing.hkjc.com/local/info/windtracker → 每場HTML再確認 | ✅ 修正後 |
| 即時賠率 (odds) | Playwright + GraphQL (bet.hkjc.com) | ✅ 正常 |
| 賽果 (results) | racing.hkjc.com/local/information/localresults | ✅ 已修正 |
| 天氣 (weather) | HKO opendata API（中文站名 沙田/跑馬地） | ✅ 修正後 |
| 馬匹檔案 (horse) | racing.hkjc.com/local/information/horse | ✅ 正常 |

**本次修正摘要（原不準的原因）**：
1. Results URL 舊了：`/local/racing/results` (404) → 改為 `/local/information/localresults`
2. Results 表格解析太寬鬆：可能吃錯 table → 現在只解析有 Pla/No/Horse 標題的主結果表
3. 天氣溫度永遠 22°C：HKO API 站名是中文（「沙田」「跑馬地」）而非英文 → 已修正，現在取到實際溫度
4. 天氣 going 用降雨估算而非官方 → 現在先取 HKJC windtracker 官方值，再以 HTML per-race 值確認

---

## 一、每次即時投注前的標準操作清單

> 以下在賽事日當天，**開跑前 2 小時**按順序操作。

### 步驟 A：確認最新賽果已回填（每週補充一次即可）

    cd "c:\Users\lccqs\Racing-predictor"
    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 160

看到「已寫入 history.json」表示完成。

---

### 步驟 B：重新優化策略門檻

    .\.venv311\Scripts\python.exe run.py --mode optimize

輸出中確認：
- 近30日真實 ROI 是正數
- 優化後最小信心值 ≥ 0.55
- 策略檔案：data/models/strategy_profile.json

---

### 步驟 C：啟動系統（Web + Scheduler）

    .\.venv311\Scripts\python.exe run.py

保持此終端開著，不要關閉。

---

### 步驟 D：瀏覽器開啟儀表板確認賽事表已載入

    http://localhost:5000

確認事項：
- [ ] 今日賽事表已顯示（場次、馬匹、賠率）
- [ ] 場地狀況（Going）欄位顯示正確（GOOD / GOOD TO YIELDING 等）
- [ ] 沙田溫度 ≠ 22.0°C（如果是 22.0 表示天氣模組出問題）

---

### 步驟 E：手動觸發預測（開賽前確認最新賠率）

在系統已啟動前提下：

    .\.venv311\Scripts\python.exe run.py --mode predict

輸出頭三建議格式：
```
第N場 | 沙田 | 1200m | Class 3 | GOOD | 14:00
  信心度: 72%
  頭3名預測:
    1. #5 GOLDEN ARROW  模型:38.4%  賠率:5.5  EV:+$11.20 ★值博
    2. #2 SUPER STAR    模型:27.1%  賠率:8.0  EV:+$7.68
    3. #9 FLASH KING    模型:19.3%  賠率:12.0 EV:+$3.16
  三T首選: #5/2/9
```

---

### 步驟 F：投注前核對清單

在投注 HKJC 之前逐項確認：

| 項目 | 確認方式 | 通過條件 |
|------|----------|----------|
| 信心度 | 看 predict 輸出 | 單場 ≥ 策略門檻 (optimize 給出的 min_confidence) |
| ★值博標記 | 看 predict 輸出 | 至少首選馬有 ★ 標記 |
| 場地狀況 | Web 儀表板 / predict 輸出 | Going 與 HKJC 官網一致 |
| 賠率是否更新 | predict 輸出中賠率 | 賠率 ≠ 0 且與 HKJC app 相差不超過 ±0.5 |
| 連輸狀態 | Web 儀表板風控區 | 連輸未到達門檻（預設連輸 3 場停） |

---

### 步驟 G：賽後更新（20:30 後系統自動完成）

排程器自動執行：
- `record_results_job` → 抓官方賽果寫入 history.json
- `optimize_strategy_job` → 更新策略門檻

**確認方法**（系統啟動後看 log）：
```
[INFO] record_results_job ===
[INFO] Recorded real result for 2026-04-06_ST_1: [3, 7, 1...]
[INFO] Strategy profile active: min_conf=0.70 max_odds=18.0 ...
```

---

## 二、你現在要用的兩個腳本

- **backfill_recent_real100.py**
  - 功能：從 HKJC Results 頁回填最近 N 場真實賽果
  - 會把 is_real_result 寫成 True
  - 若 race_id 已存在，會更新 actual_result，修正錯誤賽果

- **seed_real_history.py**
  - 功能：寫入示例資料（非官方逐場抓取）
  - 建議只在離線測試時使用，不建議當作正式對照資料

---

## 三、一步步操作（回填與優化）

### Step 0. 進到專案目錄

    cd "c:\Users\lccqs\Racing-predictor"

### Step 1. 先預覽最近 100 場（不寫檔）

    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 160 --dry-run

### Step 2. 正式寫入最近 100 場真實數據

    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 160

### Step 3. 執行 backtest 檢查結果

    .\.venv311\Scripts\python.exe run.py --mode backtest

### Step 4. 生成新策略檔（優化門檻）

    .\.venv311\Scripts\python.exe run.py --mode optimize

### Step 5. 啟動 web + scheduler 進入日常運行

    .\.venv311\Scripts\python.exe run.py

---

## 四、Telegram 對整個系統有沒有用？

有用，但屬於「通知層」，不是「核心預測層」。

| 功能 | 有 Telegram | 無 Telegram |
|------|-------------|-------------|
| 模型預測/回測/優化 | ✅ 正常 | ✅ 正常 |
| Web 儀表板 | ✅ 正常 | ✅ 正常 |
| 賽前推送（頭三建議） | ✅ 自動推送 | ❌ 沒有 |
| 每日日報（ROI/命中率） | ✅ 自動推送 | ❌ 沒有 |
| 風控提醒（連輸/日虧損） | ✅ 自動警示 | ❌ 沒有 |

**結論**：
- 想要手機即時收到建議 → 建議開 Telegram
- 只在電腦看 http://localhost:5000 → 不需要

**開啟 Telegram 方式**（需要 .env 設定，注意變數名）：

    TELEGRAM_TOKEN=你的Bot_Token
    TELEGRAM_CHAT_ID=你的Chat_ID

**正確取得 chat_id 的步驟（建議用這個）**：
1. 在 Telegram 打開你的 bot，先送一則 `/start`（沒有這步，`getUpdates` 會是空）
2. 在專案目錄執行：

       .\.venv311\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv('.env'); import os,requests; t=os.getenv('TELEGRAM_TOKEN'); r=requests.get(f'https://api.telegram.org/bot{t}/getUpdates',timeout=15).json(); print(r)"

3. 在輸出 JSON 找 `chat.id`（通常是數字，例如 `123456789`，群組常見負數 `-100...`）
4. 把 `.env` 的 `TELEGRAM_CHAT_ID` 改成該數字

**發送測試訊息**：

    .\.venv311\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv('.env'); import sys; sys.path.insert(0,'.'); from notifier.telegram import TelegramNotifier; n=TelegramNotifier(); print('configured=',n.is_configured()); print('send_ok=', n.send_sync('✅ Telegram 測試成功'))"

**一鍵測試三種模板（賽前 + 賽後 + 每日總結）**：

    .\.venv311\Scripts\python.exe run.py --mode telegram-test

若看到 `Chat not found`：
- 表示 token 多半是對的，但 chat_id 不對
- 先確定你已對 bot 發 `/start`
- 若發到群組，先把 bot 加進群組，並給發訊權限

---

## 五、常見排錯

### 1) 回填抓不到 100 場

    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 260

### 2) optimize 顯示無真實賽果

先跑回填再跑 optimize：

    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 160
    .\.venv311\Scripts\python.exe run.py --mode optimize

### 3) 賠率顯示為 0 或 mock
- 確認現在是賽事日且賭注已開放（一般開賽前 2 小時）
- Playwright 需要 chromium：

    .\.venv311\Scripts\playwright.exe install chromium

### 4) 天氣溫度顯示 22.0°C（固定值）

表示 HKO API 站名匹配失敗。檢查網路是否能訪問：

    .\.venv311\Scripts\python.exe -c "import requests; print(requests.get('https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=tc').status_code)"

應回傳 200。

### 5) 場地 Going 不準
- `fetch_going` 會讀 windtracker 頁面，此頁面在非賽事日可能不顯示 going
- 在賽事日才能取得精確 going；非賽事日預設 "GOOD"

---

## 六、建議日常流程（最短版）

**每週一次（補充最新真實資料）**

    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 260
    .\.venv311\Scripts\python.exe run.py --mode optimize

**賽事日當天（開賽前 2 小時）**

    .\.venv311\Scripts\python.exe run.py              ← 啟動後不要關
    # 確認儀表板 http://localhost:5000 顯示正確賽事
    .\.venv311\Scripts\python.exe run.py --mode predict   ← 另開終端執行


- backfill_recent_real100.py
  - 功能：從 HKJC Results 頁回填最近 N 場真實賽果
  - 會把 is_real_result 寫成 True
  - 若 race_id 已存在，會更新 actual_result，修正錯誤賽果

- seed_real_history.py
  - 功能：寫入示例資料（非官方逐場抓取）
  - 建議只在離線測試時使用，不建議當作正式對照資料

---

## 二、一步步操作（建議照順序）

### Step 0. 進到專案目錄

    cd "c:\Users\lccqs\Racing-predictor"

### Step 1. 先預覽最近 100 場（不寫檔）

    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 160 --dry-run

你會看到：
- 掃描了多少賽事日
- 抓到多少場
- 本次目標多少場
- 若能達到 100 場，下一步就可正式寫入

### Step 2. 正式寫入最近 100 場真實數據

    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 160

這一步會：
- 新增缺少的賽事記錄
- 更新已有但不正確的 actual_result
- 統一標記 is_real_result=True

### Step 3. 執行 backtest 檢查結果

    .\.venv311\Scripts\python.exe run.py --mode backtest

你會看到：
- 總預測場數
- 真實數據場數
- 近 7 日 / 30 日 ROI

### Step 4. 生成新策略檔（優化門檻）

    .\.venv311\Scripts\python.exe run.py --mode optimize

策略會寫到：
- data/models/strategy_profile.json

### Step 5. 啟動 web + scheduler 進入日常運行

    .\.venv311\Scripts\python.exe run.py

排程會在賽後自動：
- record_results_job 寫入真實賽果
- optimize_strategy_job 更新策略門檻

---

## 三、你說的「對照 HKJC 發現賽後結果不對」原因

本次已修正兩個關鍵點：

1. Results 網址舊了
- 舊網址是 /local/racing/results（已失效，會 404）
- 正確網址是 /local/information/localresults

2. 表格解析太寬鬆
- 以前會掃到不該掃的 table，導致名次可能錯位
- 現在改成只解析有 Pla/No/Horse 標題的主結果表，並依名次排序後寫入

---

## 四、Telegram 對整個系統有沒有用？

有用，但屬於「通知層」，不是「核心預測層」。

有 Telegram 時可做：
- 賽前通知（推送頭三建議、值博提示）
- 每日總結（命中率、ROI、風控狀態）
- 風險提醒（連輸與日虧損觸發）

沒有 Telegram 時：
- 模型訓練、預測、回測、優化、排程都照常運作
- 只是不會即時收到推送

結論：
- 想要即時監控和提醒，建議開 Telegram
- 只在本機看 Web 儀表板，Telegram 可不開

---

## 五、常見排錯

### 1) 回填抓不到 100 場
- 提高掃描天數：

    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 260

### 2) 某些 RaceNo 解析不到（例如 R10-R12）
- 這通常是該日場數不足或頁面無該場，不一定是錯誤
- 只要能累積到 target 場數即可

### 3) optimize 顯示無真實賽果
- 先確認 backfill 已成功寫入
- 再跑一次 backtest 看真實數據場數是否大於 0

---

## 六、建議日常流程（最短版）

1. 每週補一次最近真實資料

    .\.venv311\Scripts\python.exe backfill_recent_real100.py --target 100 --max-days 260

2. 跑回測

    .\.venv311\Scripts\python.exe run.py --mode backtest

3. 跑優化

    .\.venv311\Scripts\python.exe run.py --mode optimize

4. 啟動 web/scheduler

    .\.venv311\Scripts\python.exe run.py
