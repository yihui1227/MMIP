# 作業二 Video Corpus（請放入自錄影片）

把自己錄製的影片放在這個資料夾，重新執行 `main.ipynb` 的作業二（2.1 ~ 2.6）與 3.5 即可。

## 需要的影片

| 類型 | 建議內容（衣物洗護主題） | 長度 |
|---|---|---|
| 有說話的影片（speech） | 邊示範邊口頭講解，例如「羊毛衣手洗步驟」、「如何看懂洗標符號」 | 1 ~ 3 分鐘 |
| 沒有說話的操作影片（silent） | 只有操作畫面，例如「洗衣機設定溫和洗程」、「襯衫摺疊」、「羽絨衣烘乾放網球」 | 30 秒 ~ 2 分鐘 |

- 支援格式：`.mp4 / .mov / .mkv / .avi / .webm / .m4v`
- 建議至少 2 支有說話、2 支無聲，讓檢索有比較對象。
- 影片可能拍到臉、聲音或住家環境，已在 `.gitignore` 排除，不會上傳到 Git。

## 選用檔案

### `videos.csv`（影片標題與類型，沒有也能跑）

```csv
video_id,file,title,kind
wool_handwash,wool_handwash.mp4,羊毛衣手洗步驟（口頭講解）,speech
gentle_cycle,gentle_cycle.mp4,洗衣機設定溫和洗程（無聲操作）,silent
```

`kind` 留空時，系統依音訊能量與 ASR 結果自動判斷 speech / silent。

### `demo_dialog.txt`（2.6 多輪示範，一行一句）

```text
羊毛衣要怎麼手洗？
那洗完要怎麼晾？
洗衣機的溫和洗程要怎麼設定？
```

### `video_benchmark.jsonl`（3.5 VideoRAG 評估）

格式見 `video_benchmark_template.jsonl`，每行一題：

| 欄位 | 說明 |
|---|---|
| `qid` | 題號 |
| `dialog_id` | 同一段多輪對話用相同 id（依檔案順序視為第 1、2、3 輪） |
| `question` | 使用者問題（可含代名詞，測多輪改寫） |
| `gold_video` | 正解影片的 `video_id` |
| `gold_segments` | 正解時段（秒），例如 `[[12, 25]]` |
| `answer` | 人工撰寫的標準答案 |
| `needs` | 回答需要的資訊：`visual` / `speech` / `both` |
