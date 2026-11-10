# Week 7

所有A、B、C基礎與進階的核心實作皆位於main.ipynb。使用Python 3.11 CUDA kernel，由本資料夾依序執行。

- `data/`：notebook自動產生的自訂文字題、偏好資料及幾何影像。
- `outputs/`：逐題實測、環境版本、LoRA adapter、評審資料、比較圖及report.md。
- 實驗會下載三個開源模型，訓練與完整評估需GPU及時間；固定原始／INT8／NF4條件。
- 商用評審只匯入真實回覆，不以規則分數冒稱；完成狀態與限制以outputs/report.md為準。

`requirements.txt`記錄此次環境的實際版本。PyTorch使用CUDA12.6 wheel，請先依本機驅動安裝合適GPU版本，再安裝其餘依賴。
