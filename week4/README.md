# MMIP - w4

本專案包含 `main.ipynb`，依序完成 Q1 ~ Q4(含進階題)

## 1. 專案結構

```
.
├── main.ipynb                  # 主要作業 Notebook
├── requirements.txt            # 套件清單
└── README.md
```

## 2. 環境安裝（個人環境：NVIDIA RTX 3060 / CUDA）

建議 Python 3.10+。**重點：要先裝 CUDA 版 PyTorch**，否則 Windows 預設會裝成 CPU 版，GPU 不會被使用

```bash
# 1) 確認驅動正常（應顯示 RTX 3060）
nvidia-smi

# 2) 建立並啟用虛擬環境
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux：source .venv/bin/activate

# 3) 安裝 CUDA 版 PyTorch（cu126 為範例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4) 安裝其餘套件
pip install -r requirements.txt

# 5) 驗證 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 6) 啟動
jupyter notebook main.ipynb
```

## 3. 設定 API Key
### Gemini（Quiz 4）
至 [Google AI Studio](https://aistudio.google.com/apikey) 申請 API Key，並在與 `main.ipynb` 同一層建立 `.env`：

```
GEMINI_API_KEY=你的key
```

Notebook 會以 `python-dotenv` 讀取（等號兩邊不要空格或引號）

## 4. 使用資料集（皆以 kagglehub 下載）

| 用途 | Kaggle 資料集 | 說明 |
|---|---|---|
| Sentiment Analysis | `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews` | 影評正/負二分類 |
| Image Classification | `alessiocorrado99/animals10` | 10 類動物 |
| Image Captioning | `adityajn105/flickr8k` | 影像 + 每張 5 句描述 |

## 5. 作業項目對照

| Quiz | 項目 | Notebook 章節 |
|---|---|---|
| 1 | 文字 / 影像 / 影像-文字 資料集準備 | 1.1 ~ 1.3 |
| 2 | RNN 訓練與測試集推論 | 2.1 ~ 2.2 |
| 2 進階 | LSTM 訓練；RNN vs LSTM 指標比較（Acc / P / R / F1 / ROC-AUC） | 2.3 ~ 2.4 |
| 3 | 微調 ViT-B/16 與推論 | 3.1 |
| 3 進階 | 微調 ResNet-50；Macro-AUC 比較 | 3.2 ~ 3.3 |
| 4 | Captioning 模型訓練與產生描述；Gemini 判斷 | 4.1 ~ 4.6 |
| 4 進階 | BLEU / BERTScore；方法回覆時間與表現比較 | 4.7 ~ 4.8 |

## 6. 實驗結果

**實驗環境：** NVIDIA RTX 3060、PyTorch（CUDA），隨機種子固定為 42。以下呈現各項實驗的實際測試結果。

### Quiz 2：RNN vs LSTM（IMDB 影評情感分類）

測試集共 1,500 筆資料，其中 negative 類別 744 筆、positive 類別 756 筆。<br>
以下比較兩種循環神經網路在情感分類任務上的表現。

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | 參數量 | 訓練時間 (s) | 推論時間 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RNN | 0.6453 | 0.6349 | 0.6971 | 0.6646 | 0.6854 | 2,593,282 | 6.8378 | 0.1017 |
| **LSTM** | **0.7567** | **0.7727** | **0.7328** | **0.7522** | **0.8181** | 2,692,354 | 7.4621 | 0.1056 |


### 實驗結果分析

- **整體分類表現：** <br>LSTM 的 Accuracy 為 0.7567，高於 RNN 的 0.6453，差距為 11.14 個百分點；F1 也由 0.6646 提升至 0.7522，顯示 LSTM 在此任務中具有較佳的分類表現
- **類別辨識能力：** 
    - RNN 對 positive 類別的 Recall 為 0.6971，但對 negative 類別僅為 0.5927
    - LSTM 的 negative 與 positive Recall 分別為 0.7809 與 0.7328，兩類的辨識表現較均衡
- **模型區辨能力：** <br>LSTM 的 ROC-AUC 為 0.8181，高於 RNN 的 0.6854，表示 LSTM 對正負面影評的區辨能力較佳
- **計算成本：** 
    - LSTM 的參數量約為 269 萬，比 RNN 多約 3.82%；訓練時間約增加 0.62 秒，推論時間約增加 0.004 秒
    - 相較於分類效能的提升，額外計算成本相對有限

**> 小結 <：** 
- 在本次 IMDB 情感分類實驗中，LSTM 在 Accuracy、F1 與 ROC-AUC 等指標上均優於簡單 RNN
- 這與 LSTM 利用閘門機制控制資訊保留與更新的設計相符，有助於處理長序列中的資訊
- 不過，本次結果只能說明 LSTM 在目前資料集與實驗設定下表現較佳，不能直接推論所有任務皆會有相同結果

### Quiz 3：ViT vs ResNet-50（Animals-10 影像分類）

本實驗使用 Animals-10 資料集，共 10 個動物類別，測試集共 300 張圖片，每類各 30 張。<br>比較 Vision Transformer（ViT-B/16）與 ResNet-50 的分類效能及計算成本。

| 模型 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Macro-AUC | 參數量 (M) | 訓練時間 (s) | 推論時間 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ViT-B/16** | **0.9633** | **0.9641** | **0.9633** | **0.9634** | **0.9991** | 85.8063 | 64.6057 | 0.8213 |
| ResNet-50 | 0.9567 | 0.9582 | 0.9567 | 0.9567 | 0.9989 | 23.5285 | 42.6727 | 0.4098 |

### 實驗結果分析

- **整體分類表現：** 
    - ViT 的 Accuracy 為 0.9633，ResNet-50 為 0.9567，差距約 0.66 個百分點
    - 以 300 張測試圖片計算，兩者分別正確分類 289 張與 287 張，僅相差 2 張。
- **各類別表現：** 
    - ViT 的各類別 F1 均達 0.9333 以上
    - ResNet-50 在 dog 類別的 F1 為 0.9032，在 spider 類別的 Recall 為 0.9000，為其相對較弱的表現
- **區辨能力：** <br>ViT 的 Macro-AUC 為 0.9991，ResNet-50 為 0.9989，兩者差距僅 0.0002，顯示兩種模型在本次測試集上的整體區辨能力皆相當高
- **模型規模與速度：** 
    - ResNet-50 約有 23.53M 個參數，約為 ViT 的 27.42%；訓練時間約少 21.93 秒，推論時間約少 0.41 秒
    - 若考量計算資源與執行效率，ResNet-50 具有明顯優勢

**> 小結 <：**  <br>
ViT-B/16 在本次實驗的各項整體指標上略優於 ResNet-50，但提升幅度有限；ResNet-50 則以較少的參數量及較短的訓練、推論時間，取得接近的分類效能。因此，若以本次測試結果 **評估資源使用效率** ，ResNet-50 是具競爭力的選擇。

不過值得注意的是，每類測試圖片僅有 30 張，少數圖片的分類結果便可能影響 Accuracy 與類別指標。因此，目前尚不能判定兩種模型的差異具有統計顯著性。後續可增加每類測試圖片數量、使用多個隨機種子重複實驗，並確認訓練與測試資料的切分方式，以提高比較結果的可靠性。

###  Quiz 4：Image Captioning 方法比較

本實驗比較四種影像描述方法，包括自訓練 LSTM、Beam Search 解碼、預訓練 BLIP，以及 Gemini zero-shot。<br>評估指標包含 BLEU-1、BLEU-4、BERTScore-F1、Gemini 判斷符合率與每張圖片的平均回覆時間。

| 方法 | 評估張數 (N) | 回覆時間 (s/張) | BLEU-1 | BLEU-4 | BERTScore-F1 | Gemini 判斷符合率 | 參數量 (M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM-Greedy | 100 | 0.0062 | 0.5651 | 0.1809 | 0.8310 | 0.1 | 4.9894 |
| LSTM-Beam (k=3) | 100 | 0.0176 | 0.5275 | 0.1518 | 0.8359 | 0.1 | 4.9894 |
| BLIP-pretrained | 100 | 0.1777 | 0.5850 | 0.2427 | 0.8488 | 0.5 | — |
| Gemini-zero-shot | 10 | 19.8624 | 0.7217 | 0.2113 | 0.8871 | 1.0 | — |

> 註：
>
> - Gemini 判斷符合率是依各方法前 10 張測試圖片計算，因此每張圖片會使符合率變動 0.1。
> - Gemini-zero-shot 僅評估 10 張圖片，其他方法則評估 100 張，兩者的結果不宜直接視為同等規模的比較。
> - 「—」表示未提供可直接比較的模型參數量。
> - 回覆時間為本次實驗記錄的數值，不一定等同於純模型運算延遲。

### 實驗結果分析

- **描述品質：** 
    - 在本次結果中，Gemini-zero-shot 的 BERTScore-F1 最高，為 0.8871；其次為 BLIP 的 0.8488、LSTM-Beam 的 0.8359，以及 LSTM-Greedy 的 0.8310
    - 這表示 Gemini 生成的描述與參考描述在語意上的相似度較高，但仍須考慮其測試樣本較少的限制
- **文字重疊指標：** 
    - Gemini-zero-shot 的 BLEU-1 最高，為 0.7217；但 BLEU-4 為 0.2113，低於 BLIP 的 0.2427
    - BLEU 指標著重生成文字與參考文字之間的 n-gram 重疊，因此不同指標可能呈現不同的模型排序
- **Gemini 判斷符合率：** 
    - Gemini-zero-shot 的符合率為 1.0，BLIP 為 0.5，而兩種 LSTM 方法均為 0.1
    - 這代表在本次前 10 張圖片的評估中，Gemini-zero-shot 的描述最符合所使用的判斷標準
    - 不過，此結果依賴評估方式與樣本內容，不能直接視為所有圖片上的實際正確率
- **Greedy 與 Beam Search：** 
    - Beam Search 的 BLEU-1 由 0.5651 降至 0.5275，BLEU-4 由 0.1809 降至 0.1518，但 BERTScore-F1 從 0.8310 略升至 0.8359。
    - 其平均回覆時間由 0.0062 秒增加至 0.0176 秒，約為 Greedy 的 2.84 倍。
    - 這表示在目前的小型 LSTM 模型與實驗設定下，Beam Search 並未在所有指標上帶來一致的改善
- **速度與部署成本：** 
    - LSTM-Greedy 的平均回覆時間最低，適合重視執行速度與本地部署的情境；BLIP 的平均回覆時間為 0.1777 秒，兼顧預訓練模型的描述能力與本地推論。
    - Gemini-zero-shot 的平均回覆時間為 19.8624 秒，明顯高於其他方法。
- **時間數據的限制：** 
    - Gemini 的回覆時間可能受到遠端 API 網路延遲、速率限制等待及重試機制影響，因此不能將 19.8624 秒直接解讀為模型本身的純推論時間。
    - 此外，Gemini 與其他方法的評估張數不同，時間比較也應保留此限制。

**小結：** 
- 本次實驗中，預訓練模型 BLIP 與 Gemini-zero-shot 的 BERTScore-F1 均高於自訓練 LSTM；其中 Gemini-zero-shot 的語意相似度與 Gemini 判斷符合率最高
- LSTM-Greedy 則具有明顯的速度優勢，顯示模型選擇需要依據描述品質、推論效率及部署需求進行取捨
- 由於 Gemini 的測試樣本較少，且遠端 API 的時間成本可能受到額外因素影響，後續應使用相同測試圖片重新評估各方法，並搭配人工檢視，以獲得更公平的比較結果

### 6.4 綜合結論

1. **文字情感分類：** LSTM 的 Accuracy、F1 與 ROC-AUC 均優於簡單 RNN，顯示其在本次 IMDB 影評分類任務中具有較佳的分類能力，而額外的參數與時間成本相對有限。

2. **影像分類：** ViT-B/16 的 Accuracy 略高於 ResNet-50，但差距僅約 0.66 個百分點。ResNet-50 使用較少參數，且訓練與推論速度較快，因此在本次實驗中具有較佳的計算效率。

3. **影像描述生成：** 預訓練模型 BLIP 與 Gemini-zero-shot 在 BERTScore-F1 上優於自訓練 LSTM；LSTM-Greedy 則具有最低的平均回覆時間。Beam Search 在目前設定下並未讓所有評估指標同步提升。

**總結而言，沒有單一模型能在所有任務中同時取得最佳品質與最低成本。** 本次結果顯示，模型的選擇應根據任務目標、資料規模、計算資源、推論速度與評估方式綜合判斷。
