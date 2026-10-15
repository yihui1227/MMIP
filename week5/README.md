# MMIP - Week 5

本專案包含 `main.ipynb`，依序完成 Quiz 1 ~ Quiz 3(含進階題)

## 1. 專案結構

```
.
├── main.ipynb          # 主要作業 Notebook
├── requirements.txt    # 套件清單
├── README.md
├── data/               # (自動產生) kagglehub 快取、特徵快取
└── outputs/            # (自動產生) 結果表格與比較圖
```

## 2. 環境安裝（NVIDIA RTX 3060 / CUDA）

建議使用 Python 3.11。請先安裝 CUDA 版 PyTorch，再安裝其餘套件。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

## 3. 使用資料集（皆以 kagglehub 下載）

| 用途 | Kaggle 資料集 | 說明 |
|---|---|---|
| Quiz 1 影像檢索 | `alessiocorrado99/animals10` | 10 類動物，每類取樣 400 張，共 4,000 張 |
| Quiz 2 多模態分類 | `adityajn105/flickr8k` | 影像及描述；依描述關鍵字建立 5 類場景標籤 |
| Quiz 3 Image Captioning | `adityajn105/flickr8k` | 每張影像有 5 句描述 |

## 4. 作業項目對照

| Quiz | 項目 | Notebook 章節 |
|---|---|---|
| 1 | Animals-10 影像檢索、訓練與推論 | 1.1 ~ 1.6 |
| 1 | ResNet-50 與 ViT-B/16 比較 | 1.4、1.6 |
| 2 | 影像與文字特徵融合分類 | 2.1 ~ 2.4 |
| 2 | 更換 Backbone、模態消融比較 | 2.5 ~ 2.7 |
| 3 | CLIP + GPT-2 影像描述生成 | 3.1 ~ 3.5 |
| 3 | BLEU / BERTScore 及 Week 4 方法比較 | 3.6 ~ 3.7 |

## 5. 實驗結果

**實驗環境：** NVIDIA RTX 3060、PyTorch CUDA，隨機種子固定為 42。

### Quiz 1：影像檢索（Animals-10）

資料集共 4,000 張影像，每類 400 張；訓練集 2,800 張，檢索評估集 1,200 張。下表列出未微調的 ImageNet 特徵與微調後的結果。

| Backbone | 微調 | Recall@1 | Recall@5 | Recall@10 | mAP | 參數量 (M) | 訓練時間 (s) | 推論時間 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 否 | 0.9650 | 0.9892 | 0.9917 | 0.8257 | — | — | — |
| ResNet-50 | 是 | 0.9650 | 0.9825 | 0.9858 | 0.9423 | 24.04 | 69.18 | 3.33 |
| ViT-B/16 | 否 | 0.9600 | 0.9825 | 0.9850 | 0.8274 | — | — | — |
| ViT-B/16 | 是 | 0.9675 | 0.9850 | 0.9875 | **0.9552** | 86.00 | 140.45 | 4.86 |

#### 實驗結果分析

- 微調後兩種模型的 mAP 均明顯提升；ViT-B/16 的 mAP 最高，為 0.9552。
- ResNet-50 訓練與推論較快；ViT 的檢索表現稍好，但模型較大、執行時間較長。
- 相關影像以類別標籤相同定義，結果受每類取樣數及單一隨機種子限制。

**小結：** 微調有助於提升檢索排序品質；若重視檢索效能可選 ViT，若重視速度與資源效率則 ResNet-50 具有優勢。

### Quiz 2：多模態分類（Flickr8k 場景標籤）

依描述關鍵字建立 dog、water、ball_sport、climbing、street 五類標籤，共使用 2,498 張影像；訓練、驗證、測試分別為 1,748、375、375 張。影像與文字 Backbone 凍結，只訓練融合分類器。

| 影像 Backbone + 文字 Backbone | 測試 Accuracy | Macro-F1 | 分類器參數量 (M) | 訓練時間 (s) |
|---|---:|---:|---:|---:|
| ResNet-50 + DistilBERT | 0.9067 | 0.9068 | 1.9702 | 1.8890 |
| ResNet-50 + MiniLM | 0.8971 | 0.8976 | 1.7736 | 1.5670 |
| ViT-B/16 + DistilBERT | 0.9147 | 0.9149 | 1.3148 | 1.3745 |
| **ViT-B/16 + MiniLM** | **0.9168** | **0.9169** | **1.1182** | **1.3309** |

模態消融的測試 Macro-F1：

| 影像 + 文字 Backbone | Fusion | Image-only | Text-only |
|---|---:|---:|---:|
| ResNet-50 + DistilBERT | 0.9068 | 0.8716 | 0.8951 |
| ResNet-50 + MiniLM | 0.8976 | 0.8662 | **0.9141** |
| ViT-B/16 + DistilBERT | 0.9149 | 0.8665 | 0.8861 |
| ViT-B/16 + MiniLM | **0.9169** | 0.8583 | 0.9054 |

#### 實驗結果分析

- ViT-B/16 + MiniLM 的測試 Accuracy 與 Macro-F1 最高。
- 多數組合中融合模型優於單一模態；ResNet-50 + MiniLM 則是 Text-only 高於 Fusion。
- 標籤由描述文字關鍵字產生，因此文字模態帶有直接標籤線索；這會影響模態比較，結果需審慎解讀。

**小結：** 本次實驗中 ViT-B/16 + MiniLM 的融合結果最佳，但文字標籤關聯可能使 Text-only 得到偏高表現。後續可用人工標註標籤及多個隨機種子驗證。

### Quiz 3：Image Captioning（CLIP + GPT-2）

Week 5 測試集共 1,091 張影像。下表比較各方法在 100 張影像上的結果；Week 4 的方法數據取自其實驗紀錄。

| 方法 | 評估張數 (N) | 回覆時間 (s/張) | BLEU-1 | BLEU-4 | BERTScore-F1 | 參數量 (M) |
|---|---:|---:|---:|---:|---:|---:|
| Week 4 LSTM-Greedy | 100 | 0.0062 | 0.5651 | 0.1809 | 0.8310 | 4.9894 |
| Week 4 LSTM-Beam (k=3) | 100 | 0.0176 | 0.5275 | 0.1518 | 0.8359 | 4.9894 |
| Week 4 BLIP-pretrained | 100 | 0.1777 | 0.5850 | 0.2427 | 0.8488 | — |
| **CLIP+GPT2-Greedy** | 100 | 0.0407 | **0.6571** | **0.2504** | 0.9398 | 113.3824 |
| CLIP+GPT2-Beam (k=3) | 100 | 0.0518 | 0.6484 | 0.2446 | **0.9399** | 113.3824 |

Week 5 CLIP+GPT2 在完整 1,091 張測試集上的結果：

| 方法 | BLEU-1 | BLEU-4 | BERTScore-F1 |
|---|---:|---:|---:|
| CLIP+GPT2-Greedy | 0.6522 | 0.2328 | 0.9376 |
| CLIP+GPT2-Beam (k=3) | 0.6752 | 0.2466 | 0.9391 |

#### 實驗結果分析

- 在 100 張比較中，CLIP+GPT2-Greedy 的 BLEU-1、BLEU-4 最高；Beam 的 BERTScore-F1 略高，差距僅 0.0001。
- Beam 推論時間較 Greedy 長；完整測試集上 Beam 的 BLEU-1、BLEU-4 較高，但 BERTScore-F1 略低。
- 100 張比較中的 Week 4 與 Week 5 方法可能使用不同測試影像切分，分數不宜視為完全公平的直接比較。

**小結：** CLIP+GPT2 在本次指標中呈現較高的語意相似度與文字重疊分數；Greedy 較快，Beam 在完整測試集上的 BLEU 較高。不同切分、評估指標與模型規模均須納入解讀。

## 6. 綜合結論

1. **影像檢索：** 微調提升了 ResNet-50 與 ViT 的 mAP；ViT 表現較佳，ResNet-50 執行較快。
2. **多模態分類：** ViT-B/16 + MiniLM 的融合分類最佳，但由於標籤來自描述關鍵字，文字輸入與標籤高度相關。
3. **影像描述：** CLIP+GPT2 的 BLEU 與 BERTScore 表現佳；Greedy 與 Beam 在速度及指標間各有取捨。

**整體而言，模型選擇需依任務需求，在品質、計算成本及推論效率間取得平衡。**
