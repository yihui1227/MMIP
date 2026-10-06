# MMIP Week 3（基礎版）：動物影像分類

> 使用 Kaggle 的 **Animals Detection Images Dataset**（選用 50 個動物類別），完成影像分類作業的三個 Quiz 基礎題：準備資料集、訓練 CNN 影像分類模型、提升泛化能力（Data Augmentation）。

## 資料集

- **來源**：[Kaggle：antoreepjana/animals-detection-images-dataset](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset)，影像取自 Open Images，透過 `kagglehub` 自動下載。
- **原始規模**：29,071 張、80 個動物類別（官方 train 22,566 張、test 6,505 張）。
- **目錄結構**：`train/<類別>/xxx.jpg`、`test/<類別>/xxx.jpg`，每個類別資料夾內另有 `Label/`（物件偵測的標註）。本作業只做影像分類，不使用 `Label/`，也不使用 `ImageFolder`（它會把 `Label` 當成一個類別），而是自行掃描檔案建立索引。

### 資料檢查與清理

| 檢查項目 | 結果 | 處理方式 |
|---|---|---|
| 標籤衝突（同一張圖出現在多個類別） | 1,305 組、2,724 張 | 整組丟棄 |
| train / test 資料洩漏 | 0 組 | 無需處理 |
| 損毀影像 | 建立快取時未出現讀取失敗 | 無需處理 |
| 類別不平衡 | 訓練池每類 40 ～ 1,874 張（46.9 : 1） | 保留原始分佈 |

清理後剩 26,347 張，再以固定亂數種子（42）從訓練池張數 ≥ 30 的類別中選出 **50 個類別**：

- 訓練池（官方 train）：16,201 張，以 5 折 Stratified K-Fold 切分，使用 Fold 0（訓練 12,960 / 驗證 3,241）
- 測試集（官方 test）：4,141 張，完全不參與訓練與選擇 epoch

## 問題定義

單標籤、多類別影像分類。輸入為 224 × 224 的 RGB 動物影像，輸出為 50 個類別的 Softmax 機率，取最高者為預測類別。評估指標為 Top-1 / Top-5 Accuracy 與 Macro-AUC（One-vs-Rest）。

## 方法

| 項目 | Plain CNN | ResNet-18 |
|---|---|---|
| 架構 | 4 個區塊（Conv3×3-BN-ReLU × 2 → MaxPool），通道 32 → 64 → 128 → 256，GAP + Dropout 0.3 + Linear | torchvision ResNet-18，ImageNet 預訓練，`fc` 換成 50 類，微調整個網路 |
| 參數量 | 1,186,066 | 11,202,162 |
| 優化器 | AdamW，lr 1e-3，weight decay 1e-4 | AdamW，lr 3e-4，weight decay 1e-4 |
| 共通設定 | batch size 64、15 epochs、Cosine 學習率排程、混合精度（AMP）、以驗證 Top-1 最高的 epoch 作為最佳權重 | |

**資料擴增（Quiz 3，只作用在訓練集）**：RandomResizedCrop（面積 0.5 ～ 1.0）、水平翻轉（p = 0.5）、旋轉（±15°）、ColorJitter（亮度 / 對比 / 飽和度 ±0.3）、RandomErasing（p = 0.25）。擴增前後除此之外的條件完全相同。

**加速設計**：所有影像先一次性縮成 224 × 224 並快取成 `.npy`，訓練時整批資料放在 GPU 上，擴增也在 GPU 上完成。

## 實驗結果

### Quiz 2：Plain CNN vs ResNet-18（不使用擴增）

| 模型 | 參數量 | 驗證 Top-1 | 驗證 Top-5 | 測試 Top-1 | 測試 Top-5 | 測試 Macro-AUC |
|---|---|---|---|---|---|---|
| Plain CNN | 1.19M | 37.40% | 70.66% | 31.51% | 64.19% | 0.8843 |
| ResNet-18（預訓練） | 11.20M | 85.31% | 97.78% | 86.96% | 98.33% | 0.9935 |

### Quiz 3：擴增前後比較

| 模型 | 擴增 | 最終訓練 Top-1 | 最佳驗證 Top-1 | 測試 Top-1 | 測試 Top-5 |
|---|---|---|---|---|---|
| Plain CNN | 無 | 36.01% | 37.40% | 31.51% | 64.19% |
| Plain CNN | 有 | 30.80% | 32.61% | 27.38% | 58.92% |
| ResNet-18 | 無 | 100.00% | 85.31% | 86.96% | 98.33% |
| ResNet-18 | 有 | 98.99% | 84.97% | 86.11% | 97.97% |

## 結論

- **最佳模型**是未使用擴增的 ResNet-18：ImageNet 預訓練讓它在測試 Top-1 上比 Plain CNN 高出 55.45 個百分點，參數量約為 Plain CNN 的 9.4 倍。
- **Plain CNN** 在 15 個 epoch 內仍處於欠擬合（訓練 Top-1 低於驗證），且有 10 個以上類別的測試準確率為 0，可能與類別不平衡有關。
- **資料擴增**在本次設定下沒有提升準確率：Plain CNN 因欠擬合而被擴增拖累（測試 Top-1 −4.13 個百分點）；ResNet-18 的過擬合程度與驗證 Loss 有改善（訓練與驗證差距 15.12 → 14.01、最低驗證 Loss 0.588 → 0.542），但 Top-1 略降 0.85 個百分點。後續可嘗試增加 epoch、降低擴增強度，或讓 Plain CNN 先擬合訓練資料再加入擴增。
- **限制**：結果皆為單一 fold、單一亂數種子的單次實驗；Macro-AUC 只對未擴增的兩個模型計算。

## 執行環境與方式

**環境**：Python 3、NVIDIA GPU（原作業在 RTX 3060 上執行；沒有 GPU 時會退回 CPU，但速度會慢很多）。

**套件**：`torch`、`torchvision`、`numpy`、`pandas`、`matplotlib`、`scikit-learn`、`pillow`、`kagglehub`、`jupyter`。

**執行**：

1. 安裝上述套件；
2. 開啟 `main_basic_revised.ipynb`，由上而下依序執行所有 cell；
3. 第一次執行會自動下載資料集並建立 224 × 224 的影像快取（約數十秒），之後會直接載入快取。

**可調整的設定**（第一個程式 cell 的 `CFG`、`PLAIN_HP`、`RESNET_HP`）：

| 參數 | 說明 |
|---|---|
| `MAX_CLASSES` | 使用的類別數（50；設為 `None` 則使用全部合格類別，除錯時可設 20） |
| `MIN_PER_CLASS` | 訓練池少於此張數的類別會被丟棄（30） |
| `FOLD`、`N_SPLITS` | 使用第幾折、K-Fold 的 K（0、5） |
| `EPOCHS_PLAIN`、`EPOCHS_RESNET` | 各模型的訓練 epoch 數（15） |
| `GPU_NAME_KEYWORD` | 以名稱選擇 GPU（預設 "3060"；找不到時使用 `cuda:0`） |

## 輸出檔案（`outputs/`）

| 檔案 | 內容 |
|---|---|
| `cache_train_224_*.npy`、`cache_test_224_*.npy` | 縮放後的影像快取 |
| `test_pred_plain.csv`、`test_pred_resnet.csv` | 未擴增模型的測試集預測結果（路徑、真實類別、預測類別、信心值、Top-5、是否正確） |
| `test_pred_plain_aug.csv`、`test_pred_resnet_aug.csv` | 擴增模型的測試集預測結果 |
| `roc_compare.png` | 各類別 ROC 與 Macro ROC 比較圖 |
| `compare_basic.csv` | Plain CNN 與 ResNet-18 的準確率、Macro-AUC、參數量比較 |
| `compare_augmentation.csv` | 擴增前後比較表 |
