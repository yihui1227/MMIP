# MMIP Week 3（基礎＋進階）：動物影像分類

> 使用 Kaggle 的 **Animals Detection Images Dataset**（選用 50 個動物類別），涵蓋三個 Quiz 的既有基礎結果，以及可自行執行的進階實驗流程：準備資料集、訓練 CNN 影像分類模型、提升泛化能力（Data Augmentation）。

## 資料集

- **來源**：[Kaggle：antoreepjana/animals-detection-images-dataset](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset)，影像取自 Open Images，透過 `kagglehub` 自動下載，固定使用與原作業相同的版本 7。
- **原始規模**：29,071 張、80 個動物類別（官方 train 22,566 張、test 6,505 張）。
- **目錄結構**：`train/<類別>/xxx.jpg`、`test/<類別>/xxx.jpg`，每個類別資料夾內另有 `Label/`（物件偵測的標註）。本作業只做影像分類，不使用 `Label/`，以自訂索引掃描影像、處理重複與切分。`ImageFolder` 依根目錄下一層建立類別，巢狀 `Label/` 不會自動成為獨立類別；使用自訂索引是為明確控制資料。

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

**本次使用的 50 個類別**：Bear、Butterfly、Camel、Canary、Caterpillar、Cattle、Centipede、Cheetah、Chicken、Crocodile、Deer、Eagle、Elephant、Fish、Fox、Frog、Goldfish、Hedgehog、Jaguar、Kangaroo、Koala、Leopard、Lion、Lizard、Lynx、Monkey、Moths and butterflies、Mouse、Ostrich、Owl、Panda、Parrot、Penguin、Polar bear、Rabbit、Red panda、Rhinoceros、Scorpion、Sea lion、Shark、Snake、Sparrow、Spider、Squirrel、Swan、Tiger、Tortoise、Woodpecker、Worm、Zebra。

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

**加速設計**：所有影像先一次性縮成 224 × 224 並快取成 `.npy`，訓練時整批資料放在 GPU 上，擴增也在 GPU 上完成。裁切與旋轉以自訂 affine_grid/grid_sample 合併近似，含零填補邊界，並非直接呼叫 torchvision RandomResizedCrop 的完全相同實作。

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

- **原基礎實驗最佳模型**是未使用擴增的 ResNet-18：其測試 Top-1 比 Plain CNN 高出 55.45 個百分點，參數量約為 Plain CNN 的 9.4 倍。
- **Plain CNN** 的低訓練與驗證準確率顯示可能欠擬合（但訓練含 Dropout，指標與驗證模式不同，不能單憑訓練 Top-1 低於驗證下定論），且有 10 個以上類別的測試準確率為 0，可能與類別不平衡有關。
- **資料擴增**在本次設定下沒有提升準確率：Plain CNN 因欠擬合而被擴增拖累（測試 Top-1 −4.13 個百分點）；ResNet-18 的訓練／驗證指標差距及驗證 Loss 較小（訓練與驗證差距 15.12 → 14.01、最低驗證 Loss 0.588 → 0.542），但 Top-1 略降 0.85 個百分點；由於訓練使用擴增，指標差距不完全可比，較低 Loss 也不等於已證明校準改善。後續可嘗試增加 epoch、降低擴增強度，或讓 Plain CNN 先擬合訓練資料再加入擴增。
- **限制**：結果皆為單一 fold、單一亂數種子的單次實驗；Macro-AUC 只對未擴增的兩個模型計算。

## 執行環境與方式

**環境**：進階流程的準備環境為 Python 3.12、PyTorch 2.6.0 + CUDA 12.4、torchvision 0.21.0、NVIDIA GPU（原作業在 RTX 3060 上執行；沒有 GPU 時會退回 CPU，但速度會慢很多）。

**套件**：`torch`、`torchvision`、`numpy`、`pandas`、`matplotlib`、`scikit-learn`、`pillow`、`kagglehub`、`jupyter`。

**執行**：

1. 安裝上述套件；
2. 開啟 `main.ipynb`，由上而下依序執行所有 cell；
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

## 進階部分：全部整合在 main.ipynb

**目前進度**：基礎題已保留原有實測結果；進階程式已補齊，但本次六組完整訓練尚未完成。依你的安排，已停止訓練，交由你自行執行。Notebook 不依賴其他 `.py` 檔，也不預填進階準確率或宣稱已觀察到尚未產生的影像。

### Quiz 2 進階：超參數實驗（20 分）

| 模型 | 比較的學習率 | 其他固定條件 |
|---|---|---|
| Plain CNN | 1e-3（基準）、3e-4、3e-3 | width 32、dropout 0.3 |
| ResNet-18 | 3e-4（基準）、1e-4、1e-3 | ImageNet 預訓練、全網微調 |

各組使用相同 50 類、Fold 0、seed 42、224×224、batch 64、15 epochs、AdamW、weight decay 1e-4、Cosine 排程、label smoothing 0，並關閉擴增以隔離學習率影響。訓練 loader 每輪隨機丟棄不足一批的 32 張影像，各組規則一致。每次建立模型前重設種子；CUDA 與 cuDNN benchmark 仍可能造成細微差異。

假說：較大的學習率可能加快 Plain CNN 在有限預算內的擬合，但也可能不穩定；ResNet 較小的微調步幅可能有利於保留預訓練特徵。這些是假說，不能先當成結論。

每組依驗證 Top-1 選最佳 epoch，同分取較早者，Top-5 與 Loss 取同個 epoch。再以驗證 Top-1 選每種模型的最佳學習率，先寫入 `selection.json`，才評估本次重跑基準與選定模型的獨立測試集。新舊環境可能不同，因此調參效果以本次重跑基準比較。若基準仍最佳，也會如實保留。

### Quiz 3 進階：兩個 Kernel 與 XAI（20 分）

- **Kernel**：視覺化選定且訓練完成的 ResNet-18 第一層 `conv1` 第 0、1 個 filter，各包含三個 RGB 7×7 kernel；同時呈現卷積反應圖與方向變化統計，據此推測邊緣、紋理或色彩對比用途。
- **Grad-CAM**：使用 `layer4` 的 7×7 特徵與目標類別 logit 的梯度產生熱圖，再插值至 224×224。固定選取前兩張不同類別的正確案例及前兩張錯誤案例；錯誤案例比較預測與真實類別熱區。[Grad-CAM 原論文](https://arxiv.org/abs/1610.02391)。
- **分析**：比較遮蔽最高／最低 20% 關注像素後的預測機率，並回報熱圖重心位置。執行後仍需依實際疊圖補充動物部位、背景或遮擋等觀察，不能只由數字猜測影像內容。

### 執行方式

1. 在 `main.ipynb` 選擇專案 `.venv` 的 Python 3.12 kernel。已建立 GPU 環境；套件實際版本保存在 `requirements-lock.txt`。
2. 需要重建環境時，可執行下方安裝命令。
3. **完整作業**：由上往下執行 Notebook，包含四組基礎訓練及六組進階訓練。
4. **只跑進階**：依序執行 imports／CFG、資料下載、重複檢查、資料清理、快取與 K-Fold、GPU loader、共用訓練工具，以及 `PlainCNN`／`build_resnet` 定義的程式 cell；接著跳至「Quiz 2 進階」執行該節與最後的展示 cell。依 Notebook 零起算 cell 索引，前置程式為 **1、3、4、5、7、8、11、13**，進階程式為 **32、33、35**。索引供定位，請以標題與內容辨識。

```powershell
# 若既有 .venv 可使用，無需再次建立
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

進階 cell 內含 `run_advanced()`，會直接開始六組訓練。每組保存完整 history 與驗證最佳權重；重跑時會跳過已完成組別，中斷在未完成組別則重跑該組。所有六組完成後，才能執行 `run_advanced(visualize_only=True)` 重畫解釋圖與重新評估，這個選項不會訓練缺少的組別。

進階結果保存於 `outputs/advanced/`，不覆寫原有基礎 CSV。若修改訓練設定、資料切分或套件環境，請先將該目錄備份並更名，避免混用紀錄。程式會比對設定與切分 SHA-256。

### 執行後產生的進階輸出

| 檔案（位於 outputs/advanced/） | 內容 |
|---|---|
| `config.json`、`split_manifest.csv` | 設定、類別順序、切分及 SHA-256 |
| `history_*.csv`、`hyperparameter_results.csv` | 六組每 epoch 紀錄及驗證指標彙整 |
| `*.pt` | 各組驗證最佳權重，本機保存且由 Git 忽略 |
| `selection.json`、`test_results.csv` | 驗證選模與最終測試指標 |
| `test_pred_*.csv`、`probabilities_*.npz` | 逐張測試預測與完整機率，npz 由 Git 忽略 |
| `hyperparameter_curves.png` | 驗證 Top-1 學習曲線 |
| `kernels.png`、`kernels_conv1.npy`、`kernel_statistics.csv` | 兩個 filter、輸入反應與方向統計 |
| `gradcam.png`、`gradcam_maps.npy` | 四個案例的預測／真實類別解釋圖與熱圖數值 |
| `xai_cases.csv`、`visualization_metadata.json` | 案例索引、目標層、遮蔽對照與機率 |
| `REPORT.md` | 由實測結果自動產生的表格、量化分析、用途推測及待補的逐圖觀察 |

**限制**：單 fold、單 seed、三種學習率的有限搜尋，沒有重複實驗或顯著性檢定；固定 15 epochs 只比較此預算下的表現。不同架構比較同時包含容量與預訓練差異，不能單獨推論架構因果。ROC-AUC 衡量排序，Top-1 衡量分類，兩者數值不可直接比較，負例多不會自動使 AUC 提高。Grad-CAM 解析度有限且逐張正規化，四個示例與遮蔽對照不能證明因果或整體定位品質。
