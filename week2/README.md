## 環境與執行

1. 如果線上抓取失敗則下載 `UCI_Credit_Card.csv`，放在與 notebook 同一資料夾。
2. 依序執行 notebook（Run All）。Quiz 3 會自行重現 Quiz 2 的前處理與 MLP，可獨立執行。

# Quiz 1：顧客流失分類（Telco Customer Churn）
 
## 1.1 資料與前處理
 
| 項目 | 內容 |
|---|---|
| 資料筆數 | 7,043 筆 |
| 目標欄位 | `Churn`（Yes/No → 1/0），流失比例 **26.537%**（類別不平衡） |
| 前處理 | `TotalCharges` 轉數值並用中位數補缺失；移除識別碼 `customerID`；其餘類別欄位（`gender`、`Contract`、`PaymentMethod` 等）One-Hot Encoding（`drop_first=True`） |
| One-Hot 後欄位數 | 31 欄（含目標欄位） |
| 資料切分 | `train_test_split`，`test_size=0.2`，`stratify=y`，`random_state=42` → **Train 5,634 筆 / Validation 1,409 筆** |
| Feature Scaling | `StandardScaler`，僅對數值欄位 `tenure`、`MonthlyCharges`、`TotalCharges`，只用 Train `fit` 再套用到 Validation |
 
Scaling 後 Train 數值欄位平均 ≈ 0、標準差 ≈ 1（`mean` 在 1e-17 量級、`std` = 1.000089），確認縮放正確。
 
## 1.2 模型一：Logistic Regression（baseline）
 
| Threshold | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 0.50 | 0.8055 | 0.6572 | 0.5588 | 0.6040 |
| 0.30 | 0.7495 | 0.5193 | 0.7540 | 0.6150 |
 
Confusion Matrix：
 
| Threshold | TN | FP | FN | TP |
|---|---|---|---|---|
| 0.50 | 926 | 109 | 165 | 209 |
| 0.30 | 774 | 261 | 92 | 282 |
 
降低門檻（0.5 → 0.3）讓 Recall 由 0.5588 提升到 0.7540，但 Precision 由 0.6572 降到 0.5193，是典型的 Precision / Recall trade-off：門檻越低，模型越傾向預測「會流失」，抓到的真實流失客戶變多（FN 由 165 降到 92），但也誤判更多不會流失的客戶（FP 由 109 升到 261）。
 
## 1.3 模型二：Random Forest（進階）
 
| Threshold | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 0.20 | 0.7133 | 0.4777 | 0.8583 | 0.6138 |
| 0.50 | 0.7921 | 0.6355 | 0.5080 | 0.5646 |
 
Confusion Matrix（由 Accuracy / Precision / Recall 與 Validation 類別數〔374 位流失客戶、1,035 位未流失客戶〕回推，數值與報表 Accuracy 吻合）：
 
| Threshold | TN | FP | FN | TP |
|---|---|---|---|---|
| 0.20（約） | 684 | 351 | 53 | 321 |
| 0.50（約） | 926 | 109 | 184 | 190 |
 
## 1.4 兩模型比較（Validation Dataset 相同）
 
| Model | Threshold | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.8055 | 0.6572 | 0.5588 | 0.6040 |
| Logistic Regression | 0.30 | 0.7495 | 0.5193 | 0.7540 | 0.6150 |
| Random Forest | 0.20 | 0.7133 | 0.4777 | 0.8583 | 0.6138 |
| Random Forest | 0.50 | 0.7921 | 0.6355 | 0.5080 | 0.5646 |
 
### 觀察
 
- **相同門檻 0.5 下**：Logistic Regression 的 Accuracy（0.8055 vs 0.7921）與 Recall（0.5588 vs 0.5080）都優於 Random Forest，Random Forest 的 Precision 略低（0.6355 vs 0.6572）。在這份資料上，未經調參的 Random Forest 在門檻 0.5 時並沒有優於 Logistic Regression。
- **各自「找到的合適門檻」下**：Random Forest 在 threshold=0.20 時 Recall 高達 0.8583，是四組結果中最高，代表它最擅長「抓出真正會流失的客戶」；但 Precision 只有 0.4777，是四組中最低，代表預測為流失時，準確度較差、容易誤判。Logistic Regression 在 threshold=0.30 時的 Recall（0.7540）與 F1（0.6150）則在準確度與召回率間取得較好的平衡。
- **哪一類錯誤（FP / FN）較多，代表什麼意義**：
  - False Negative（實際會流失，模型卻沒抓到）：在流失預測中代價較高，因為公司會忽略這些客戶、錯失挽留機會。
  - False Positive（實際不會流失，模型誤判會流失）：頂多多花行銷／挽留成本在不需要的客戶身上，代價相對較低。
  - 若商業目標是「盡量不漏抓流失客戶」，應選 **FN 最少** 的組合，也就是 **Random Forest @ threshold=0.20**（FN≈53），代價是 FP 大幅增加（≈351）。
  - 若希望預測結果較準確、不浪費太多挽留成本，可選 **Logistic Regression @ threshold=0.5**（FN=165、FP=109，兩者較平衡）。
- **調整 Threshold 帶來的影響**：兩個模型都呈現「降低 threshold → Recall 上升、Precision 下降」的一致趨勢（LR：0.5→0.3，Recall +0.195／Precision −0.138；RF：0.5→0.2，Recall +0.350／Precision −0.158）。Random Forest 對門檻變動更敏感，可見其機率輸出的分布較分散。

# Quiz 2 深度學習信用卡違約預測

使用 [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)（UCI / Kaggle），以 PyTorch 建立 Multi-Layer Perceptron (MLP) 預測客戶下個月是否違約。

- **目標**：建立基礎 MLP，加入 Dropout、L2、Early Stopping 改善，比較前後 Loss 與分類指標。


## 一、資料與前處理

| 項目 | 內容 |
|---|---|
| 資料筆數 | 30,000 筆 |
| 目標欄位 | `default.payment.next.month`（1 = 違約、0 = 未違約），違約比例約 22% |
| 前處理 | 移除 `ID`；`EDUCATION` 的 0/5/6 → 4；`MARRIAGE` 的 0 → 3；`SEX / EDUCATION / MARRIAGE` One-Hot |
| 特徵數量 | **29** |
| 資料切分 | Train / Validation / Test = **21,000 / 4,500 / 4,500**（70% / 15% / 15%，stratify，`SEED=42`） |
| Feature Scaling | `StandardScaler`，只用 Train `fit`，避免資料洩漏 |

Scaling 驗證：Train 前 3 欄平均皆為 0、標準差皆為 1。


## 二、MLP 與改善策略

### 2.1 模型與超參數

結構：`Input(29) → [Linear → ReLU → (Dropout)] × 2 → Linear(1)`，損失函數 `BCEWithLogitsLoss`。

| 超參數 | Baseline | Improved | 說明 |
|---|---|---|---|
| Epochs | 50 | 最多 100（實際 88） | 完整訓練資料被看過的次數 |
| Batch Size | 256 | 256 | 每次更新權重的樣本數 |
| Learning Rate | 1e-3 | 1e-3 | 梯度更新步伐 |
| Optimizer | Adam | Adam | 權重更新演算法 |
| Hidden Layers / Neurons | 2 層：64 → 32 | 2 層：64 → 32 | 隱藏層結構 |
| Dropout | 0 | 0.3 | 訓練時隨機關閉神經元 |
| L2 (weight_decay) | 0 | 1e-4 | 懲罰過大權重 |
| Early Stopping | 無 | patience = 10 | Val Loss 連續 10 epoch 無進步即停止，並還原最佳權重 |

### 2.2 Baseline 訓練過程（節錄）

| Epoch | Train Loss | Val Loss | Val Acc |
|---|---|---|---|
| 1 | 0.5077 | 0.4656 | 0.8051 |
| 10 | 0.4261 | 0.4392 | 0.8173 |
| 20 | 0.4177 | 0.4382 | 0.8158 |
| 30 | 0.4110 | 0.4421 | 0.8151 |
| 40 | 0.4042 | 0.4477 | 0.8140 |
| 50 | 0.3979 | 0.4484 | 0.8127 |

Training Loss 持續下降，Validation Loss 在約第 16 epoch 後開始回升，是典型的 overfitting。

![alt text](image.png)

### 2.3 改善前後比較

![alt text](image-1.png)

| 穩定度／過擬合指標 | Baseline | Improved |
|---|---|---|
| 實際訓練 epoch 數 | 50 | 88 |
| Val Loss 最低點 | 0.4380（epoch 16） | 0.4350（epoch 83） |
| 最後 epoch 的 Val − Train Loss | +0.0504 | +0.0032 |
| 最後 10 epoch Val Loss 標準差 | 0.0018 | 0.0003 |

### 2.4 觀察

- **Overfitting 明顯改善**：<br>Train / Val Loss 差距由 +0.0504 縮小到 +0.0032。Baseline 的 Val Loss 在第 16 epoch 達到最低 0.4380 後回升至 0.4484，Improved 則沒有出現回升。
- **Validation Loss 下降**：<br>0.4484 → 0.4350；即使與 Baseline 自己的最低點 0.4380 相比，仍低 0.0030。<br>Test Loss 也由 0.4466 降至 0.4355，結果一致。
- **模型更穩定**：<br>最後 10 epoch 的 Val Loss 標準差由 0.0018 降至 0.0003。
- **Accuracy 與 Precision 提升**：<br>Validation Accuracy 0.8127 → 0.8198，Precision 0.6238 → 0.6783。
- **Recall 與 F1 下降**：<br>Recall 0.3849 → 0.3518，F1 0.4761 → 0.4633。正則化讓模型預測更保守，且資料只有約 22% 違約、分類門檻固定 0.5，對少數類別不利。這代表「機率預測品質變好、但 0.5 門檻不再適合」，可用調整 threshold 或 `pos_weight` 改善。



# Quiz 3 模型表現評估

延續 Quiz 2 的信用卡違約預測資料集與 MLP 模型

- **目標**：繪製 ROC Curve、計算 AUC，並與 Random Forest(隨機森林) 比較正負樣本區分能力

### 3.1 設定

- **MLP**：使用 Quiz 2 的 Improved MLP，對 Validation Dataset（4,500 筆）輸出違約機率 (sigmoid) 作為 prediction score
- **第二個模型**：Random Forest（`n_estimators=300, max_depth=8, min_samples_leaf=20`），使用相同的 Train / Validation 資料與 Scaling，score 取 `predict_proba` 的第 2 欄

### 3.2 MLP Prediction Score

- Score 形狀：`(4500,)`，範圍 0.0001 ~ 0.9474。
- 前 10 筆 Validation 資料：

| # | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Score | 0.1834 | 0.2161 | 0.1641 | 0.0906 | 0.2705 | 0.0923 | 0.3445 | 0.1865 | 0.0957 | 0.0886 |
| 真實標籤 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

第 2 筆為違約客戶，score 為 0.2161，低於 0.5 門檻而被判為未違約；但 ROC / AUC 只看「排序」，不受單一門檻影響。

### 3.3 AUC 結果

| Model | AUC (Validation) | AUC (Test，參考) |
|---|---|---|
| MLP (improved) | **0.7770** | 0.7736 |
| Random Forest | 0.7729 | **0.7768** |

Bootstrap（1,000 次）：AUC 差 (MLP − Random Forest) = **+0.0042**，95% CI = **[−0.0034, +0.0122]**，區間包含 0。

### 3.4 不同 FPR 下的 TPR

| FPR | MLP TPR | Random Forest TPR | 較高者 |
|---|---|---|---|
| 0.05 | 0.360 | 0.346 | MLP |
| 0.10 | 0.488 | 0.474 | MLP |
| 0.20 | 0.601 | 0.625 | Random Forest |
| 0.30 | 0.695 | 0.698 | 幾乎相同 |
| 0.50 | 0.846 | 0.826 | MLP |

### 3.5 ROC 與 AUC 的意義

- **ROC Curve**：掃過所有分類門檻，畫出 FPR = FP/(FP+TN)（橫軸）對 TPR = TP/(TP+FN)（縱軸）的曲線，呈現「抓到多少違約者」與「多少好客戶被誤判」的取捨，不依賴單一門檻。曲線越靠左上角越好，對角線代表隨機猜測。
- **AUC**：ROC 曲線下面積（0 ~ 1）。意義為「隨機抽一位違約、一位未違約的客戶，模型給違約者較高分數的機率」。0.5 = 無區分能力，1.0 = 完美區分。
- 本次 MLP 的 AUC = 0.7770，表示約有 77.7% 的機率能把違約客戶排在未違約客戶之前，區分能力屬於「尚可」（0.7 ~ 0.8）。

### 3.6 比較與結論

- 兩條 ROC 曲線互相接近且有交叉：FPR ≤ 0.1 及 FPR = 0.5 時 MLP 的 TPR 較高，FPR = 0.2 時 Random Forest 較高，FPR = 0.3 時幾乎重疊。
- 在 Validation 上 MLP 的 AUC 略高（0.7770 vs 0.7729，差 0.0042），但在 Test 上 Random Forest 略高（0.7768 vs 0.7736）。
- Bootstrap 95% CI（−0.0034 ~ +0.0122）包含 0，表示**兩者差異不顯著**。
- 結論：MLP 在 Validation 上名義 AUC 較高，但兩個模型的正負樣本區分能力實質上相當；對這類中小型表格資料，Random Forest 這類樹模型與 MLP 表現接近是常見現象。
- 補充：AUC 衡量排序能力，與門檻無關；實務上仍需依 Precision / Recall 需求選擇分類門檻。