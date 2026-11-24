# MMIP - Week 10：檢索增強生成（RAG）

延續 Week 9 的「洗標辨識前處理客服」情境，自建衣物洗護知識庫，完成作業一～四：多輪對話 RAG、多輪對話 VideoRAG（自錄影片）、RAGAS 評估與 GraphRAG。所有模型皆在本機 RTX 3060 執行，不使用雲端 API。

## 1. 專案結構

```text
.
├── main.ipynb              # 作業一～四（含執行結果）
├── requirements.txt        # 實測環境套件版本
├── README.md               # 實驗報告
├── data/
│   ├── kb/care_kb.txt              # 自製洗護知識庫：28 份文件、210 條事實（每條有 fact id）
│   ├── qa_dataset.jsonl            # 自製 QA Dataset：1,050 組（doc2query，5W1H 問法）
│   ├── benchmark/
│   │   ├── single_turn.jsonl       # 手寫單輪題 70 題 + 範圍外 4 題（標註正解 fact id）
│   │   ├── multi_turn.jsonl        # 手寫多輪對話 16 段 × 3 輪（含代名詞／省略）
│   │   └── paper_qa.jsonl          # GraphRAG Benchmark：VideoRAG 論文 24 題（含標準答案與關鍵要點）
│   ├── paper/videorag.pdf          # 作業四論文（arXiv 2501.05874v3）
│   └── videos/                     # 作業二 Video Corpus（自錄影片 *.mp4 已 gitignore；說明見 data/videos/README.md）
│       ├── videos.csv              # 影片 id、標題、speech / silent
│       ├── demo_dialog.txt         # 2.6 多輪示範問題
│       └── video_benchmark.jsonl   # 3.5 評估：10 題（含正解時段與 needs 標註）
└── outputs/
    ├── hw1_*.csv / *.png / *.md    # 作業一：chunk grid、Top-K、Query 技術、生成 Prompt、示範對話、最終設定
    ├── hw3_*.csv / *.jsonl / *.png # 作業三：RAGAS 結果、擾動測試、逐題預測
    ├── hw4_*                       # 作業四：抽取結果、知識圖譜 JSON / 圖、社群報告、Benchmark 結果
    ├── environment.json / summary.json
    └── cache/                      # （gitignore）LLM 呼叫快取，重跑結果一致
```

## 2. 環境安裝（Anaconda + RTX 3060 12GB / CUDA 12.6）

```bash
conda create -n MMIP-w10 python=3.11 -y
conda activate MMIP-w10
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m ipykernel install --user --name mmip-w10 --display-name MMIP-w10
```

在 `week10/` 啟動 Jupyter，選 `MMIP-w10` kernel 由上而下執行。第一次執行會下載 bge-m3、bge-reranker-v2-m3、Qwen2.5-3B-Instruct（約 6GB）；作業二放入影片後才會載入 Qwen2-VL-2B、CLIP 與 Whisper。

| 環境變數 | 作用 |
|---|---|
| `W10_QUICK=1` | 少量樣本快速檢查流程（結果另存 `*_quick`） |
| `W10_FORCE=1` | 忽略快取，全部重算 |
| `W10_LAUNCH_UI=0` | 不啟動 Gradio（批次執行時使用） |
| `W10_VIDEO_DIR` | 指定影片資料夾（預設 `data/videos/`） |

實測環境：Python 3.11.18、PyTorch 2.14.1+cu126、Transformers 5.22.2。文字 LLM 與 VLM 不會同時常駐，峰值顯存約 10GB。所有 LLM 呼叫都用 greedy decoding，並以 prompt hash 快取，有快取時完整重跑約 16 分鐘。

## 3. 資料與模型

### 3.1 自製資料

| 資料 | 內容 | 規模 |
|---|---|---|
| 洗護知識庫 | 洗標符號（ISO 3758 / GINETEX / ASTM / JIS）、材質保養、去漬、洗衣技巧、晾曬收納、疑難排解、Week 9 洗標辨識系統說明 | 28 份文件、210 條事實、7,883 字 |
| QA Dataset | 每條事實用 LLM 產生 5 個問題（5W1H、口語化），答案即該事實 | 1,050 組 |
| 單輪 Benchmark | 手寫口語化問題，標註正解 fact id；另含 4 題範圍外（股價、美食、程式…） | 70 + 4 題 |
| 多輪 Benchmark | 16 段 3 輪對話，第 2、3 輪多為「那它可以烘嗎？」這類省略問句 | 48 輪 |
| 論文 Benchmark | VideoRAG 論文問答：15 題細節（local）、5 題多跳（multi-hop）、4 題全域摘要／5W1H（global） | 24 題 |

**Embedding 數量：** QA 問題向量 1,050 + Chunk 向量 161 + 事實段落 210，**合計 1,421 個**。

### 3.2 模型

| 用途 | 模型 |
|---|---|
| Embedding（作業一、四） | `BAAI/bge-m3`（dense，CLS pooling） |
| Re-Ranker | `BAAI/bge-reranker-v2-m3`（Cross-Encoder） |
| 生成 / 改寫 / RAGAS Judge | `Qwen/Qwen2.5-3B-Instruct`（bf16） |
| VideoRAG 生成（作業二） | `Qwen/Qwen2-VL-2B-Instruct` |
| 影格特徵 / 影片 ASR | `openai/clip-vit-base-patch32` / `openai/whisper-small` |

## 4. 作業項目對照

| 作業 | 項目 | Notebook 章節 |
|---|---|---|
| 一 基礎 | 自製 QA Dataset、Embedding ≥ 1,000 | 1.1 ~ 1.3 |
| 一 基礎 | 多輪對話經 RAG 得到合理回覆與 Reference | 1.4 |
| 一 進階 | 最穩定的 Chunk Size、Overlap、Top-K | 1.5 |
| 一 進階 | 最合適的 Prompting Technique | 1.6 ~ 1.7 |
| 一 進階 | User Interface（Gradio） | 1.8 |
| 二 | 多輪對話 VideoRAG（自錄有說話影片 + 無聲操作影片） | 2.1 ~ 2.6 |
| 三 基礎 | RAGAS 評估作業一 | 3.1 ~ 3.4 |
| 三 進階 | VideoRAG 評估方法設計與實作 | 3.5 |
| 四 | 論文 Graph 化、24 題 Benchmark、GraphRAG 評估 | 4.1 ~ 4.6 |

## 5. 實驗結果

**實驗環境：** RTX 3060 12GB，隨機種子 42，所有生成皆 greedy decoding。

### 作業一：多輪對話 RAG

**系統架構：** 使用者問題 → Query 改寫（依最近 4 句使用者發言補齊代名詞）→ HyDE → Hybrid 檢索（Chunk Index + QA Index，RRF 融合，取 20 個候選）→ bge-reranker 重排 → 去重後取 Top-8 → 檢索閘門（最高 re-rank 分數 < τ 時直接婉拒）→ Qwen2.5-3B 依 G4 prompt 生成，並標註引用 [n]。

**示範（`outputs/hw1_demo_dialogs.md`）：**

| 輪 | 使用者 | 改寫後查詢 | 回覆（摘要） | 引用 |
|---|---|---|---|---|
| 1 | 我有一件羊毛大衣想自己洗 | （同左） | 30°C 以下冷水、中性洗劑、毛巾捲壓吸水、平放陰乾 | WO02、WO04 |
| 2 | 那它可以用洗衣機烘乾嗎？ | 那件羊毛大衣可以用洗衣機烘乾嗎？ | 不可以，滾筒烘乾會造成縮絨 [1] | WO05 |
| 3 | 洗完要怎麼晾？ | 洗完之後該如何晾乾這件羊毛大衣？ | 毛巾捲壓吸水後平放陰乾，不可吊掛 [1] | WO04 |

#### 1.5 Chunk Size / Overlap / Top-K

只用 Chunk Index、不 re-rank。每題分數 = ½·MRR@5 + ½·BudgetRecall@600，其中 MRR@5 依講義「繁中檢索能力評測」計分（第 1 名 1 分、第 2 名 1/2…），BudgetRecall@600 為 context 限 600 字時正解被涵蓋的比例。手寫題與合成題各切 5 folds，穩定度以「平均 − 標準差」排序。

| Chunk Size | Overlap | Chunk 數 | MRR@5（手寫） | MRR@5（合成） | BudgetR@600（手寫） | Fold 平均 | Fold 標準差 | **穩定度下界** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **80** | **32（40%）** | 161 | 0.8774 | 0.7194 | **0.9786** | **0.8684** | 0.0703 | **0.7981** |
| 250 | 50（20%） | 43 | 0.8762 | 0.7754 | 0.9000 | 0.8455 | 0.0554 | 0.7901 |
| 250 | 100（40%） | 45 | 0.8940 | 0.7830 | 0.9143 | 0.8542 | 0.0651 | 0.7892 |
| 150 | 30（20%） | 73 | 0.8595 | 0.7289 | 0.9786 | 0.8560 | 0.0687 | 0.7873 |
| 600 | 0 / 120 / 240 | 28 | **0.9190** | **0.8059** | 0.8786 | 0.8456 | 0.0669 | 0.7787 |
| 400 | 80（20%） | 30 | 0.9167 | 0.8010 | 0.8786 | 0.8450 | 0.0667 | 0.7783 |
| 80 | 0 | 115 | 0.8052 | 0.6642 | 0.9071 | 0.8075 | 0.0975 | 0.7100 |

Top-K（最佳 chunk 設定、Hybrid + Re-Ranker，手寫 70 題）：

| K | Recall@K | Precision@K | 平均 context 字數 | Faithfulness | Answer Relevance | Correctness |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9095 | 0.9286 | 50 | 0.8697 | 0.2420 | 0.2929 |
| 3 | 0.9643 | 0.7000 | 196 | 0.9461 | 0.3774 | 0.4118 |
| 5 | 0.9857 | 0.4371 | 334 | **0.9632** | 0.4674 | 0.5357 |
| **8** | 0.9857 | 0.2768 | 528 | 0.9171 | **0.5721** | **0.5786** |

（生成指標為 35 題分層子集。）

##### 實驗結果分析

- 只看 MRR@5，chunk 越大分數越高（600 字為 0.919），但這是因為大 chunk 幾乎等於整份文件，容易「剛好包含」正解。加入固定 600 字預算的召回率後，**80 字 chunk + 40% overlap** 的分數最高，波動也小，穩定度下界最高（0.798）。
- Overlap 對小 chunk 幫助最大：80 字時從 0% 提升到 40%，穩定度下界由 0.710 升到 0.798，因為事實句不再被切斷。600 字以上的 chunk 已涵蓋整份文件，三種 overlap 的結果完全相同。
- Top-K 增加時 Recall 很快飽和（K=5 已達 0.986），Precision 則持續下降。但 3B 模型回答的完整度隨 K 上升（Correctness 0.29 → 0.58），Faithfulness 在 K=5 最高，K=8 略降。最後依 Faithfulness、Answer Relevance、Correctness 的平均選 **K=8**（context 約 530 字）。

**小結：** 這個知識庫的事實短而獨立，最穩定的設定是「小 chunk + 高 overlap + 較大 K」；重排後的 Top-8 只有約 530 字，對 3B 模型的負擔不大。

#### 1.6 Prompting Technique

**(A) Query 轉換（MRR@5）**

| 技術 | 多輪（48 輪） | 多輪第 2–3 輪 | 單輪（70 題） | 整體 |
|---|---:|---:|---:|---:|
| raw（原句） | 0.5681 | 0.3990 | 0.9536 | 0.7968 |
| concat（串接歷史） | 0.7663 | 0.6964 | 0.9536 | 0.8774 |
| rewrite（Query Rewriting） | 0.7733 | 0.7068 | 0.9536 | 0.8802 |
| rewrite + Multi-Query | 0.7566 | 0.6818 | 0.9557 | 0.8747 |
| rewrite + Step-Back | 0.7431 | **0.7396** | 0.9500 | 0.8658 |
| **rewrite + HyDE** | **0.7743** | 0.7083 | **0.9576** | **0.8831** |
| rewrite（無 Re-Ranker） | 0.6920 | 0.6604 | 0.8886 | 0.8086 |

**(B) 生成 Prompt（多輪 48 輪 + 單輪 23 題 + 範圍外 4 題）**

| Prompt | Faithfulness | Answer Rel. | Correctness | 有引用比例 | 引用正確率 | 範圍外婉拒率 | 誤拒率 | 平均字數 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 基本 | 0.7810 | **0.7082** | 0.5986 | 0.0147 | 1.0000 | 1.00 | 0.0423 | 220 |
| G2 嚴格依據 + 引用 | **0.8724** | 0.5184 | 0.4859 | 0.8958 | 0.7500 | 1.00 | 0.3239 | 55 |
| G3 嚴格 + 單例示範 | 0.7997 | 0.5974 | 0.5704 | **1.0000** | 0.6874 | 1.00 | 0.1268 | 58 |
| **G4 Self-RAG 式先反思再回答** | 0.7698 | 0.6781 | **0.6690** | 0.9552 | 0.6404 | 1.00 | 0.0563 | 79 |

##### 實驗結果分析

- **多輪對話一定要處理 Query：** 用原句檢索時，第 2–3 輪的 MRR@5 只有 0.399；改寫後提升到 0.707。單純串接歷史也有效（0.696），但會把舊話題帶進查詢。
- Rewrite、HyDE 與 Multi-Query 的差距不到 0.01，差異不大。HyDE 整體略高，因此列為最終設定；若考量成本，只做 rewrite（少一次 LLM 呼叫）幾乎一樣好。Step-Back 對追問輪最有幫助，但抽象化後的問題會拉進不相關的背景知識，單輪略降。
- **Re-Ranker 很重要：** 拿掉後整體 MRR@5 從 0.880 掉到 0.809，單輪題掉得最多。
- 生成 Prompt 有明顯取捨。G2「只能依據資料、不足就說不知道」的 Faithfulness 最高，但 3B 模型因此過度保守，32% 範圍內問題被婉拒。G1 回答最長、最切題，卻幾乎不標引用。G4 讓模型先列出「相關資料：[n]」再作答，Correctness 最高（0.669），誤拒率低（5.6%），引用覆蓋率 96%。
- 選擇標準為 Faithfulness、Answer Relevance、Correctness 與有引用比例的平均，**G4 最高**；由於作業要求回覆附 Reference，所以把引用比例納入。

**小結：** 最合適的組合是 **Query Rewriting（+HyDE）+ Re-Ranker + G4 反思式生成 + 檢索閘門**。閘門讓 4 題範圍外問題全部被婉拒。

#### 1.8 User Interface

Gradio 介面（`main.ipynb` 1.8，預設 http://127.0.0.1:7860）：

- 左側為多輪聊天室，附範例問題。
- 右側顯示「改寫後的檢索查詢」，以及 Reference 清單：✅ 表示回答有引用，並列出文件名、段落座標、fact id 與內容摘要，最下方顯示最高 re-rank 分數與延遲。
- 「檢索設定」可即時切換 Query 技術、Top-K 與 Re-Ranker／閘門，用來展示不同設定的差異。

**介面實測（瀏覽器操作）：** 「我有一件羊毛大衣想自己洗」→「那它可以烘乾嗎？」時，改寫正確（「那件羊毛大衣可以烘乾嗎？」），第 1 名檢索結果就是 WO05「羊毛衣物不可滾筒烘乾」（re-rank 0.987），但 G4 生成卻回答「知識庫中沒有相關資訊」；同一段對話改問「那它可以用洗衣機烘乾嗎？」則能正確回答並附引用。範圍外問題（「台積電今天股價多少？」）會被閘門婉拒。由此可見，「檢索正確但 3B 生成器仍婉拒」是現階段的殘餘問題（評估集中約 4 題）。可以考慮的改進是：檢索信心高、生成卻婉拒時，改用 G1 重新生成（Corrective 式回退）。

### 作業二：多輪對話 VideoRAG

**Video Corpus（自錄，1280×720、30 fps）：**

| video_id | 類型 | 長度 | 內容 | 輔助文字 |
|---|---|---:|---|---|
| `care_label` | 有說話（speech） | 29 秒 | 翻看 T 恤洗標並口頭講解：成分棉 75% / 聚酯纖維 25%；聚酯纖維怕高溫，可以烘但不要高溫；建議晾乾；30°C 水洗、不可漂白 | Whisper 逐字稿 5 段 + 影格描述 17 段 |
| `dryer_coin` | 無聲操作（silent） | 29 秒 | 自助洗衣店烘衣機：右側投幣盒投幣 → 面板顯示 30 → 按面板按鍵 → 再投幣，面板增加到 45 | 影格描述 19 段 |

`dryer_coin` 開頭 4 秒有一句旁白，Whisper 轉出約 12 字，依自動規則（逐字稿 ≥10 字）會被判成 speech。因此在 `videos.csv` 手動標成 silent。Whisper 有幾處錯字（「水洗」→「熟悉」、「圖解」→「凸解」、Polyester → Poyester），但「30 度」「不能漂白」「不要用太高溫」等關鍵詞都轉對了。

| VideoRAG 元件（Jeong et al., 2025） | 本作業實作 |
|---|---|
| 影格 1 fps + CLIP 特徵 | PyAV 解碼（不需另裝 ffmpeg），CLIP ViT-B/32 |
| k-means++ 縮小搜尋空間 | 檢索用 8 個候選選 4 張，生成用最多 64 個候選，取最接近各群中心的影格 |
| 檢索用 frame selection | 無訓練資料時，在 C(8,4) 中取「覆蓋度」最高的一組；可選用論文 A.3 的自動標註 + 3 層 MLP（`train_frame_scorer`） |
| 生成用 frame selection | 隨機抽 40 組 m=8 張，以 mean(影格向量)·查詢向量打分取最高一組（論文為 32 張，受限 3060 顯存） |
| 輔助文字 | 有聲影片用 Whisper ASR（含時間戳）；**影格描述由 Qwen2-VL 產生**（每 2 秒 1 張 + 檢索候選影格，提示中帶影片標題，要求寫出畫面上的數字與文字），無聲操作影片主要靠它 |
| 影片表示 | 分數融合：α·z(文字) + (1−α)·z(CLIP 視覺)，α=0.6（論文最佳區間 0.5~0.7） |
| 多輪對話 | 先依最近 3 輪歷史改寫成完整中文問題，再翻成英文給 CLIP；生成時只用改寫後的問題，不放入先前回答 |
| 回答與 Reference | 回答附依據時間點（mm:ss），並列出影片名稱、選用影格時間與逐字稿片段 |

**多輪示範（2.6，`demo_dialog.txt`）：**

| 輪 | 使用者 | 改寫後查詢 | 檢索影片 | 回答（節錄） | 評語 |
|---|---|---|---|---|---|
| 1 | 這件衣服是什麼材質？ | （同原句） | care_label ✅ | 75% 棉和 25% 聚酯纖維混合而成 | 正確 |
| 2 | 那它可以拿去烘嗎？ | 那件衣服可以拿去烘嗎？ | care_label ✅ | 可以，但洗衣溫度不能超過 30 度，不要使用漂白劑 | 「可以烘」正確，但漏了講解中「不要高溫」的原因 |
| 3 | 自助洗衣店的烘衣機要怎麼操作？ | （同原句） | dryer_coin ✅ | 按下電源按鈕開始操作 | 過於簡略，沒提到投幣 |
| 4 | 那如果想烘久一點呢？ | （未改寫） | care_label ❌ | 選擇更高的溫度…… | 改寫失敗，選錯影片 |

**實驗結果分析**
- **檢索：** 兩支影片主題差異明顯，完整問題都能選對影片（3.5 的 Video Recall@1 = 1.0）。
- **多輪改寫是主要瓶頸：** 第 4 輪「那如果想烘久一點呢？」沒被 2B VLM 補上「烘衣機」，檢索被「烘」字帶到洗標影片（講解中有「可以烘」）。作業一用 3B 文字模型改寫時同類問題較少；受限於 12GB 顯存，此處不與 VLM 同時載入。
- **小型 VLM 的生成問題（除錯紀錄）：** 第一版 prompt 下，Qwen2-VL-2B 常直接照抄影格描述（「在這張圖片中……」）或重複句子。另外也觀察到：把上一輪回答放進對話歷史時，模型會照抄上一輪回答；在 prompt 中舉例句型（「可以，但……」）時，模型也會照抄句型。最終做法是三項：生成只用改寫後的問題，不放入歷史；設定 repetition penalty 1.1；後處理移除開頭語、句首時間戳與重複句。

### 作業三：評估 RAG 系統（RAGAS）

依 RAGAS 論文定義，以本機 Qwen2.5-3B 作為 Judge 實作 Faithfulness、Answer Relevance、Context Relevance；另外利用 Benchmark 的正解 fact id 計算 Context Recall / Precision，並以 1~5 分 Judge 評估 Correctness。評估集為單輪 70 題、多輪 48 輪與範圍外 4 題。

| 系統 | Faithfulness | Answer Rel. | Context Rel. | Context Recall | Context Precision | **Correctness** | 有引用比例 | 範圍外婉拒率 | 誤拒率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A. LLM-only（無 RAG） | — | 0.6981 | — | — | — | 0.4195 | — | 0.00 | 0.0000 |
| B. Naive RAG（原句、chunk 400、Top-3、無 rerank、G1） | 0.7216 | **0.7469** | 0.0338 | 0.8432 | 0.7853 | 0.5953 | 0.0085 | 0.00 | 0.0000 |
| **C. 最終 RAG** | **0.8276** | 0.7033 | **0.1316** | **0.9449** | **0.8559** | **0.7479** | **0.9737** | **1.00** | 0.0339 |

單輪與多輪分開比較：

| 系統 | 子集 | Faithfulness | Context Recall | Correctness |
|---|---|---:|---:|---:|
| A. LLM-only | 單輪 / 多輪 | — / — | — / — | 0.4857 / 0.3229 |
| B. Naive RAG | 單輪 / 多輪 | 0.7541 / 0.6741 | 0.9571 / 0.6771 | 0.6643 / 0.4948 |
| C. 最終 RAG | 單輪 / 多輪 | 0.8801 / 0.7440 | 1.0000 / 0.8646 | 0.8464 / 0.6042 |

指標鑑別力（擾動測試，最終系統 30 題單輪）：

| 情境 | Faithfulness | Answer Rel. | Context Recall | Correctness |
|---|---:|---:|---:|---:|
| 原始 | 0.8536 | 0.6651 | 1.0000 | 0.7917 |
| 換成錯誤 Context | **0.1869** | 0.6651 | **0.0333** | 0.7917 |
| 換成錯誤 Answer | **0.2728** | **0.4547** | 1.0000 | **0.0167** |

#### 實驗結果分析

- **RAG 的效益：** 最終系統的 Correctness 由 LLM-only 的 0.420 提升到 0.748，Faithfulness 0.83，正解召回 0.94。LLM-only 常給看似合理但錯誤的數值，例如熨斗溫度、圓點代表的水溫。
- **多輪是主要弱點：** Naive RAG 用原句檢索，多輪 Context Recall 只有 0.677；最終系統提升到 0.865，但 Correctness 仍比單輪低約 0.24。失敗案例集中在 3B 改寫器沒補齊主詞，例如「如果是毛衣呢？」「它是亮亮的緞面材質一直反光」，改寫後的查詢偏離主題，最後被閘門婉拒或答非所問。
- **Answer Relevance 的盲點：** Naive RAG 的 Answer Relevance 最高，因為 G1 回答冗長、會重述問題；最終系統要求簡短並帶引用，這項反而略低。這個指標只看切不切題，不看對不對，需要和 Correctness 一起看。
- **Context Relevance 偏低：** RAGAS 這項指標以「必要句子 / 總句子」計算，Top-K 越大分數越低。最終系統的 0.13 仍是 Naive RAG 的 4 倍，代表小 chunk + 重排後的雜訊比例較低。
- **指標可信度：** 擾動測試中，錯誤 Context 會讓 Faithfulness 從 0.85 掉到 0.19，錯誤 Answer 會讓 Correctness 從 0.79 掉到 0.02，可見指標確實有鑑別力。但 Judge 與生成器是同一個 3B 模型，有自我偏好偏誤，也有明顯誤判（例如 S14、S15 回答正確卻被判 0 分）。因此只適合比較系統之間的相對高低，不宜當成絕對分數。
- **改寫歷史的取捨：** 最初改寫時也放入助理的長回答，在介面實測中 3B 模型反而漏掉「它」指的對象。改成只給使用者發言後，介面案例正常；評估集上多輪 Correctness 為 0.604（含助理回答版本為 0.635），差異在誤差範圍內，因此採用與 1.6 實驗一致的設定。

**小結：** 最終系統在忠實度、召回與正確性上都明顯優於 LLM-only 與 Naive RAG，並能婉拒範圍外問題；多輪改寫仍是最大的改進空間。

#### 3.5 進階：VideoRAG 評估方法設計

把 RAGAS 延伸到「影格 + 語音」兩種 context，並加入影片特有的時間定位與模態歸因。評估集為 `video_benchmark.jsonl`：10 題，分 4 段對話，正解時段依 1 fps 影格人工標註；needs 分佈為 visual 6、both 3、speech 1。

| 面向 | 指標 | 計算方式 |
|---|---|---|
| 影片檢索 | Video Recall@1 | 檢索到的影片是否為正解影片 |
| 時間定位 | Frame Hit Rate、Segment Recall | 選用影格落在正解時段（±2 秒）的比例；是否至少 1 張命中 |
| 影格選擇品質 | Visual Context Relevance | 選用影格與查詢的 CLIP 相似度（Adaptive vs Uniform） |
| 文字忠實度 | Transcript Faithfulness | 回答陳述能否由 ASR／影格描述支持（Judge：Qwen2.5-3B） |
| 視覺忠實度 | Visual Faithfulness | Qwen2-VL 看選用影格，逐條回答「是／否」 |
| 多模態忠實度 | MM Faithfulness | 陳述被文字或影格任一支持的比例 |
| 答案品質 | Correctness | 與人工標準答案比較（1~5 → 0~1） |
| 模態歸因 | V / T / VT 消融、Uniform vs Adaptive | 對應論文 Table 3、4，並依 `needs`（visual / speech / both）分組 |

設計重點：
- 有聲影片的答案多來自語音（T），無聲操作影片只能依賴影格與影格描述（V）。用 `needs` 分組，才看得出系統是否真的「看」了影片。
- 2B VLM 判斷視覺陳述偏寬鬆（容易回答「是」），因此文字面評估改用較強的 3B LLM。視覺忠實度建議人工抽查。

**結果（`outputs/hw3_videorag_eval.csv`）：**

| 變體 | Video Recall@1 | Frame Hit | Segment Recall | Visual Ctx Rel. | Text Faith. | Visual Faith. | MM Faith. | **Correctness** | 延遲 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T-only 檢索（α=1） | 1.0 | 0.488 | 1.0 | 0.267 | 0.458 | 0.733 | 0.900 | 0.306 | 2.16 |
| V-only（生成不給文字） | 1.0 | 0.513 | 1.0 | 0.262 | 0.140 | 0.905 | 0.905 | 0.275 | 2.29 |
| **VT-Adaptive（本系統）** | 1.0 | 0.488 | 1.0 | 0.267 | 0.458 | 0.733 | 0.900 | 0.306 | 2.63 |
| VT-Uniform | 1.0 | 0.513 | 1.0 | 0.260 | 0.325 | 0.850 | 0.850 | **0.375** | 2.43 |

依 `needs` 分組的 Correctness：

| 變體 | visual（6 題） | both（3 題） | speech（1 題） |
|---|---:|---:|---:|
| VT-Adaptive | 0.417 | 0.083 | —（Judge 輸出無法解析） |
| VT-Uniform | 0.417 | 0.417 | 0.000 |
| V-only | 0.292 | 0.333 | 0.000 |

**實驗結果分析**
- **檢索與時間定位沒問題：** 4 種變體 Video Recall@1 與 Segment Recall 都是 1.0。影片只有 30 秒，8 張影格中約一半落在正解時段（±2 秒）。T-only 與 VT-Adaptive 結果完全相同，因為 α 只影響「選哪支影片」，兩者都選對，後續流程一致。只有 2 支影片時，視覺分數對檢索沒有額外貢獻；要看出 α 的影響，需要更多主題相近的影片。
- **答案品質受限於 2B VLM：** Correctness 只有 0.3 左右。能答對的多是「畫面上直接看得到」的題目，例如材質 75/25、面板顯示 30。需要整合多段資訊或理解多輪指代的題目，模型常重複第一題的答案。例如 V03、V04 問水溫與漂白，卻回答材質。只有 VT-Uniform 在 V04 答對「不可漂白」。
- **文字 context 有幫助，但不一定提升正確性：** 給文字的變體 Text Faithfulness 0.46，V-only 只有 0.14，代表回答確實引用了逐字稿與影格描述。V-only 的 Visual Faithfulness 最高（0.905），但它的回答多是對畫面的泛泛描述。這類回答容易被判「畫面看得出來」，卻沒有回答問題，可見忠實度高不代表答案好，和作業三的 Answer Relevance 盲點相同。
- **Adaptive vs Uniform：** 30 秒的影片每秒取 1 張，uniform 取 8 張已經涵蓋所有步驟，adaptive selection 沒有優勢。Uniform 的 Correctness 0.375 略高，但 10 題中差異只有 1 題左右，在誤差範圍內。論文中 adaptive 的優勢出現在長影片。
- **評估本身的限制：** 題數少（10 題）；speech 題只有 1 題，且 Judge 在 VT-Adaptive 上解析失敗。2B 模型在 prompt 小改後結果波動很大：除錯過程中 V-only 曾在 0.15 ~ 0.50 之間變動。因此這組數字只能看大方向。要得到可靠結論，需要更長的影片、更多題目，以及更大的 VLM（例如 7B 量化版）。

### 作業四：GraphRAG（VideoRAG 論文）

**建圖：** 論文去除參考文獻、致謝與行內引用後共 59,768 字元，切成 67 個 chunk（1,200 字元，overlap 200）。由 Qwen2.5-3B 抽取實體與關係，經名稱正規化與 embedding 相似度（>0.93）合併，得到 **244 個節點、194 條邊**，再以 Louvain 分成 **26 個社群**，並由 LLM 為每個社群撰寫報告。圖見 `outputs/hw4_graph.png`。

| 方法 | Correctness | Key-Point Recall | Semantic Sim | Faithfulness |
|---|---:|---:|---:|---:|
| No-RAG | 0.2813 | 0.0833 | 0.6137 | — |
| Naive RAG（chunk Top-4） | **0.5109** | **0.5035** | **0.7220** | **0.8725** |
| GraphRAG-Local | 0.4167 | 0.3854 | 0.6968 | 0.7785 |
| GraphRAG-Global（社群 map-reduce） | 0.2708 | 0.1458 | 0.6176 | 0.6899 |
| GraphRAG-Hybrid（社群報告 + 實體關係 + 原文 chunk） | 0.4375 | 0.4028 | 0.7077 | 0.8542 |

依題型的 Key-Point Recall：

| 方法 | Local（15） | Multi-hop（5） | Global（4） |
|---|---:|---:|---:|
| No-RAG | 0.0389 | 0.2333 | 0.0625 |
| Naive RAG | **0.5167** | **0.6500** | **0.2708** |
| GraphRAG-Local | 0.4222 | 0.4500 | 0.1667 |
| GraphRAG-Global | 0.1500 | 0.1667 | 0.1042 |
| GraphRAG-Hybrid | 0.4222 | 0.5167 | 0.1875 |

#### 實驗結果分析

- **No-RAG 幾乎答不出來**（Key-Point Recall 0.08）：3B 模型不認識這篇論文，會編造年份與方法。
- **Naive RAG 在各題型都最好：** 整體 Correctness 0.51，細節、多跳、全域題的要點覆蓋都最高。GraphRAG 系方法中以 Hybrid 最好（0.44），多跳題與 Naive 接近（0.52 vs 0.65）。但加入社群報告與圖譜脈絡後，context 變長、雜訊變多，3B 生成器反而容易漏掉原文中的關鍵數字。
- **社群報告沒有幫上全域題：** 預期 GraphRAG 最擅長全域題，但 Hybrid 的要點覆蓋只有 0.19，Global 只有 0.10，都低於 Naive 的 0.27。社群報告是 3B 模型從籠統的實體描述再摘要而來，論文的貢獻、限制等內容在抽取與摘要兩關中都流失了。
- **可重現性（除錯紀錄）：** 起初社群內成員、社群順序、子圖邊的順序與關係排序都依賴 Python 集合順序，會隨 `PYTHONHASHSEED` 改變。因此每次重開 kernel，prompt 都不同，LLM 快取失效，Hybrid 的 Correctness 曾在 0.44～0.54 之間變動，也曾出現「Hybrid 優於 Naive」的結論。改成明確排序（度數相同時依名稱，邊依名稱排序）後，以兩個不同的 hash seed 重跑，結果完全一致；第二次執行全部命中快取（0 次 LLM 呼叫）。這也說明：在 24 題、1~5 分 Judge 的規模下，0.05 左右的差距只是雜訊。
- **細節題以 Naive RAG 較好：** 細節題與多跳題的答案多是表格數字或某段原文。3B 抽取器常漏掉數字、把關係描述得很籠統，因此只靠圖（Local）或只靠社群摘要（Global）都會遺失細節。
- **純 Global 表現最差：** map-reduce 每一步都由 3B 模型摘要，資訊逐層流失。最大社群（55 個節點，以「VideoRAG」為中心）把大部分概念聚在一起，社群報告也偏籠統。
- **圖的品質是瓶頸：** 圖中仍有數學符號（˜V、q、f）與附錄案例表格（Making Clay Rose）被抽成實體，同義名稱也只合併 2 組。換用更大的抽取模型或加入 gleaning（多輪補抽）應能改善。

**小結：** 在單篇論文、3B 本機模型的條件下，GraphRAG 沒有勝過直接檢索原文的 Naive RAG。瓶頸在小模型的抽取與摘要品質：圖譜與社群報告遺失了細節，也沒有換來更好的全域理解。Hybrid 是 Graph 系方法中最好的，但要超越 Naive RAG，需要更強的抽取模型（或加入 gleaning）和更精簡的 context 組裝。

## 6. 綜合結論

1. **多輪對話 RAG：** 自製 210 條事實、1,050 組 QA（共 1,421 個 embedding）的洗護知識庫。最穩定的設定是 chunk 80 字 + 40% overlap + Top-8；Query Rewriting 是多輪檢索的關鍵（追問輪 MRR@5 0.40 → 0.71），Re-Ranker 再提升約 0.07。
2. **Prompting：** 嚴格的「不知道就說不知道」會讓小模型過度婉拒；Self-RAG 式「先列相關資料再回答」在正確性、引用與誤拒之間取得最佳平衡。
3. **RAGAS 評估：** 最終系統 Correctness 0.748、Faithfulness 0.83，明顯優於 LLM-only（0.42）與 Naive RAG（0.60），並能婉拒範圍外問題。擾動測試證實指標有鑑別力，但同模型 Judge 有偏誤，只宜做相對比較。
4. **VideoRAG：** 以自錄的有說話影片（洗標講解）與無聲操作影片（自助烘衣機）完成多輪 VideoRAG，影片檢索與時間定位全對。但受限於 2B VLM，答案正確性只有約 0.3，主要卡在多輪改寫與照抄 context。評估方法（時間定位、文字／視覺忠實度、V/T/VT 消融、依 needs 分組）都能實際產出結果，並揭露了這些弱點。
5. **GraphRAG：** 在 3B 本機模型下，Naive RAG（0.51）仍優於所有 GraphRAG 變體（最佳為 Hybrid 0.44），全域題也是如此。圖譜品質受限於小模型抽取，社群報告過於籠統。另外修正了集合順序造成的不可重現問題，修正後結果在不同 hash seed 下完全一致。

**整體而言，在本機 3B 模型的限制下，檢索端的設計（改寫、重排、chunk 粒度）對最終品質的影響，遠大於單純增加 context；GraphRAG 與 VideoRAG 額外加入的脈絡（社群報告、影格描述），在小模型下常成為雜訊。**
