# LovePairing

以繁體中文文字進行情緒分析、人格傾向分類與伴侶描述相似度比較的 NLP 實驗專案。專案目前由數支獨立的 Python 腳本組成，使用本機模型、情緒詞典及句向量模型進行測試；尚未整合成網站或完整配對服務。

## 功能

- **人格傾向分類**：使用 `Emotion_model/` 中的微調分類模型，將輸入文字分類為溫柔型、活潑型、理性型、感性型或成熟型。
- **伴侶描述相似度比較**：使用原始與微調後的 Sentence Transformer，計算範例文字配對的 cosine similarity。
- **詞典式情緒分析**：以 `dict_emotion.csv` 建立情緒詞典，對輸入句子中的情緒詞加總分數。

## 專案結構

```text
.
├── Classfier.py                       # 人格傾向文字分類（檔名沿用現況）
├── Pair.py                             # 伴侶描述語意相似度比較
├── Lexicon.py                          # 詞典式情緒分析
├── Emotion_model/                      # 本機文字分類模型與 tokenizer
├── finetuned-sbert-love/                # 微調後的 Sentence Transformer
├── dict_emotion.csv                     # 情緒詞典來源資料
├── love_similarity_dataset.csv         # 句子配對相似度資料
├── love_similarity_dataset_final.csv   # 句子配對相似度資料（final 版本）
└── personality_type_dataset.csv        # 人格類型文字標記資料
```

## 環境需求

- Python 3.9 以上
- 執行 `Pair.py` 時需可連線至 Hugging Face，以取得首次執行所需的基礎模型；也可使用已快取的模型。

安裝 Python 套件：

```bash
python -m pip install pandas opencc-python-reimplemented jieba torch transformers sentence-transformers scikit-learn
```

## 執行方式

請在專案根目錄執行，確保腳本能找到相對路徑下的模型與 CSV 檔案。

### 人格傾向分類

```bash
python Classfier.py
```

目前腳本會分類內建的測試句並印出模型結果。分類標籤為：溫柔型、活潑型、理性型、感性型、成熟型。

### 伴侶描述相似度比較

```bash
python Pair.py
```

腳本會對程式內建的句子配對，分別輸出原始模型與微調模型的相似度分數。分數越高代表模型判定兩段文字的語意越接近；它不是配對成功率或機率。

### 詞典式情緒分析

```bash
python Lexicon.py
```

依提示輸入繁體中文句子後，程式會以 jieba 切詞，並輸出情緒詞典中各情緒類別的分數總和。此方式只統計詞典命中的詞，不等同於理解句子的完整語意。

## 資料集

- `personality_type_dataset.csv`：欄位為 `text` 與 `label`，內容是文字及對應的人格類型標籤。
- `love_similarity_dataset.csv`、`love_similarity_dataset_final.csv`：欄位為 `sentence1`、`sentence2` 與 `score`，內容是句子配對及相似度分數。
- `dict_emotion.csv`：情緒詞典來源，供 `Lexicon.py` 讀取；程式會將簡體詞彙轉為繁體後分析。

目前入口腳本沒有提供資料集訓練流程；資料集可作為訓練或評估用途的素材，實際用途需依後續實驗程式決定。

## 注意事項

- `Classfier.py` 的檔名拼法沿用專案現況。
- `Pair.py` 會載入 `paraphrase-multilingual-MiniLM-L12-v2` 與 `bert-base-multilingual-cased`；後者目前雖有載入，但對應的相似度計算程式已註解。
- 模型輸出僅供實驗參考，不應視為對人格、情緒或伴侶適配度的客觀判定。