from sentence_transformers import SentenceTransformer, util
import jieba
import jieba.analyse
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# sentences = ["我喜歡安靜的女生", "我討厭吵鬧的女生"]

# embeddings = model.encode(sentences, convert_to_tensor=True)
# similarity = util.cos_sim(embeddings[0], embeddings[1])

# print("語意相似度：", similarity.item())

# 載入原始模型
base_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
base_model = SentenceTransformer(base_model_name)

# 載入bge-m3模型的 tokenizer 和 model（BERT-based）
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

# 載入訓練好的模型
finetuned_model_path = "finetuned-sbert-love"
finetuned_model = SentenceTransformer(finetuned_model_path)

# 要測試的句子對
pairs = [
    ("我應該喜歡外向活潑的人，喜歡打遊戲", "喜歡玩電腦遊戲，喜歡交朋友"),
    ("喜歡登山", "熱愛戶外活動"),
    ("我喜歡寫程式", "我喜歡社交場合"),
    ("希望對方喜歡運動", "我平常每週都去健身房"),
    ("我喜歡打籃球", "我不喜歡打球"),
    ("我信道教", "我信佛教"),
]

def extract_keywords(text, topK=3):
    keywords = jieba.analyse.extract_tags(text, topK=topK)
    return keywords

# 計算語句嵌入
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 使用平均池化來獲得句子的嵌入
    return embeddings

# 計算相似度（原始模型）
print("=== 原始模型分數 ===")
for s1, s2 in pairs:
    emb1 = base_model.encode(s1, convert_to_tensor=True)
    emb2 = base_model.encode(s2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    print(f"{s1}  vs  {s2} → {score:.3f}")

# 計算相似度（訓練後的模型）
print("\n=== Finetuned 模型分數 ===")
for s1, s2 in pairs:
    emb1 = finetuned_model.encode(s1, convert_to_tensor=True)
    emb2 = finetuned_model.encode(s2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    print(f"{s1}  vs  {s2} → {score:.3f}")

# 關鍵字比對
print("\n=== 關鍵字比對分數 ===")
for s1, s2 in pairs:
    kw1 = extract_keywords(s1)
    kw2 = extract_keywords(s2)
    kw1 = " ".join(kw1)
    kw2 = " ".join(kw2)
    print(f"關鍵字(組成句子): {kw1}  vs  {kw2}")
    emb1 = finetuned_model.encode(kw1, convert_to_tensor=True)
    emb2 = finetuned_model.encode(kw2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    print(f"{s1}  vs  {s2} → {score:.3f} (關鍵字: {kw1} vs {kw2})")

# BGE-M3 模型分數
print("\n=== BGE-M3 模型分數 ===")
for s1, s2 in pairs:
    emb1 = get_embeddings(s1)
    emb2 = get_embeddings(s2)
    score = cosine_similarity(emb1.numpy(), emb2.numpy())[0][0]
    print(f"{s1}  vs  {s2} → {score:.3f}")