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
    # 0
    ("我希望未來的另一半能夠喜歡跟我一起打遊戲，尤其是傳說對決，最好是可以配合得上，玩到後期能夠有良好的配合。平常我也喜歡看些懸疑電影，喜歡那種需要動腦的情節。還有，喜歡一個安靜的生活，週末可以去逛書店，聊一些輕鬆的話題。"
     , "我是一個非常喜歡打遊戲的人，尤其是傳說對決，還有一些策略類的遊戲，平常晚上常常會花時間跟朋友組隊，爭取更高的排位。我比較內向，喜歡安靜的環境，周末我會去書店或是看一些比較有深度的電影。如果能找到一個也喜歡這些的伴侶，我覺得應該會很合得來。"),
    # 1
    ("我希望對方能夠喜歡戶外活動，像是登山、騎腳踏車之類的，最好能在周末有時間陪我一起去爬山，享受那種清新的空氣。當然，如果你喜歡挑戰自己也沒問題，我也喜歡極限運動，像是跳傘，這些能讓我感覺到生活的刺激。"
     , "平常我很喜歡戶外運動，尤其是登山和騎腳踏車，一個月大概會爬兩三次山，對我來說，爬山不只是運動，還是一種放鬆心情的方式。除了這些，我也有在學習一些極限運動，最近學會了滑板和滑雪，對我來說挑戰自己是一個不小的樂趣。"),
    # 2
    ("理想型是一個能夠一起看動畫，特別是那些畫風很精緻的日本動畫。最好能夠和我一起討論劇情和角色，因為我覺得這樣的共鳴感很重要。除了動畫，還喜歡一起研究一些科幻小說，探索一些未來世界的概念。"
     , "我是一個動畫迷，尤其是日本的動畫，特別喜歡那種畫風精緻的作品，像是《你的名字》這類的電影我已經看了很多遍，每次都能有不同的感受。除此之外，我也對科幻小說有點興趣，最近看了《三體》，對未來科技和世界觀的探討感覺很有趣。"),
    # 3
    ("我喜歡會打籃球的人", "我不喜歡運動"),
    # 4
    ("我喜歡做瑜伽的人", "我覺得瑜伽太無聊"),
    # 5
    ("我喜歡會打高爾夫的人", "我喜歡打保齡球"),
    # 6
    ("我喜歡打棒球的人", "我喜歡打壘球"),
    # 7
    ("我喜歡愛旅行的人", "我不喜歡長途旅行，偏好宅在家裡"),
    # 8
    ("我喜歡平常會看動漫的人", "我對動漫沒興趣"),
    # 9
    ("我信道教", "我信佛教"),
    # 10
    ("我喜歡有養狗的人", "我喜歡貓咪"),
    # 11
    ("我怕高", "我喜歡跳傘"),
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

pair = 0

# 計算相似度（原始模型）
print("=== 原始模型分數 ===")
for s1, s2 in pairs:
    emb1 = base_model.encode(s1, convert_to_tensor=True)
    emb2 = base_model.encode(s2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    print(f"配對{pair} → {score:.3f}")
    pair += 1

pair = 0

# 計算相似度（訓練後的模型）
print("\n=== Finetuned 模型分數 ===")
for s1, s2 in pairs:
    emb1 = finetuned_model.encode(s1, convert_to_tensor=True)
    emb2 = finetuned_model.encode(s2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    print(f"配對{pair} → {score:.3f}")
    pair += 1

# # 關鍵字比對
# print("\n=== 關鍵字比對分數 ===")
# for s1, s2 in pairs:
#     kw1 = extract_keywords(s1)
#     kw2 = extract_keywords(s2)
#     kw1 = " ".join(kw1)
#     kw2 = " ".join(kw2)
#     print(f"關鍵字(組成句子): {kw1}  vs  {kw2}")
#     emb1 = finetuned_model.encode(kw1, convert_to_tensor=True)
#     emb2 = finetuned_model.encode(kw2, convert_to_tensor=True)
#     score = util.cos_sim(emb1, emb2).item()
#     print(f"{s1}  vs  {s2} → {score:.3f} (關鍵字: {kw1} vs {kw2})")

# # BGE-M3 模型分數
# print("\n=== BGE-M3 模型分數 ===")
# for s1, s2 in pairs:
#     emb1 = get_embeddings(s1)
#     emb2 = get_embeddings(s2)
#     score = cosine_similarity(emb1.numpy(), emb2.numpy())[0][0]
#     print(f"{s1}  vs  {s2} → {score:.3f}")