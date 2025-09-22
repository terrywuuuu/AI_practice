import pandas as pd
import opencc
from collections import defaultdict
import jieba

# 把簡體字轉換成繁體字
converter = opencc.OpenCC('s2t')

# 情緒代碼轉中文
emotion_map = {
    'PA': '喜愛', 'PE': '喜悅', 'PD': '尊敬', 'PH': '安心',
    'NA': '仇恨', 'NB': '鄙視', 'NN': '厭惡', 'NE': '憂愁',
    'ND': '恐懼', 'NI': '驚訝', 'PC': '信任', 'NC': '焦慮'
}

# 讀入詞典
df = pd.read_csv('dict_emotion.csv')

# 建立情緒詞典（繁體詞 → [(情緒, 強度)]）
emotion_dict = defaultdict(list)

for _, row in df.iterrows():
    word_simp = str(row['词语'])
    word_trad = converter.convert(word_simp)        
    emo_code = str(row['情感分类']).strip()
    intensity = float(row['强度'])
    
    if emo_code in emotion_map:
        emo_name = emotion_map[emo_code]
        emotion_dict[word_trad].append((emo_name, intensity))
    
    # 補充處理副情緒分類
    aux_code = str(row['辅助情感分类']).strip()
    if aux_code and aux_code in emotion_map:
        aux_emo = emotion_map[aux_code]
        aux_intensity = float(row['强度.1'])
        emotion_dict[word_trad].append((aux_emo, aux_intensity))

def analyze_emotion(text, emotion_dict):
    words = jieba.lcut(text)            # 分割句子
    scores = defaultdict(float)
    for word in words:
        if word in emotion_dict:
            for emo, score in emotion_dict[word]:
                scores[emo] += score
    return dict(scores)


text = input("請輸入一句繁體中文：")
result = analyze_emotion(text, emotion_dict)
print("情緒分數：", result)
