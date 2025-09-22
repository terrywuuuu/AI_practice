from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

model_path = "Emotion_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 測試模型
test_text = "我比較偏好不輕易發脾氣類型的伴侶"
result = classifier(test_text)

print(result)