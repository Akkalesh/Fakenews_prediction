import torch
from transformers import BertTokenizer
from model import FakeNewsClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FakeNewsClassifier()
model.load_state_dict(torch.load(r"D:\IITJ\Project_dl\bert_fake_news_model.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_news(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1).item()

    label = "🟩 Real News" if pred == 0 else "🟥 Fake News"
    print(f"\n📰 {text}\nPrediction → {label}")

# 🧾 Try examples
predict_news("NASA confirms discovery of water on Mars surface.")
predict_news("Government announces 10% tax reduction for middle class families.")
predict_news("Aliens have landed in Delhi and are dancing at India Gate.")
predict_news("The Prime Minister inaugurated a new AI research center in Bangalore.")
predict_news("A man claims he met dinosaurs in Mumbai after time travel experiment.")
predict_news("WHO warns about new global flu variant detected in 2025.")

