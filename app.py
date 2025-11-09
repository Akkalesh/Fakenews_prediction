import torch
import streamlit as st
from transformers import BertTokenizer
from model import FakeNewsClassifier

# ✅ Set Streamlit page config FIRST
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = FakeNewsClassifier()
    model.load_state_dict(torch.load(r"D:\IITJ\Project_dl\bert_fake_news_model.pth", map_location=device))
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

model, tokenizer = load_model()

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

    return "🟩 Real News" if pred == 0 else "🟥 Fake News"

# ✅ Streamlit UI
st.title("📰 Fake News Detection using BERT")
st.markdown("Enter a news headline or paragraph below to classify it as Real or Fake.")

user_input = st.text_area("📝 Enter News Text:", height=150)

if st.button("🔍 Predict"):
    if user_input.strip():
        result = predict_news(user_input)
        st.subheader("Prediction:")
        st.success(result)
    else:
        st.warning("Please enter some text to analyze.")
