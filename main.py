import torch
from torch.utils.data import DataLoader
#from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model import FakeNewsClassifier
from data_utils import FakeNewsDataset
import pandas as pd
from tqdm import tqdm

# Load dataset
df = pd.read_csv(r"D:\IITJ\Project_dl\data\cleaned_fake_news.csv")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Tokenizer & dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_ds = FakeNewsDataset(train_df['text'], train_df['label'], tokenizer)
test_ds = FakeNewsDataset(test_df['text'], test_df['label'], tokenizer)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FakeNewsClassifier().to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = torch.nn.CrossEntropyLoss()

# ✅ GPU mixed precision optimization
scaler = torch.cuda.amp.GradScaler()

print(f"\n🚀 Using device: {device}")

# Training loop
for epoch in range(3):  # Change to 1 or 5 if needed
    model.train()
    total_loss, total_correct = 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")

    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # mixed precision
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

        # Backward pass (scaled)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loop.set_postfix(loss=loss.item())

    print(f"\nEpoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {total_correct/len(train_ds):.4f}")

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n" + classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

# Save model
torch.save(model.state_dict(), r"D:\IITJ\Project_dl\bert_fake_news_model.pth")
print("\n✅ Training complete and model saved successfully!")
