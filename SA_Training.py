from libraries import *

def read_txt_file(file_path):
    def map_emotion(e):
        if e in [-2, -1]:
            return 0
        elif e == 0:
            return 1
        elif e in [1, 2]:
            return 2

    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    samples = data.split("=" * 80)
    dialogues, emotions = [], []

    for sample in samples:
        lines = sample.strip().split("\n")
        dialogue = []
        emotion_values = []

        for line in lines:
            if line.startswith("Summary:"):
                break

            dialogue.append(line)
            match = re.search(r"\(Emotion=(-?\d+)\)", line)
            if match:
                raw_emotion = int(match.group(1))
                emotion_values.append(map_emotion(raw_emotion))

        if dialogue and emotion_values:
            dialogues.append(" ".join(dialogue))
            emotions.append(emotion_values[-1])

    return dialogues, emotions

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def eval_epoch(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, predictions = torch.max(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

if __name__=="__main__":
    
    train_dialogues, train_emotions = read_txt_file("train_dialogues.txt")
    val_dialogues, val_emotions = read_txt_file("val_dialogues.txt")
    
    train_dataset = Dataset.from_dict({"text": train_dialogues, "label": train_emotions})
    val_dataset = Dataset.from_dict({"text": val_dialogues, "label": val_emotions})

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment",num_labels=3)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.to(device)

    num_epochs = 10
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_dataloader, optimizer)
        print(f"Training Loss: {train_loss:.4f}")
        
        val_accuracy = eval_epoch(model, val_dataloader)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
    
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained("fine-tuned-sentiment-model")
            tokenizer.save_pretrained("fine-tuned-sentiment-model")
            print("Model saved!")
    
    model.save_pretrained("final-sentiment-model")
    tokenizer.save_pretrained("final-sentiment-model")
    