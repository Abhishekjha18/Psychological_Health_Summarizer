from libraries import *
from scores import SentimentModel, PHQ9Detector, ImportanceFilter

def read_txt_file2(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    samples = data.split("=" * 5)
    dialogues, reference_summaries = [], []

    for sample in samples:
        lines = sample.strip().split("\n")
        dialogue_lines = []
        summary = None

        for line in lines:
            if line.startswith("Summary:"):
                summary = line[len("Summary:"):].strip()
            else:
                dialogue_lines.append(line.strip())

        if dialogue_lines and summary:
            dialogue_text = " ".join(dialogue_lines).strip()
            dialogues.append(dialogue_text)
            reference_summaries.append(summary)

    return dialogues, reference_summaries
    
class TextSummaryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        labels = torch.tensor(example["labels"], dtype=torch.long)
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "labels": labels
        }

def preprocess_T5(examples):
    inputs = ["summarize: " + doc for doc in examples["input"]]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_rouge_scores(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougel_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougel = sum(rougel_scores) / len(rougel_scores)

    print(f"ROUGE-1 (R-1): {avg_rouge1:.4f}")
    print(f"ROUGE-2 (R-2): {avg_rouge2:.4f}")
    print(f"ROUGE-L (R-L): {avg_rougel:.4f}")

def plot_loss_curve(train_losses, val_losses, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss", marker="o")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training & Validation Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()

def generate_summary_T5(input_text):
    input_ids = tokenizer(
        "Summarize the following Mental Health Counselling Session:" + input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).input_ids.to(device)

    output_ids = model.generate(
        input_ids,
        max_length=150,
        num_beams=2,                      
        no_repeat_ngram_size=4,          
        repetition_penalty=2.20,          
        length_penalty=1.56,             
        early_stopping=True
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

if __name__=="__main__":

    train_dialogues, train_summaries = read_txt_file2("train_dialogues_f.txt")
    val_dialogues, val_summaries = read_txt_file2("val_dialogues_f.txt")
    test_dialogues, test_summaries = read_txt_file2("test_dialogues_f.txt")
    train_dataset = Dataset.from_dict({"input": train_dialogues, "output": train_summaries})
    val_dataset = Dataset.from_dict({"input": val_dialogues, "output": val_summaries})
    test_dataset = Dataset.from_dict({"input": test_dialogues, "output": test_summaries})
    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)

    model_name = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    train_dataset = train_dataset.map(preprocess_T5, batched=True)  
    val_dataset = val_dataset.map(preprocess_T5, batched=True)     
    test_dataset = test_dataset.map(preprocess_T5, batched=True)     

    model.to(device)
    
    train_dataset_pt = TextSummaryDataset(train_dataset)
    val_dataset_pt = TextSummaryDataset(val_dataset)
    
    BATCH_SIZE = 2
    train_loader = DataLoader(train_dataset_pt, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset_pt, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), 
                     lr=2.368863950364079e-05,
                     weight_decay=0.031203728088487304)  
    
    EPOCHS = 10
    num_training_steps = EPOCHS * len(train_loader)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.031198904067240532 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience = 3
    early_stopping_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\n Epoch {epoch+1}/{EPOCHS}:")
        
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device).long()
    
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
    
            train_loss += loss.item()
    
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Training Loss: {avg_train_loss:.4f}")
    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(device).long()
    
                outputs = model(**inputs, labels=labels)
                val_loss += outputs.loss.item()
    
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f" Validation Loss: {avg_val_loss:.4f}")
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            print(" Best validation loss improved! Saving model...")
            model.save_pretrained("t5-large-best-optimized")
            tokenizer.save_pretrained("t5-large-optimized")
        else:
            early_stopping_counter += 1
            print(f" Early stopping counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print(" Early stopping triggered. Training stopped!")
                break
    
    model.save_pretrained("t5-large-final-optimized")
    tokenizer.save_pretrained("t5-large-final-optimized")

    plot_loss_curve(train_losses, val_losses, "T5 final (optimized)")
    
    model_name = "t5-large-final-optimized"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    generated_summaries = [generate_summary_T5(text) for text in test_dialogues] 

    print("Generated summaries for T5 (optimized):")
    for i, summary in enumerate(generated_summaries[:5]):
        print(f"Dialogue {i+1}: {test_dialogues[i]}")
        print(f"Generated Summary {i+1}: {summary}\n")

    bleu = sacrebleu.corpus_bleu(generated_summaries, [test_summaries])
    print(f"BLEU Score: {bleu.score:.2f}")

    compute_rouge_scores(generated_summaries, test_summaries)

    P, R, F1 = bert_score.score(generated_summaries, test_summaries, model_type="roberta-large",lang="en")
    print(f" BERTScore (F1): {F1.mean().item():.4f}")