from libraries import *

class SentimentModel:
    def __init__(self, model_name="fine-tuned-sentiment-model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1).numpy()[0]
        return scores[2] - scores[0]  # Positive - Negative

class PHQ9Detector:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.descriptions = {
            'little_interest': "Little interest or pleasure in doing things",
            'feeling_down': "Feeling down, depressed, or hopeless",
            'sleep_issues': "Trouble falling or staying asleep, or sleeping too much",
            'low_energy': "Feeling tired or having little energy",
            'appetite_issues': "Poor appetite or overeating",
            'feeling_bad': "Feeling bad about yourself or that you are a failure",
            'concentration': "Trouble concentrating on things",
            'moving_slowly': "Moving or speaking slowly, or being restless",
            'self_harm': "Thoughts that you would be better off dead or of hurting yourself"
        }
        self.embeddings = {
            cat: self.sentence_model.encode([desc])[0]
            for cat, desc in self.descriptions.items()
        }
        self.keywords = {
            'little_interest': ['don\'t care', 'not interested', 'nothing matters', 'pointless', 'pleasure', 'enjoy'],
            'feeling_down': ['sad', 'depressed', 'hopeless', 'worthless', 'negative', 'down'],
            'sleep_issues': ['can\'t sleep', 'insomnia', 'sleeping too much', 'tired', 'awake', 'nightmares'],
            'low_energy': ['exhausted', 'no energy', 'fatigue', 'tired', 'can\'t focus', 'drained'],
            'appetite_issues': ['not eating', 'eating too much', 'weight', 'appetite', 'hungry', 'food'],
            'feeling_bad': ['failure', 'disappointed', 'let down', 'guilt', 'shame', 'blame'],
            'concentration': ['can\'t concentrate', 'focus', 'distracted', 'mind wandering', 'attention'],
            'moving_slowly': ['moving slow', 'restless', 'fidgety', 'agitated', 'pacing'],
            'self_harm': ['better off dead', 'hurt myself', 'suicidal', 'end it all', 'no point living']
        }

    def check_with_embeddings(self, text, threshold=0.4):
        text_emb = self.sentence_model.encode([text])[0]
        matches = {}
        for cat, emb in self.embeddings.items():
            sim = cosine_similarity([text_emb], [emb])[0][0]
            if sim > threshold:
                matches[cat] = sim
        return {'categories': matches, 'score': sum(matches.values())}

    def check_with_keywords(self, text):
        text = text.lower()
        matches = {}
        for cat, keys in self.keywords.items():
            for k in keys:
                if k in text:
                    matches.setdefault(cat, []).append(k)
        return {'categories': matches, 'score': len(matches)}

    def hybrid_check(self, text):
        emb = self.check_with_embeddings(text)
        key = self.check_with_keywords(text)
        combined = {}

        for cat, sim in emb['categories'].items():
            combined[cat] = {'similarity': sim, 'keywords': []}
        for cat, kws in key['categories'].items():
            if cat in combined:
                combined[cat]['keywords'] = kws
            else:
                combined[cat] = {'similarity': 0.0, 'keywords': kws}

        score = emb['score'] + 0.5 * key['score']
        return {'categories': combined, 'score': score}

# Define a class to filter and tag utterances
class ImportanceFilter:
    def __init__(self, sentiment_model, phq9_detector):
        self.sentiment_model = sentiment_model
        self.phq9_detector = phq9_detector

    def calculate_importance_score(self, utterance):
        sentiment_score = self.sentiment_model.analyze(utterance)
        phq9_score = self.phq9_detector.hybrid_check(utterance)['score']
        importance_score = 0

        if abs(sentiment_score) > 0.5:
            importance_score += abs(sentiment_score) * 2

        importance_score += phq9_score * 1.5
        return importance_score

    def label_importance(self, line, threshold=0.3):
        match = re.match(r'(.*?\(Emotion=[^)]*\))\s*(.*?:)\s*(.*)', line)
        if match:
            prefix, speaker_info, utterance = match.groups()
            utterance = utterance.strip()
            score = self.calculate_importance_score(utterance)
            relevance_label = "(Relevance=High)" if score >= threshold else "(Relevance=Low)"
            return f"{prefix} {relevance_label}: {utterance}"
        return line



def filter_dialogue_file(input_file_path, output_file_path, importance_filter, threshold=0.5):
    with open(input_file_path, "r", encoding="utf-8") as file:
        data = file.read()

    conversations = data.strip().split("=====")
    tagged_conversations = []

    for convo in conversations:
        convo = convo.strip()
        if not convo:
            continue

        lines = convo.split('\n')
        tagged_lines = []

        for line in lines:
            if line.startswith("Patient") or line.startswith("Therapist"):
                tagged_line = importance_filter.label_importance(line, threshold)
                tagged_lines.append(tagged_line)
            else:
                tagged_lines.append(line)

        if any(line.startswith("Patient") or line.startswith("Therapist") for line in tagged_lines):
            tagged_conversations.append("\n".join(tagged_lines))

    final_output = "\n=====\n".join(tagged_conversations)

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(final_output.strip())

if __name__ == "__main__":
    sentiment_model = SentimentModel("fine-tuned-sentiment-model")
    phq9_detector = PHQ9Detector()
    importance_filter = ImportanceFilter(sentiment_model, phq9_detector)
    
    filter_dialogue_file('train_dialogues.txt', 'train_dialogues_f.txt', importance_filter)
    filter_dialogue_file('val_dialogues.txt', 'val_dialogues_f.txt', importance_filter)
    filter_dialogue_file('test_dialogues.txt', 'test_dialogues_f.txt', importance_filter)
