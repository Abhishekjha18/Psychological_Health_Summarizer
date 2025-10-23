# Mental Health Summarization

## Dataset

The dataset used in this project consists of therapist-client conversations, providing a rich source of dialogue data for summarization tasks. It includes a total of **190** samples, which are divided into three subsets:

- **Training Set**: 131 samples (70%)
- **Validation Set**: 21 samples (10%)
- **Testing Set**: 39 samples (20%)

Each sample in the dataset contains:
- The full conversation between the therapist and the client.
- Extracted primary and secondary topics relevant to the conversation.
- A corresponding summary that encapsulates the key points of the dialogue.

## Data Preprocessing

### 4.1 Preprocessing and Structuring Dialogues

1. **Remove Non-Essential Entries**
   - Eliminate metadata fields such as `primary_topic`, `secondary_topic`, and `summary`.
   - Remove filler words or entries not useful for downstream tasks.

2. **Compute Global Emotion Score**
   - Average all emotion values across the dialogue to compute a single emotion score representing the emotional tone of the conversation.

3. **Restructure Utterances**
   - Each utterance is reformatted for consistency and to preserve key contextual information:
     ```
     Speaker [DialogueFunction=...] (Sub_topic=...) (Emotion=...): <utterance_text>
     ```
   - Example:
     ```
     Therapist [DialogueFunction=question] (Sub_topic=routine) (Emotion=1): How have you been sleeping lately?
     ```

4. **Flatten Dialogues**
   - Dialogues are flattened into a list of structured utterances to facilitate model input.

---

### 4.2 Modeling Relevance using Sentiment and PHQ-9 Cues

1. **Sentiment Classification**
   - A RoBERTa-base model is fine-tuned on the dialogue data to classify each utterance into:
     - Positive
     - Neutral
     - Negative
   - **Sentiment Score** is computed using model probabilities:
     ```
     Sentiment Score = P_positive − P_negative
     ```

2. **PHQ-9 Signal Detection**
   - Each utterance is evaluated based on two PHQ-9-based components:
     - **Keyword Match Score**: Presence of depression-related terms (e.g., "worthless", "fatigue", "better off dead").
     - **Embedding Similarity Score**: Cosine similarity with PHQ-9 item sentence embeddings using `all-MiniLM-L6-v2`.

   - **PHQ Score** is calculated as:
     ```
     PHQ Score = α · Embedding Score + β · Match Score
     ```
     Where:
     - α = 1.0
     - β = 0.5

3. **Compute Relevance Score**
   - Final relevance score for each utterance:
     ```
     Relevance Score = 2 · |Sentiment Score| + 1.5 · PHQ Score
     ```

4. **Label Relevance**
   - Utterances are tagged as:
     - `(Relevance=High)` if the score exceeds a predefined threshold
     - `(Relevance=Low)` otherwise

5. **Append Relevance Labels**
   - These labels are added to each utterance's metadata from the previous preprocessing step and used as auxiliary signals during summarization model training.


## Methodology

### Baselines

Our methodology begins with strong pretrained **abstractive summarization models**, which serve both as **benchmarks** and as **foundational architectures** for subsequent enhancements.

We selected the following baseline models:

- **Pegasus-large**
  - Pre-trained using a **gap-sentence generation** objective.
  - Optimized for **information-dense inputs**, making it well-suited for therapeutic dialogues.

- **T5-large**
  - Based on a **text-to-text** framework.
  - Generalizes well across a variety of NLP tasks: summarization, question answering, and translation.

**Fine-Tuning Strategy:**
- Both models were fine-tuned on the counseling dataset.
- No domain-specific augmentations were applied at this stage.
- Training data consisted of:
  - **Input**: Structured dialogues from the preprocessing step.
  - **Target**: Ground-truth summaries.

These results provide a reference point for model performance **without relevance-based supervision**.

---

### Relevance-Guided Summarization

Building upon the baseline models, we introduce a **relevance-augmented training setup** using the **Flan-T5-large** model.

#### Relevance-Aware Input Representation

- Each utterance in a dialogue is enriched with:
  - `Speaker`
  - `DialogueFunction`
  - `Sub_topic`
  - `Emotion`
  - `Relevance` score computed from sentiment and PHQ-9 cues

- Example enriched utterance:
   -Therapist [DialogueFunction=question] (Sub_topic=routine) (Emotion=1) (Relevance=High): How have you been sleeping lately?
   
- These tags provide **clinical and emotional cues** to help the model prioritize therapeutically significant content.

#### Modeling Strategy

- **Model Used**: Flan-T5-large
- **Input**: Relevance-tagged dialogues
- **Target**: Ground-truth summaries (unchanged)
- **Architecture**: No modifications were made to the model.

**Why Flan-T5?**
- It is instruction-tuned and excels at **dialogue-centric** NLP tasks.
- Trained on a diverse mixture of tasks including:
- Summarization
- Instruction-following
- Conversational modeling

#### Objective

This strategy tests whether **lightweight, input-level supervision** via relevance tagging can guide the model to produce:
- More **clinically meaningful**
- More **emotionally resonant**
summaries, without requiring architectural changes.


## Results

The following table presents a quantitative comparison between the baseline models (Pegasus and T5) and our relevance-guided model (FlanT5 with rel score). Evaluation metrics include ROUGE scores, BLEU, and BERTScore—each offering different insights into the quality and fidelity of generated summaries.

| Metric     | Pegasus | T5    | FlanT5 with rel score |
|------------|---------|-------|------------------------|
| ROUGE-1    | 31.35   | 35.04 | 39.31                  |
| ROUGE-2    | 11.18   | 12.30 | 15.30                  |
| ROUGE-L    | 19.68   | 21.53 | 24.70                  |
| BLEU       | 2.63    | 2.62  | 5.04                   |
| BERTScore  | 85.78   | 86.69 | 87.08                  |

The relevance-guided FlanT5 model outperformed both baselines across all metrics, indicating that the use of sentiment and PHQ-9-based relevance cues led to more informative and clinically aligned summaries. Notably, improvements are significant in ROUGE-1 and BLEU, suggesting better content overlap and fluency in generated summaries.


## How to Run
git clone the repo
cd Mental-Health-Summarization
For running the baseline models you can use the already existing  train_dialogues.txt, val_dialogues.txt and test_dialogues.txt files whereas for our main model you would need train_dialogues_f.txt, val_dialogues_f.txt and test_dialogues_f.txt
Then run py files in this order
-preprocessing.py
-SA_Training.py
-scores.py
-main.py
