from libraries import *

def preprocess_data(folder_path, output_file):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    with open(output_file, "w", encoding="utf-8") as f_out:
        for file_path in csv_files:
            df = pd.read_csv(file_path)

            try:
                primary_topic = df[df["Utterance"] == "primary_topic"]["Sub topic"].values[0]
                secondary_topic = df[df["Utterance"] == "secondary_topic"]["Sub topic"].values[0]
                summary = df[df["Utterance"] == "summary"]["Sub topic"].values[0]
            except IndexError:
                primary_topic = "Unknown"
                secondary_topic = "Unknown"
                summary = "No summary available"

            df = df[~df["Utterance"].isin(["primary_topic", "secondary_topic", "summary"])]

            df["Emotion"] = pd.to_numeric(df["Emotion"], errors="coerce").fillna(0).astype(int)

            global_emotion = df["Emotion"].mean()

            f_out.write(f"<PrimaryTopic: {primary_topic}> <SecondaryTopic: {secondary_topic}> <GlobalEmotion: {global_emotion:.2f}>\n")

            for _, row in df.iterrows():
                speaker = "Therapist" if row["Type"] == "T" else "Patient"
                dialogue_function = row.get("Dialogue_Act", "unknown")  # Handle missing column
                sub_topic = row["Sub topic"] if pd.notna(row["Sub topic"]) else "general"
                emotion = int(row["Emotion"])
                utterance = row["Utterance"]
                
                f_out.write(f"{speaker} [DialogueFunction={dialogue_function}] (Sub_topic={sub_topic}) (Emotion={emotion}): {utterance}\n")

            f_out.write(f"Summary: {summary}\n\n")
            f_out.write("\n" + "="*80 + "\n\n")

    print(f" Preprocessed data saved to {output_file}")

if __name__=="__main__":
    
datasets = {
    "Train": "train_dialogues.txt",
    "Validation": "val_dialogues.txt",
    "Test": "test_dialogues.txt"
}

for folder, output in datasets.items():
    preprocess_data(folder, output)