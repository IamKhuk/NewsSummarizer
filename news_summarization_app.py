import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
from datasets import load_dataset, Dataset
import nltk
from collections import Counter
import heapq
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, BartForConditionalGeneration, BartTokenizer
import evaluate
import ssl

# Fix SSL issue for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Ensure necessary NLTK resources are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load the CNN/DailyMail dataset
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
rouge = evaluate.load("rouge")

# Preprocessor class
class Preprocessor:
    def __init__(self):
        self.stop_words = stop_words
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        sentences = nltk.sent_tokenize(text)
        preprocessed_sentences = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            words = [word.lower() for word in words if word.isalnum()]
            words = [word for word in words if word not in self.stop_words]
            words = [self.lemmatizer.lemmatize(word) for word in words]
            preprocessed_sentences.append(words)
        return preprocessed_sentences

# Summarization class
class NewsSummarization:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def extractive_summary(self, text, num_sentences=3):
        sentences = nltk.sent_tokenize(text)

        # Handle short text: If fewer sentences than requested, return all sentences
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Compute TF-IDF scores
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Normalize scores by sentence length and flatten to list
        sentence_scores = (tfidf_matrix.sum(axis=1).A1 / (tfidf_matrix != 0).sum(axis=1).A1).tolist()

        # Rank sentences based on scores
        ranked_indices = sorted(range(len(sentence_scores)), key=lambda x: sentence_scores[x], reverse=True)
        ranked_sentences = [sentences[i] for i in ranked_indices]

        # Return the top `num_sentences`
        summary = " ".join(ranked_sentences[:num_sentences])
        return summary

    def abstractive_summary(self, text):
        result = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
        return result[0]['summary_text']

    def evaluate_summary(self, actual_summary, generated_summary):
        scores = rouge.compute(predictions=[generated_summary], references=[actual_summary])
        return {key: round(value * 100, 2) for key, value in scores.items()}

    def fine_tune_model(self, user_data):
        # Load the model and tokenizer
        model_name = "facebook/bart-large-cnn"
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)

        # Prepare the dataset
        dataset = Dataset.from_pandas(user_data)
        tokenized_dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True
        )
        tokenized_dataset = tokenized_dataset.map(
            lambda x: tokenizer(x["summary"], truncation=True, padding="max_length", max_length=128), batched=True
        )

        # Fine-tune the model
        training_args = Seq2SeqTrainingArguments(
            output_dir="./fine_tuned_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            save_total_limit=1,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")

# Streamlit UI
def main():
    st.title("News Article Summarization")

    # Sidebar options
    st.sidebar.header("Options")
    summarization_type = st.sidebar.radio(
        "Select Summarization Type", ["Extractive", "Abstractive"]
    )

    # Allow user to control summary length
    summary_length = st.sidebar.slider("Number of Sentences in Summary", min_value=1, max_value=10, value=3)

    # Input text
    st.header("Input Text")
    input_text = st.text_area("Paste your text here:", height=300)

    # File uploader (optional)
    uploaded_file = st.file_uploader("Or upload a text file", type="txt")
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded Text:", input_text, height=300)

    if st.button("Summarize"):
        if not input_text.strip():
            st.warning("Please provide some text.")
        else:
            summarizer = NewsSummarization()

            if summarization_type == "Extractive":
                st.subheader("Extractive Summary")
                summary = summarizer.extractive_summary(input_text, num_sentences=summary_length)
            else:
                st.subheader("Abstractive Summary")
                summary = summarizer.abstractive_summary(input_text)

            st.write(summary)

            # Word cloud visualization
            st.subheader("Word Cloud")
            wordcloud = WordCloud(collocations=False, background_color="white").generate(input_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

            # Feedback Collection
            feedback = st.radio("Do you find this summary helpful?", ["👍 Yes", "👎 No"])
            if feedback:
                with open("feedback_log.csv", "a") as f:
                    f.write(f"{feedback},{input_text},{summary}\n")
                st.success("Thank you for your feedback!")

    # Fine-tuning section
    st.sidebar.subheader("Fine-Tune the Model")
    st.sidebar.info(
        "Upload a CSV file with two columns:\n"
        "- **text**: The original text or article\n"
        "- **summary**: The desired summary for the text.\n\n"
        "Once uploaded, click the 'Fine-Tune Model' button to train the model with your custom data."
    )
    uploaded_finetune_file = st.sidebar.file_uploader("Upload text-summary pairs (CSV)", type="csv")
    if uploaded_finetune_file:
        user_data = pd.read_csv(uploaded_finetune_file)
        st.sidebar.write("Uploaded Data:", user_data.head())
        if st.sidebar.button("Fine-Tune Model"):
            summarizer = NewsSummarization()
            summarizer.fine_tune_model(user_data)
            st.sidebar.success("Model fine-tuned successfully!")

if __name__ == "__main__":
    main()