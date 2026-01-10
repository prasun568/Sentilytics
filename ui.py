

# ===== IMPORTS =====
import gradio as gr
import pandas as pd
import pickle, re, nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from transformers import pipeline

# ===== NLTK =====
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ===== LOAD MODEL & VECTORIZER =====
# Using the globally defined nb_model and tfidf objects directly
# as they are available in the kernel from previous cell executions.
model = nb_model
# tfidf is already globally defined and accessible, so no re-assignment is needed.

# ===== LOAD BERT =====
bert = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# ===== CLEAN TEXT =====
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join([w for w in text.split() if w not in stop_words])

# ===== TEXT ANALYSIS =====
def analyze_text(text_input):
    if text_input.strip() == "":
        return "❌ Please enter some text"

    clean = clean_text(text_input)
    vec = tfidf.transform([clean])
    ml_pred = model.predict(vec)[0]

    bert_out = bert(text_input)[0]

    return (
        f"""
✨ **Sentiment Analysis Result**

🔹 **ML Sentiment:** {ml_pred}
🔹 **BERT Sentiment:** {bert_out['label']}
🔹 **Confidence Score:** {round(bert_out['score'],3)}
        """
    )

# ===== CSV ANALYSIS =====
def analyze_csv(csv_file, text_column):
    if csv_file is None:
        return "❌ Please upload a CSV file", None

    df = pd.read_csv(csv_file.name)

    if text_column not in df.columns:
        return f"❌ Column '{text_column}' not found in CSV", None

    df['clean_text'] = df[text_column].apply(clean_text)
    vec = tfidf.transform(df['clean_text'])
    df['sentiment'] = model.predict(vec)

    counts = df['sentiment'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title("💬 Sentiment Distribution (CSV Data)")

    summary = f"""
📊 **CSV Sentiment Summary**

🧾 Total Records: {len(df)}

😊 Positive: {counts.get('positive',0)}
😐 Neutral: {counts.get('neutral',0)}
😡 Negative: {counts.get('negative',0)}
    """

    return summary, fig


# ===== UI LAYOUT =====
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    #  Sentilytics
    ### Social Media Sentiment Analysis using ML & BERT

    Analyze public sentiment from **Text or CSV data**
    """)

    with gr.Tabs():

        # ----- TAB 1: TEXT -----
        with gr.Tab("📝 Analyze Text"):
            gr.Markdown("Enter a sentence to instantly detect sentiment")
            text_input = gr.Textbox(
                lines=4,
                placeholder="Type your text here…",
                label="Input Text"
            )
            text_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
            text_output = gr.Markdown()

            text_btn.click(
                analyze_text,
                inputs=text_input,
                outputs=text_output
            )

        # ----- TAB 2: CSV -----
        with gr.Tab("📂 Analyze CSV"):
            gr.Markdown("📊 Upload bulk feedback data and visualize sentiment")
            csv_file = gr.File(label="Upload CSV file")
            text_column = gr.Textbox(
                value="text",
                label="Text Column Name",
                placeholder="e.g. text, review, comment"
            )
            csv_btn = gr.Button("📈 Analyze CSV", variant="primary")
            csv_text = gr.Markdown()
            csv_plot = gr.Plot()

            csv_btn.click(
                analyze_csv,
                inputs=[csv_file, text_column],
                outputs=[csv_text, csv_plot]
            )

    gr.Markdown("""
    ---
    💡 **Powered by:** TF-IDF + Naive Bayes & BERT
    🎓 **Made by:** Prasun Singh & Praman Jain
    """)

demo.launch()

