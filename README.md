Sentilytics - Social Media Sentiment Analyser
A tool to analyze sentiment from social media text data using Natural Language Processing (NLP) techniques. This project processes datasets from platforms like Twitter, Reddit, and YouTube to classify text into positive, negative, and neutral sentiment.

📌 Project Overview
This project implements an automated Social Media Sentiment Analysis system that classifies user-generated text into Positive, Negative, or Neutral sentiments.
The system combines traditional Machine Learning techniques with Large Language Models (BERT) to provide accurate and context-aware sentiment predictions.
The project was developed as part of a Skill Internship Program and is inspired by recent academic research in sentiment analysis.

📚 Research Reference
This project is inspired by and aligned with the research paper:
“Social Media Sentiment Analysis”, Encyclopedia, 2024.
The methodology and results of this implementation closely match the findings discussed in the paper.

📂 Repository Structure
social_media_sentiment_analyser/
├── .config/                    # Environment/configuration files
├── .gradio/                    # UI-related config (Gradio app)
├── sample_data/                # Folder for sample/training datasets
│   ├── Reddit_Data.csv         # Reddit comments dataset
│   ├── Twitter_Data.csv        # Twitter posts dataset
│   ├── YoutubeCommentsDataSet.csv # YouTube comments dataset
│   └── sentimentdataset.csv    # Combined sentiment dataset
├── main.py / app.py            # Entry point for running the app
├── requirements.txt            # Python package dependencies
├── README.md                  # This file
└── LICENSE                    # License file

🚀 Features
✔️ Multi-platform sentiment analysis – supports Twitter, Reddit, and YouTube data
✔️ Data preprocessing – text cleaning, tokenization, and normalization
✔️ Sentiment prediction – classifies text as positive, negative, or neutral
✔️ Interactive UI – Optional Gradio interface for real-time testing
✔️ Multiple datasets supported – ready-to-use .csv data for experimentation

🗂️ Dataset
The following datasets were used and merged:
• Twitter Dataset
• Reddit Dataset
• YouTube Comments Dataset
• Generic Sentiment Dataset
All datasets were cleaned, standardized, and combined to ensure data diversity and robustness.

🛠️ Tech Stack
➤ Programming Language: Python
➤ Libraries: pandas, numpy, nltk, scikit-learn
➤ LLM: BERT (Hugging Face Transformers)
➤ Feature Extraction: TF-IDF
➤ ML Model: Multinomial Naive Bayes
➤ UI: Gradio
➤ Environment: Google Colab

📌 Example Usage
from sentiment_analyser import analyse_sentiment
text = "I love how helpful this project is!"
result = analyse_sentiment(text)
print(result) # Positive / Negative / Neutral

📊 Results
• The ML model provides fast and efficient sentiment predictions
• BERT demonstrates superior contextual understanding
• Results align with existing academic research on sentiment analysis

📌 Applications
• Public opinion analysis
• Political sentiment monitoring
• Brand reputation analysis
• Social media trend analysis

🔮 Future Scope
➤ Real-time sentiment analysis
➤ Multilingual sentiment detection
➤ Multimodal sentiment analysis
➤Cloud deployment

👥 Contributors
Praman Jain & Prasun Singh

📄 License
This project is for educational and research purposes only.
