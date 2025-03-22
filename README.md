# Fake News Detection in Malaysian Bilingual Political Context
This repository documents a deep learning project aimed at the automatic detection of political fake news in Malaysia, leveraging bilingual datasets and advanced deep learning techniques. The project includes data collection, processing, model development, and a user-friendly web application for fake news detection.

# Overview
This project focuses on the detection of political fake news in English and Malay languages within the Malaysian context. It combines a bilingual approach to dataset development and the implementation of state-of-the-art deep learning architectures. Below are the highlights of the project:

# Key Features
1. Systematic Review
    - A systematic review was conducted, analyzing 114 relevant and recent studies on fake news detection, bilingual documents, and deep learning technologies.
    - This review provided a deep understanding of effective deep learning architectures and their applications in fake news detection.
  
2. Curated Datasets
    - Primary Data Collection:
      - Collected 127,743 English and 28,646 Malay political news articles from renowned Malaysian sources like AstroAwani, FreeMalaysiaToday (FMT), MalaysiaKini, New Straits Times, Sebenarnya.my, and others.
    - Open-Source Data Integration:
      - Incorporated datasets from platforms like Mesolitica GitHub community on Kaggle, which included news from The Star and Berita RTM.
    - Bilingual Pre-Processing:
      - Employed tailored pre-processing techniques for English and Malay datasets to ensure seamless data integration and high-quality preparation.
        
3. Exploratory Data Analysis (EDA)
    - Conducted in-depth EDA to uncover linguistic patterns unique to Malaysian political fake news.
    - Utilized graphs and charts to derive insights into frequent words, linguistic structures, and fake news narratives.
      
4. Deep Learning Models
    - Developed, trained, and fine-tuned five models: CNN, LSTM, Bi-LSTM, BERT, and HAN.
    - Performance was evaluated using the F1-score:
      - English Dataset: Average F1-score of 0.95 (Best: CNN with 0.9679).
      - Malay Dataset: Average F1-score of 0.85 (Best: BERT with 0.8679).
    - Results demonstrated the models' ability to effectively handle bilingual datasets and accurately detect misinformation.
      
5. Web Application
    - Deployed a GUI-based web application for fake news detection:
      - Input: News title and content.
      - Output: Clear classification result (real or fake) with a confidence score.
      - Features an interactive dashboard showcasing the results of EDA, including insights into frequent words, linguistic patterns, and other key findings.
