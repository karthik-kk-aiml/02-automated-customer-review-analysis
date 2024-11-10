# PROJECT - Robo Reviews---Automated-Customer-Reviews

### Project Overview:

Robo Reviews is a customer review analysis tool designed to provide insights into Amazon products through sentiment analysis, clustering, and summarization. The approach is to use Transformer based models on the Customer reviews to perform Sentiment analysis. Classify the Customer reviews and Generate a Summary of the Reviews delivering user-friendly insights via an HTML dashboard and Gradio interface. The target audience who will consume the output is either the Client who presented their own customer reviews for their products OR the anyone who is interested to learn about the product, highlights and issues.   

### Table of Contents

- Folder Structure
- Environment Setup
- Project Components
- Usage
- Future Enhancements
  
### Folder Structure

- robo_review_kk_sentianalysis_roberta_final.py - Sentiment Analysis script using RoBERTa.
- robo_review_kk_clustering_k_means_final.py - Product clustering script using K-means.
- robo_review_kk_summarizer_t5_base_final.py - Summarization script using T5-Base.
- Final Product Review Output_ReviewHive.html - HTML file with summarized product reviews.
- README.md - Project documentation (this file).
- requirements.txt - Lists all dependencies required for the project.
- notebook_files/ - Work-in-progress files used during project development.

### Project Components

This project was executed in 4 Sprints.
1. Sentiment Analysis (robo_review_kk_sentianalysis_roberta_final.py):
- Utilizes a RoBERTa model to label reviews as Positive, Neutral, or Negative based on ratings.
- Class weights address data imbalance, and evaluation metrics like F1 Score and Accuracy are calculated.
2. Clustering (robo_review_kk_clustering_k_means_final.py):
- Employs K-means clustering to categorize products into five main groups (e.g., Kindle, Fire HD 8).
- Dimensionality reduction (PCA) provides a visual representation of clusters.
3. Summarization (robo_review_kk_summarizer_t5_base_final.py):
- Uses T5-Base to create blog-style summaries for each product category, highlighting features and common issues.
- Generates an HTML file for easy access to product summaries.
4. Gradio Interface :
- Provides a user-friendly interface for viewing product summaries, highlights, and issues.
- Accessible through the Gradio link generated upon running the summarization script.

### Usage:
Below is the sequence of ipynb that should be run. 
1. Run Sentiment Analysis:
- robo_review_kk_sentianalysis_roberta_final.py: Performs preprocessing and performs the sentiment analysis of customer reviews.
2. Run Clustering:
- robo_review_kk_clustering_k_means_final.py: Performs clustering of the reviews based on Category names (products)
3. Run Summarization:
- robo_review_kk_summarizer_t5_base_final.py: Summarizes the Customer reviews and presents it in a user friendly fashion. 3 steps to it.
   first step provides a general summary, second one provides highlight & issues summary and third one aggregates both and produces a html output.
   This creates a Gradio app as well which could be hosted in Huggingface.
4. Access Final Output:
- Open "Final Product Review Output_ReviewHive.HTML" in a browser or use Gradio link generated. 
    
### Future Enhancements
- Enhance sentiment analysis by incorporating title, review text, and ratings.
- Refine clustering by reevaluating input data and exploring advanced classification methods.
- Improve summarization with more advanced models and optimized prompts.
