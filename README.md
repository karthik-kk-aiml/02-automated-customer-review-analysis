PROJECT - Robo Reviews---Automated-Customer-Reviews

Description of Project:
The scope of this project 'Robo Reviews' is to process the accumulated customer reviews and make sense out of it. The approach is to use Transformer based models on the Customer reviews to perform Sentiment analysis. Classify the Customer reviews and Generate a Summary of the Reviews for user friendly presentation. The target audience who will consume the output is either the Client who presented their own customer reviews for their products OR the anyone who is interested to learn about the product, highlights and issues. 

This project was executed as 4 Sprints during Week 6 of the Bootcamp
1. Sentiment Analyzer
2. Classifier
3. Summarizer
4. Final output - to generate a HTML wireframe and Gradio App (if possible host it on AWS/huggingface)

Description of Files:
From this REPO, below is the sequence of ipynb that should be run. 
1. robo_review_kk_sentianalysis_roberta_final.py: Performs preprocessing and performs the sentiment analysis of customer reviews.
2. robo_review_kk_clustering_k_means_final.py: Performs clustering of the reviews based on Category names (products)
3. robo_review_kk_summarizer_t5_base_final.py: Summarizes the Customer reviews and presents it in a user friendly fashion. 3 steps to it.
   first step provides a general summary, second one provides highlight & issues summary and third one aggregates both and produces a html output.
   This creates a Gradio app as well which could be hosted in Huggingface.
4. Final Product Review Output_ReviewHive.HTML: This is a Wireframe to inform how the actual website will look like, out put of robo_review_kk_summarizer_t5_base_final.py.
5. ReviewHive_Customer Reviews.pdf: This is the Final presentation of the project. 
    
