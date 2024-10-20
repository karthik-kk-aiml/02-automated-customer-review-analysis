"""
This module summarizes product reviews using the T5-Base model and provides
blog-style summaries, highlights, and issues for each product category. 
Then it aggregates all the summaries creating a gradio app & html wireframe file.
"""

# Standard imports
import re
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from bs4 import BeautifulSoup
import gradio as gr

# Set device for torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the T5-Base model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(DEVICE)
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Load the K-means clustering output dataset
FILE_PATH = "categorized_dataset_k5_with_names.csv"
df = pd.read_csv(FILE_PATH)

# Data cleaning: Convert reviews.text to lowercase and remove NULLs
df["reviews.text"] = df["reviews.text"].astype(str).str.lower()
df = df[df["reviews.text"].notnull()]

# Group all reviews under each category
grouped_reviews = (
    df.groupby("category_name")["reviews.text"]
    .apply(" ".join)  # Lambda replaced by str.join
    .reset_index()
)

# Clear the CUDA cache
torch.cuda.empty_cache()

# List of common pronouns to remove
PRONOUNS = [
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
    "us",
    "me",
    "ll",
    "have",
]


# Function to remove pronouns
def remove_pronouns(text):
    """Removes pronouns from the given text."""
    pattern = r"\b(?:{})\b".format("|".join(PRONOUNS))
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


# Summarization function for blog-style summaries
def generate_blog_style_summary(text):
    """Generates a blog-style summary for the given text."""
    cleaned_text = remove_pronouns(text)
    input_text = f"summarize: write a blog-style summary about the product features and exclude any personal mentions. {cleaned_text}"  # Changed to f-string
    inputs = tokenizer.encode(
        input_text, return_tensors="pt", max_length=512, truncation=True
    ).to(DEVICE)

    summary_ids = model.generate(
        inputs,
        max_length=300,
        min_length=150,
        num_beams=6,
        length_penalty=2.5,
        early_stopping=True,
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Generate blog-style summaries for each product category
grouped_reviews["blog_style_summary"] = grouped_reviews["reviews.text"].apply(
    generate_blog_style_summary
)

# Save the final summaries to a CSV file
grouped_reviews.to_csv("T5-base_summary_prefinal.csv", index=False)

# Write the summaries to an HTML file
with open(
    "T5-base_summary_prefinal_1.html", "w", encoding="utf-8"
) as f:  # Encoding specified
    for _, row in grouped_reviews.iterrows():
        f.write(f"<h2>Product: {row['category_name']}</h2>\n")
        f.write(f"<p>{row['blog_style_summary']}</p>\n")
        f.write("<hr>\n")

# Save the model and tokenizer
model.save_pretrained("./summarizer-T5_Base_Prefinal")
tokenizer.save_pretrained("./summarizer-T5_Base_Prefinal")


# Function to generate general and issue summaries for each category
def generate_summaries(dataframe):
    """Generates general and issue summaries for each product category."""
    results = []

    for category in dataframe["category_name"].unique():
        category_df = dataframe[dataframe["category_name"] == category]

        # General summary (label = 2)
        general_reviews = " ".join(
            category_df[category_df["label"] == 2]["reviews.text"].tolist()
        )
        general_summary = generate_blog_style_summary(general_reviews)

        # Issue summary (label = 0)
        issue_reviews = " ".join(
            category_df[category_df["label"] == 0]["reviews.text"].tolist()
        )
        issue_summary = generate_blog_style_summary(issue_reviews)

        results.append(
            {
                "category_name": category,
                "general_summary": general_summary,
                "issues_summary": issue_summary,
            }
        )

    return pd.DataFrame(results)


# Generate summaries
summaries_df = generate_summaries(df)

# Save to HTML
with open(
    "T5-base_summary_prefinal_2.html", "w", encoding="utf-8"
) as f:  # Encoding specified
    for _, row in summaries_df.iterrows():
        f.write(f"<h2>Product: {row['category_name']}</h2>\n")
        f.write(f"<h3>Highlights:</h3>\n<p>{row['general_summary']}</p>\n")
        f.write(f"<h3>Issues:</h3>\n<p>{row['issues_summary']}</p>\n")
        f.write("<hr>\n")


# Function to load and parse HTML file
def load_html_file(html_file_path):
    """Loads and parses the HTML content."""
    with open(html_file_path, "r", encoding="utf-8") as f:  # Encoding specified
        html_content = f.read()
    return BeautifulSoup(html_content, "html.parser")


# Load HTML files
general_soup = load_html_file("T5-base_summary_prefinal_1.html")  # General summaries
issues_soup = load_html_file("T5-base_summary_prefinal_2.html")  # Highlights and issues

# Combine the content by matching category_name
final_html = "<html><body>\n"
issues_dict = {}

# Create dictionary from highlights and issues file
for h2 in issues_soup.find_all("h2"):
    category = h2.get_text()
    highlights = h2.find_next("h3", text="Highlights:").find_next("p").get_text()
    issues = h2.find_next("h3", text="Issues:").find_next("p").get_text()
    issues_dict[category] = {"highlights": highlights, "issues": issues}

# Combine content by matching category_name
for h2 in general_soup.find_all("h2"):
    category = h2.get_text()
    summary = h2.find_next("p").get_text()

    final_html += f"<h2>{category}</h2>\n"
    final_html += f"<h3>General Summary:</h3>\n<p>{summary}</p>\n"

    # Add highlights and issues if available
    if category in issues_dict:
        final_html += (
            f"<h3>Highlights:</h3>\n<p>{issues_dict[category]['highlights']}</p>\n"
        )
        final_html += f"<h3>Issues:</h3>\n<p>{issues_dict[category]['issues']}</p>\n"

    final_html += "<hr>\n"

final_html += "</body></html>"

# Save the combined result to a new HTML file
with open(
    "T5_Base_Final_Product_Summary.html", "w", encoding="utf-8"
) as f:  # Encoding specified
    f.write(final_html)

print("Final HTML output created successfully.")

# Load the final HTML content
HTML_FILE_PATH = "T5_Base_Final_Product_Summary.html"
products_data = load_html_file(HTML_FILE_PATH)


# Gradio function to show product details
def show_product_details(product_name):
    """Returns the summary, highlights, and issues for the selected product."""
    for product in products_data.find_all("h2"):
        if product.get_text() == product_name:
            general_summary = product.find_next("p").get_text()
            highlights = (
                product.find_next("h3", text="Highlights:").find_next("p").get_text()
            )
            issues = product.find_next("h3", text="Issues:").find_next("p").get_text()
            return general_summary, highlights, issues
    return "Not available", "Not available", "Not available"


# Gradio interface setup
product_list = [h2.get_text() for h2 in products_data.find_all("h2")]

gr.Interface(
    fn=show_product_details,
    inputs=gr.Dropdown(choices=product_list, label="Select Product"),
    outputs=[
        gr.Textbox(label="General Summary"),
        gr.Textbox(label="Highlights"),
        gr.Textbox(label="Issues"),
    ],
    title="Product Review Summaries",
    description="Select a product to view the general summary, highlights, and issues.",
).launch(share=True)
