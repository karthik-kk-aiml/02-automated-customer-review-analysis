"""
This module uses the RoBERTa model for sentiment analysis on customer reviews.
It performs fine-tuning, evaluation, and sentiment prediction. 
"""

# Standard imports
import os

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# pylint: disable=too-many-ancestors
# pylint: disable=no-member


# Environment configuration
os.environ["WANDB_DISABLED"] = "true"

# Set device for torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
df = pd.read_csv(r"dataset/1429_1.csv")

# Drop rows where rating is missing
df = df.dropna(subset=["reviews.rating"])

def label_sentiment(rating):
    """
    Classifies sentiment based on the rating.
    1-2: Negative, 3: Neutral, 4-5: Positive.
    """
    if rating in [1, 2]:
        return 0  # Negative
    if rating == 3:
        return 1  # Neutral
    return 2  # Positive

# Apply the function to create a new column
df["label"] = df["reviews.rating"].apply(label_sentiment)

# Drop rows where 'reviews.text' is missing
df = df.dropna(subset=["reviews.text"])

# Split dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["reviews.text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenize the text data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class SentimentDataset(torch.utils.data.Dataset):
    """Custom dataset class for sentiment analysis."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["target_labels"] = torch.tensor(self.labels[idx])  # renamed from 'labels'
        return item

    def __len__(self):
        return len(self.labels)

# Create PyTorch datasets
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Compute class weights based on training labels
class_weights = compute_class_weight("balanced", classes=[0, 1, 2], y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

class WeightedRobertaForSequenceClassification(RobertaForSequenceClassification):
    """Custom RoBERTa model class with weighted loss for class imbalance."""
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, target_labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        logits = outputs.logits
        if target_labels is not None:
            self.class_weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), target_labels.view(-1))
            return torch.nn.functional.softmax(logits, dim=-1)
        return outputs

# Initialize the custom model with class weights
model = WeightedRobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=3, class_weights=class_weights_tensor
)

# Move model to the appropriate device
model.to(DEVICE)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="steps",
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    eval_steps=500,
)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Get predictions from validation dataset
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids  # Renamed from 'labels'

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")

# Print evaluation results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Compute and display confusion matrix
conf_matrix = confusion_matrix(true_labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, annot=True, fmt="d", cmap="viridis", cbar=False,
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"]
)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.show()

# Save the model and tokenizer
model.save_pretrained("./sentiment-analysis-roberta-Classweight3")
tokenizer.save_pretrained("./sentiment-analysis-roberta-Classweight3")
