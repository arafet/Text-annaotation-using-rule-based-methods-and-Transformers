import numpy as np
import os
import torch
import nltk
import pandas as pd
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Argument parsing for the model path
def parse_args():
    parser = argparse.ArgumentParser(description="Prediction script for sentence categorization")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the model checkpoints")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the CSV file containing the dataset")
    return parser.parse_args()

# Tokenizer and model loading based on the argument
def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # Load the best checkpoint from the model directory
    checkpoint_folders = [folder for folder in os.listdir(model_dir) if folder.startswith("checkpoint-")]
    max_value = max([int(folder.replace("checkpoint-", "")) for folder in checkpoint_folders])
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_dir}/checkpoint-{max_value}/", num_labels=3)
    return tokenizer, model

# Custom Trainer class (could be extended if needed)
class CustomTrainer(Trainer):
    pass

# Preprocess the dataset to match the expected input format for the model
def preprocess(sentence, tokenizer):
    encoded_sentence = tokenizer([i for i in sentence['content']], truncation=True)
    return {'input_ids': encoded_sentence['input_ids'], 'attention_mask': encoded_sentence['attention_mask']}

# Make predictions on the dataset and compare with actual labels
def make_predictions(df, trainer, tokenizer):
    label_map = {0: 'other', 1: 'scheduled activity', 2: 'reminder'}

    # Preprocess and predict
    dataset = Dataset.from_pandas(df[['content']])
    tokenized_dataset = dataset.map(lambda x: preprocess(x, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['content'])

    # Get model predictions
    output = trainer.predict(tokenized_dataset)
    predicted_labels = output.predictions.argmax(axis=-1)

    # Map predicted labels to their corresponding text
    predicted_labels_text = [label_map[pred] for pred in predicted_labels]
    return predicted_labels_text

if __name__ == "__main__":
    args = parse_args()

    # Load model and tokenizer from specified model directory
    tokenizer, model = load_model_and_tokenizer(args.model_dir)

    # Data collator and trainer setup
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator,
    )

    # Load dataset from CSV
    df = pd.read_csv(args.data_file)
    df = df.dropna(subset=['content', 'label'])

    # Make predictions
    df['predicted_label'] = make_predictions(df, trainer, tokenizer)

    # Evaluate model performance
    accuracy = accuracy_score(df['label'], df['predicted_label'])
    precision, recall, f1_score, _ = precision_recall_fscore_support(df['label'], df['predicted_label'], average='weighted')

    print(f'Overall Accuracy: {accuracy:.2f}')
    print(f'Overall Precision: {precision:.2f}')
    print(f'Overall Recall: {recall:.2f}')
    print(f'Overall F1-Score: {f1_score:.2f}')

    # Save results to CSV
    df.to_csv('prediction_results.csv', index=False)
    print("Results saved to 'prediction_results.csv'")
