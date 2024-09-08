import argparse
import numpy as np
import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from collections import defaultdict
from transformers import DataCollatorWithPadding, GPT2Config
import matplotlib.pyplot as plt
import transformers
# Argument Parser to handle model name input
def parse_args():

    parser = argparse.ArgumentParser(description="Train a transformer model for sequence classification.")
    parser.add_argument('--model_name', type=str, required=True, 
                        choices=['gpt2'],
                        help="Specify the model to use: distilbert, bert, albert, or roberta.")
    return parser.parse_args()

# Function to choose model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    configuration = GPT2Config.from_pretrained("gpt2")
    configuration.num_labels = 3
    configuration.pad_token_id = configuration.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(model_name,  config=configuration)

    assert model.config.pad_token_id == model.config.eos_token_id

    return model, tokenizer

# Main function to run the training process
def main():
    print("transformer:",transformers.__version__)

    # Parse arguments
    args = parse_args()
    model_name = args.model_name
    
    print(f"Using model: {model_name}")
    
    # Load the dataset
    dataset = load_dataset('csv', data_files=f'./data/annotated_data_balanced.csv')

    # Split the dataset into training and validation
    train_data, val_data = train_test_split(dataset['train'], test_size=0.1, stratify=dataset['train']['label'], random_state=42)

    # Convert to Dataset format
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})

    # Define one-hot encoding function
    def one_hot_encoding(data):
        categories = ['other', 'scheduled activity', 'reminder']
        labels = []
        for label in data:
            label_ids = categories.index(label)
            one_hot = [0] * len(categories)
            one_hot[label_ids] = 1
            labels.append(one_hot)
        return labels

    # Encode the dataset
    def encode(example, tokenizer):
        content = example['content']
        label = example['label']
        encoded_example = tokenizer(content, truncation=True)
        encoded_label = one_hot_encoding(label)
        return {
            'input_ids': encoded_example['input_ids'], 
            'attention_mask': encoded_example['attention_mask'], 
            'label': encoded_label
        }

    # Choose model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Tokenize the dataset
    tokenized_dataset = dataset_dict.map(lambda x: encode(x, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['content'])

    # Check for existing checkpoints
    model_name_map = {
    "gpt2":"gpt2",
}

    checkpoint_dir = f"./checkpoints/{model_name_map[model_name]}"
    checkpoint_path = None  # Default is None, i.e., no checkpoint

    # If the directory is not empty, find the latest checkpoint
    if os.listdir(checkpoint_dir):
        checkpoint_folders = [folder for folder in os.listdir(checkpoint_dir) if folder.startswith("checkpoint-")]
        if checkpoint_folders:
            max_checkpoint = max([int(folder.split("-")[1]) for folder in checkpoint_folders])
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{max_checkpoint}/")
            print(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            print("No valid checkpoints found, starting from scratch.")
    else:
        print("Checkpoint directory is empty. Starting training from scratch.")

    # Custom Trainer class for weighted loss
    class CustomTrainer(Trainer):
        def __init__(self, *args, label_counts=None, **kwargs):
            super().__init__(*args, **kwargs)
            if label_counts:
                n_samples = sum(label_counts)
                self.class_weights = torch.tensor([n_samples / label_counts[i] for i in range(self.model.config.num_labels)])
                self.class_weights = self.class_weights.to(self.model.device)
        
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                #token_type_ids=inputs['token_type_ids']
            )
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(outputs.logits.float(), inputs['labels'].float())
            return (loss, outputs) if return_outputs else loss

    # Calculate class weights
    nb_reminder = len(np.nonzero(np.array(train_dataset['label']) == 'other')[0])
    nb_task_todo = len(np.nonzero(np.array(train_dataset['label']) == 'reminder')[0])
    nb_task_meet = len(np.nonzero(np.array(train_dataset['label']) == 'scheduled activity')[0])
    label_counts = [nb_reminder, nb_task_meet, nb_task_todo]

    # Freeze all but the last layer for fine-tuning
# Freeze all but the last layer
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.score.parameters():
        param.requires_grad = True

    # Set up the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=os.path.abspath(f"checkpoints/{model_name_map[model_name]}"),
        #output_dir=f"/",
        learning_rate=2e-5,
        save_steps=5000,
        save_total_limit=10,
        load_best_model_at_end=True,
        #resume_from_checkpoint=f"checkpoints/{model_name_map[model_name]}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=9,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=1000
    )

    # Initialize the trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        label_counts=label_counts
    )

    # Train the model from the last checkpoint (if available)
    print('checkpoint_path')
    if checkpoint_path:
    # Update trainer state with the new checkpoint path
       if trainer.state.best_model_checkpoint:
          trainer.state.best_model_checkpoint = checkpoint_path
   # trainer.state.best_model_checkpoint = None  # Reset the best checkpoint

    print("*************************************************************************",checkpoint_path)
    trainer.train(resume_from_checkpoint=checkpoint_path)
    

    # Plot training/validation losses
    train_logs = trainer.state.log_history
    training_loss = [entry.get('loss') for entry in train_logs if 'loss' in entry]
    validation_loss = [entry.get('eval_loss') for entry in train_logs if 'eval_loss' in entry]

    # Remove None values
    training_losses = [loss for loss in training_loss if loss is not None]
    validation_losses = [loss for loss in validation_loss if loss is not None]

    # Plotting losses
    epochs = np.arange(len(training_losses))
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, np.interp(epochs, np.arange(len(validation_losses)), validation_losses), label='Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print(transformers.__version__)
    main()
