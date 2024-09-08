
# Enron Email Data Annotation and Transformer Model Training

This project allows users to annotate the Enron email dataset, train various transformer-based models, and test them on the annotated dataset.

## 1. Setup

### Download the Enron Email Dataset
- Download the Enron email dataset and save it in the `/data` folder.
- Ensure the file is named `email.csv`.

### Install Required Libraries
Make sure you have all necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

### Directory Structure
Ensure your project structure looks like this:

```
/data
    email.csv  # The downloaded dataset
/checkpoints
    model_name
enron_data_annotation.py  # Script to annotate the data
train.py  # Script to train the model
train_gpt2.py #Script to train gpt2
test.py  # Script to test the model
requirements.txt
```

## 2. Annotate Data

To annotate the Enron dataset (you can skip this step if you want to train models on the annotated data):

```bash
python enron_data_annotation.py --input data/email.csv
```

This script parses the email content, extracts relevant fields, and generates a labeled dataset with categories like `Scheduled Activity`, `Reminder`, and `Other`. The annotated data will be saved to `annotated_data_balanced.csv`.

### Example of Usage:
```bash
python enron_data_annotation.py --input data/email.csv
```

## 3. Train Transformer Models

You can train different transformer models (DistilBERT, BERT, ALBERT, RoBERTa) for email classification using the provided `train.py` script.

### Supported Models:
- `distilbert-base-uncased`
- `bert-base-uncased`
- `albert-base-v2`
- `roberta-base`

### Example of Usage:
```bash
python train.py --model_name bert-base-uncased
```
For gpt2:
```bash
python train_gpt2.py --model_name gpt2
```

### Note:
If you encounter an error such as:
```bash
ValueError: 'best_model_checkpoint' is not in list
```
You can fix this by editing the `trainer_state.json` file in the checkpoint folder and updating the value of the key `best_model_checkpoint` by an absolute path.

## 4. Test the Model on Annotated Enron Dataset

To test the trained model on the annotated Enron dataset:

### Example of Usage:
```bash
python test.py --model_dir checkpoints/distilbert --data_file data/annotated_data_balanced.csv
```
