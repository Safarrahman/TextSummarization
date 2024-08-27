from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_evaluation import ModelEvaluation
from textSummarizer.logging import logger

from transformers import AutoTokenizer
from transformers import Trainer, AutoModelForSeq2SeqLM
from datasets import load_dataset

import csv

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Load your model and tokenizer with the correct paths
        model = AutoModelForSeq2SeqLM.from_pretrained("artifacts/model_trainer/t5-large-model")
        tokenizer = AutoTokenizer.from_pretrained("artifacts/model_trainer/tokenizer")

        # Load your evaluation dataset
        dataset = load_dataset("samsum", split="test")

        # Tokenize the input (dialogue) and target (summary) fields with padding and truncation
        def preprocess_function(examples):
            # Tokenize the inputs (dialogue) with padding and truncation
            model_inputs = tokenizer(examples["dialogue"], max_length=1024, truncation=True, padding="max_length")

            # Tokenize the labels (summary) with padding and truncation
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Apply the preprocessing function to the dataset
        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Prepare your trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
        )

        # Evaluate the model
        eval_results = trainer.evaluate(eval_dataset=tokenized_dataset)

        # Define the path to save the CSV file
        csv_file_path = "artifacts/model_evaluation/metrics.csv"

        # Save evaluation metrics to a CSV file
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])  # Header row
            for key, value in eval_results.items():
                writer.writerow([key, value])

        print(f"Evaluation metrics saved to '{csv_file_path}'")