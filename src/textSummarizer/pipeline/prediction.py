from textSummarizer.config.configuration import ConfigurationManager
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import re

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def preprocess_text(self, text):
        # Enhanced text preprocessing
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
        text = re.sub(r'[^a-zA-Z0-9.?! ]+', '', text)  # Remove non-alphanumeric characters (except punctuation)
        text = text.strip()  # Trim leading and trailing spaces
        return text

    def postprocess_summary(self, summary):
        # Capitalize the first letter of each sentence
        summary = re.sub(r'([.?!]\s+)([a-z])', lambda x: x.group(1) + x.group(2).upper(), summary)
        
        # Ensure the first character of the entire summary is capitalized
        if summary:
            summary = summary[0].upper() + summary[1:]

        # Normalize spaces
        summary = re.sub(r'\s+', ' ', summary).strip()

        # Check if the summary ends with a valid punctuation mark
        if summary and summary[-1] not in '.?!':
            # Split summary into sentences, keeping the punctuation
            sentences = re.split(r'(?<=[.?!])\s+', summary)
            
            # Reconstruct the summary without the last incomplete sentence
            if len(sentences) > 1:
                summary = ' '.join(sentences[:-1])  # Remove the last sentence if it's incomplete
            else:
                summary = sentences[0]  # If there are no complete sentences, keep it as is

            summary = summary.strip()  # Remove any trailing spaces

        # Ensure the summary ends with a period if it's complete and doesn't have punctuation
        if summary and summary[-1] not in '.?!':
            summary += '.'
        
        return summary


    def get_summarization_prompt(self, text):
        prompt = f"""Summarize: {text}

    Rules:
    1. Provide an exact summary of the input text.
    2. Do not add any information not present in the original text.
    3. If the input is very short (less than 10 words), repeat it verbatim without changes or additions.
  
    Summary:"""
        return prompt


    def predict(self, text, summary_length="short"):
        # Preprocess the text
        text = self.preprocess_text(text)

        # Generate the summarization prompt
        # text = self.get_summarization_prompt(text)
        
        # Load the locally saved T5 model and tokenizer
        model_path = 'artifacts/model_trainer/t5-large-model'
        tokenizer_path = 'artifacts/model_trainer/tokenizer'
        
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        gen_kwargs = {
            "length_penalty": 1.0,
            "num_beams": 4,
            "no_repeat_ngram_size": 2,  # Avoid repetition
        }

        if summary_length == "short":
            gen_kwargs["min_length"] = 20
            gen_kwargs["max_length"] = 50
        elif summary_length == "medium":
            gen_kwargs["min_length"] = 50
            gen_kwargs["max_length"] = 100
        elif summary_length == "long":
            gen_kwargs["min_length"] = 100
            gen_kwargs["max_length"] = 200

        # Use the locally loaded T5 model
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        print("length:::", len(text))

            # Calculate the length of the input text
        input_length = len(tokenizer.encode(text, return_tensors='pt')[0])

        # Compare input length with max_length
        if input_length < gen_kwargs["max_length"]:
            print(f"Your max_length is set to {gen_kwargs['max_length']}, but your input_length is only {input_length}. "
                f"Since this is a summarization task, where outputs shorter than the input are typically wanted, "
                f"you might consider decreasing max_length manually, e.g., summarizer('...', max_length={input_length}).")
            gen_kwargs["max_length"] = input_length  # Automatically adjust max_length to input_length

        # Check if the text length exceeds the minimum length required for summarization
        if len(text.split()) > gen_kwargs["min_length"]:
            output = pipe(text, **gen_kwargs)[0]["summary_text"]
        else:
            output = text  # If text is too short, return the original text
        print("\nModel Summary:")
        print(output)

        print("output length::",len(output))
        # Postprocess the summary
        output = self.postprocess_summary(output)

        return output
