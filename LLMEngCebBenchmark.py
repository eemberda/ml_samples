import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# Load the benchmark dataset using the datasets library
dataset = load_dataset("eemberda/english-ceb-bible-prompt")

# Convert the dataset to a Pandas DataFrame
data = dataset['train'].to_pandas()

# Select a subset of prompts and references for evaluation
prompts = data['prompt'].tolist()[:10]  # Adjust the number as needed
references = data['reference'].tolist()[:10]  # Ensure this column exists in your dataset

# Define the models to evaluate
models = {
    "BLOOM": "bigscience/bloom",
    "Llama 3": "meta-llama/Llama-3-7b-hf",
    "Mistral 7B": "mistralai/Mistral-7B"
}

# Function to perform translation
def translate(tokenizer, model, text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=512)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Initialize BLEU metric
bleu = evaluate.load("bleu")

# Evaluate each model
results = []
for model_name, model_path in models.items():
    print(f"Evaluating {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    translations = []
    for prompt in prompts:
        translation = translate(tokenizer, model, prompt)
        translations.append(translation)
    # Compute BLEU score
    bleu_score = bleu.compute(predictions=translations, references=[[ref] for ref in references])
    results.append({"Model": model_name, "BLEU Score": bleu_score["bleu"]})

# Create a DataFrame for the results
results_df = pd.DataFrame(results)

# Display the results in a table
print("\nBLEU Scores for Each Model:")
print(results_df)

# Plot the BLEU scores
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='BLEU Score', data=results_df)
plt.title('BLEU Scores by Model')
plt.ylabel('BLEU Score')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.show()
