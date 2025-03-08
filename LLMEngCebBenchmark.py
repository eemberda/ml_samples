import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load the benchmark dataset using the datasets library
dataset = load_dataset("eemberda/english-ceb-bible-prompt")

# Convert the dataset to a Pandas DataFrame
data = dataset['train'].to_pandas()

# Select a subset of prompts and references for evaluation
prompts = data['prompt'].tolist()[:10]  # Adjust the number as needed
references = data['cebuano'].tolist()[:10]  # Ensure this column exists in your dataset

# Define the models to evaluate
models = {
    "Llama 3": "meta-llama/Meta-Llama-3-8B",
    "Mistral 7B": "mistralai/Mistral-7B-v0.1",
    "Deepseek R1": "DeepSeek-R1-Distill-Qwen-7B",
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name, model_path in tqdm(models.items()):
    print(f"Evaluating {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    tokenizer.to(device)
    model.eval()
    translations = []
    for prompt in tqdm(prompts):
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
