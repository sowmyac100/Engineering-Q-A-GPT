import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Special tokens for padding/spacing (optional)
tokenizer.pad_token = tokenizer.eos_token

# Prepare your Q&A dataset (ensure you have question-answer pairs)
def create_dataset_from_qa(qa_pairs):
    """
    Convert question-answer pairs into a format suitable for fine-tuning GPT-2.
    """
    data = []
    for q, a in qa_pairs:
        data.append({"text": f"Q: {q} A: {a}"})  # Format as 'Q: question A: answer'

    return pd.DataFrame(data)

# Example: QA pairs generated in the previous step
qa_pairs = [
    ("What is the function of the propulsion system?", "The propulsion system pushes the spacecraft using ionized gas."),
    ("What is the role of the thermal control system?", "The thermal control system regulates the temperature of the spacecraft."),
    # Add more Q&A pairs here...
]

# Create dataset
df = create_dataset_from_qa(qa_pairs)

# Split dataset into train and validation sets (80% train, 20% validation)
train_df, val_df = train_test_split(df, test_size=0.2)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Create a DatasetDict to hold train and validation datasets
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# Tokenize the datasets (you can adjust max_length)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",           # Directory to save the model
    evaluation_strategy="epoch",             # Evaluate at the end of each epoch
    learning_rate=5e-5,                     # Learning rate (adjust as needed)
    per_device_train_batch_size=2,          # Batch size per device
    per_device_eval_batch_size=2,           # Batch size per evaluation
    num_train_epochs=3,                     # Number of epochs (adjust based on dataset size)
    weight_decay=0.01,                      # Regularization
    save_steps=10_000,                      # Save checkpoint every 10,000 steps
    save_total_limit=2,                     # Keep only the 2 most recent checkpoints
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                             # The model to train
    args=training_args,                      # Training arguments
    train_dataset=train_dataset,             # Training dataset
    eval_dataset=val_dataset,                # Validation dataset
    tokenizer=tokenizer,                     # Tokenizer
)

# Start training (fine-tuning)
trainer.train()

# Save the fine-tuned model
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
