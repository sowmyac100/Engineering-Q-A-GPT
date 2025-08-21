from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

# Function to generate answers based on input questions
def generate_answer(question):
    input_text = f"Q: {question} A:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("A:")[1].strip()

# Example question
question = "What is the function of the propulsion system?"
answer = generate_answer(question)
print(f"Answer: {answer}")
