import torch
import tkinter as tk
import tiktoken
from tkinter import messagebox
from importlib.metadata import version
from previous_labs import GPTModel
from pathlib import Path
from previous_labs import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

# Load fine-tuned model
finetuned_model_path = Path("gpt2-medium355MProjet3.pth")
if not finetuned_model_path.exists():
    print(f"Could not find '{finetuned_model_path}'.")

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load(finetuned_model_path, map_location=torch.device("cpu")))
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()

def custom_question(question, A, B, C="", D=""):
    instruction_text = (
        f"Below is an instruction that describes a question with 4 possible answers. "
        f"Choose a letter that appropriately answers the question."
        f"\n\n### Instruction:\n{question}"
    )
    
    input_text = "\n\n### Input:\n"
    if A: input_text += f"A. {A}\n"
    if B: input_text += f"B. {B}\n"
    if C: input_text += f"C. {C}\n"
    if D: input_text += f"D. {D}\n"

    return instruction_text + input_text

def predict():
    question = question_entry.get()

    A = entryA.get()
    B = entryB.get()
    C = entryC.get()
    D = entryD.get()

    if not A or not B:
        messagebox.showerror("Input Error", "Please fill at least A and B.")
        return
    
    input_question = custom_question(question, A, B, C, D)

    print(input_question)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_question, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    response = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(response, input_question)
    
    print(response)
    result_label.config(text=f"Predicted Answer: {response}")

# GUI Setup
root = tk.Tk()
root.title("LLM Question Classifier")

# Create input fields
tk.Label(root, text="Question:").pack()
question_entry = tk.Entry(root, width=50)
question_entry.pack()

tk.Label(root, text=f"Answer A:").pack()
entryA = tk.Entry(root, width=50)
entryA.pack()

tk.Label(root, text=f"Answer B:").pack()
entryB = tk.Entry(root, width=50)
entryB.pack()

tk.Label(root, text=f"Answer C:").pack()
entryC = tk.Entry(root, width=50)
entryC.pack()

tk.Label(root, text=f"Answer D:").pack()
entryD = tk.Entry(root, width=50)
entryD.pack()

# Button to run prediction
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

# Label to display result
result_label = tk.Label(root, text="Predicted Answer: ", font=("Arial", 14, "bold"))
result_label.pack()

# Run the application
root.mainloop()
