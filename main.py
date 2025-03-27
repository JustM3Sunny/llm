from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Model & Tokenizer Load
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Change to "TheBloke/Vicuna-7B-1.1" for Vicuna
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

@app.get("/")
def read_root():
    return {"message": "LLM API is running!"}

@app.post("/generate/")
def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
