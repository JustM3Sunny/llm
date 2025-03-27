from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

app = FastAPI()

# Load Model & Tokenizer
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# FastAPI Route
@app.get("/")
def read_root():
    return {"message": "LLM API with GUI is running!"}

@app.post("/generate/")
def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# Gradio Chatbot Function
def chat_with_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio UI
chatbot_ui = gr.Interface(
    fn=chat_with_llm,
    inputs="text",
    outputs="text",
    title="🚀 LLM Chatbot",
    description="Type your question and get AI-powered responses!",
)

# Run Gradio App
@app.get("/chat/")
def start_chat():
    chatbot_ui.launch(server_name="0.0.0.0", server_port=7860, share=True)
    return {"message": "Chatbot started at /chat/"}
