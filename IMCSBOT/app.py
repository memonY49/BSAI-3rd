from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch


app = FastAPI()

if torch.cuda.is_available():
    device = 0
else:
    device = -1

generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device=device
)

class prompt(BaseModel):
    text: str


@app.post("/generate")
def generate(prompt: prompt):
    user = prompt.text
    agent = generator(user, max_new_tokens=100)[0]["generated_text"]
    return {"agent":agent}