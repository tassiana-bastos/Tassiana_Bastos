"""
Parte 1 - Questão 1
API FastAPI para continuação de texto com GPT-2 (Hugging Face).
"""
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="TP2 - GPT-2 Text Generation")


class TextRequest(BaseModel):
    text: str = Field(min_length=1, description="Frase inicial em inglês")


class TextResponse(BaseModel):
    input_text: str
    generated_text: str


@lru_cache
def load_model():
    # O modelo é carregado uma vez e reutilizado nas requisições.
    return pipeline("text-generation", model="openai-community/gpt2")


@app.post("/generate", response_model=TextResponse)
def generate_text(data: TextRequest):
    generator = load_model()
    result = generator(
        data.text,
        max_new_tokens=60,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    generated = result[0]["generated_text"]
    return {"input_text": data.text, "generated_text": generated}


@app.get("/")
def home():
    return {"message": "GPT-2 text generation API"}
