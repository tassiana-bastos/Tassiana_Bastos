
"""
Parte 1 - Questão 2

API FastAPI para tradução de inglês para francês
utilizando Hugging Face Transformers e MarianMT.

Modelo: Helsinki-NLP/opus-mt-en-fr
"""

from functools import lru_cache

import torch

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# CONFIGURAÇÃO DA API
app = FastAPI(
    title="TP2 - English to French Translation",
    description="API de tradução utilizando Hugging Face",
    version="1.0.0"
)


# MODELO DE ENTRADA
class TranslationRequest(BaseModel):
    text: str = Field(
        min_length=1,
        description="Texto em inglês"
    )


# MODELO DE RESPOSTA
class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str


# CARREGAMENTO DO MODELO
@lru_cache
def load_model():

    model_name = "Helsinki-NLP/opus-mt-en-fr"

    device = "cpu"

    # Carrega o tokenizador e o modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# ROTA DE TRADUÇÃO
@app.post(
    "/translate",
    response_model=TranslationResponse
)
def translate(data: TranslationRequest):

    # Carrega o modelo
    tokenizer, model, device = load_model()

    # Tokenizacao do texto de entrada
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True)

    # Envua os dados para o dispositivo utilizado
    inputs= inputs.to(device)

    # Gera a tradução
    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            max_new_tokens=128
        )

    # Decodifica os tokens traduzidos
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Retorna os resultados
    return {
        "original_text": data.text,
        "translated_text": translated_text
    }


# ROTA PRINCIPAL
@app.get("/")
def home():

    return {
        "message": "English to French Translation API"
    }
