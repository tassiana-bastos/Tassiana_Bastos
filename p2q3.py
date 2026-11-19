
"""
Parte 2 - Questão 3

API FastAPI para tradução de inglês para alemão
utilizando LangChain e Hugging Face.

Modelo: Helsinki-NLP/opus-mt-en-de
"""

from functools import lru_cache

import torch

from fastapi import FastAPI
from pydantic import BaseModel, Field

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# CONFIGURAÇÃO DA API

app = FastAPI(
    title="TP2 - English to German Translation",
    description="API de tradução utilizando LangChain e Hugging Face",
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


# CARREGAMENTO DO MODELO HUGGING FACE

@lru_cache
def load_model():

    model_name = "Helsinki-NLP/opus-mt-en-de"

    device = torch.device("cpu")
    

    # Carrega o tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # Carrega o modelo de tradução
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    model.to(device)

    model.eval()

    return tokenizer, model, device


# FUNÇÃO DE TRADUÇÃO

def generate_translation(prompt_value):

    # Converte o objeto recebido do LangChain em string
    text= prompt_value.to_string()

    # Recupera o modelo e o tokenizer

    tokenizer, model, device = load_model()

    # Tokenização do texto
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Envia os dados para CPU ou GPU
    inputs = inputs.to(device)

    # Geração da tradução
    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            max_new_tokens=128
        )

    # Converte os tokens em texto
    translated_text = tokenizer.decode(
        translated_tokens[0],
        skip_special_tokens=True
    )

    return translated_text


# CONFIGURAÇÃO DO LANGCHAIN

@lru_cache
def build_chain():

    # Criação do prompt
    prompt = PromptTemplate.from_template(
        "{text}"
    )

    # Integração do modelo Hugging Face com LangChain
    translation_model = RunnableLambda(
        generate_translation
    )

    # Construção da chain
    chain = (
        prompt
        | translation_model
        | StrOutputParser()
    )

    return chain


# ROTA DE TRADUÇÃO

@app.post(
    "/translate",
    response_model=TranslationResponse
)
def translate(data: TranslationRequest):

    # Recupera a chain
    chain = build_chain()

    # Executa a tradução
    translated_text = chain.invoke(
        {"text": data.text}
    )

    # Retorna o resultado
    return {
        "original_text": data.text,
        "translated_text": translated_text
    }


# ROTA PRINCIPAL

@app.get("/")
def home():

    return {
        "message": "English to German Translation API"
    }
