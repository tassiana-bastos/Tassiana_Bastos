"""
Parte 2 - Questão 2
Tradução inglês-francês com LangChain + Groq (API compatível com OpenAI).
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

app = FastAPI(title="TP2 - LangChain and Groq Translation")


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, description="Texto em inglês")


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str


def build_chain():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="A variável de ambiente GROQ_API_KEY não foi configurada."
        )

    prompt = PromptTemplate.from_template(
        "Translate the following text from English to French. "
        "Return only the translation:\n\n{text}"
    )
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    return prompt | llm | StrOutputParser()


@app.post("/translate", response_model=TranslationResponse)
def translate(data: TranslationRequest):
    chain = build_chain()
    try:
        translated = chain.invoke({"text": data.text})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Falha ao consultar o serviço de tradução: {exc}")

    return {"original_text": data.text, "translated_text": translated}


@app.get("/")
def home():
    return {"message": "LangChain + Groq English to French API"}
