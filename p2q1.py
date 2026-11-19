"""
Parte 2 - Questão 1
Chatbot simulado com LangChain FakeListLLM e FastAPI.
Executar: uvicorn p2q1:app --reload --port 8003
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_core.language_models.fake import FakeListLLM

app = FastAPI(title="TP2 - Fake LLM Chatbot")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    question: str
    answer: str


# Respostas fixas para simular um modelo sem chamadas externas.
fake_llm = FakeListLLM(
    responses=[
        "Olá! Sou um chatbot de demonstração.",
        "Python é uma linguagem de programação muito utilizada em Ciência de Dados.",
        "FastAPI permite criar APIs web com Python."
    ]
)


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    answer = fake_llm.invoke(data.message)
    return {"question": data.message, "answer": answer}


@app.get("/")
def home():
    return {"message": "Fake LLM chatbot API"}
