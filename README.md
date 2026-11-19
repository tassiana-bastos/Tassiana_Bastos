# TP2 – Desenvolvimento de Data-Driven Apps com Python

Este repositório contém as cinco aplicações solicitadas no TP2. Cada arquivo representa uma aplicação FastAPI independente.

## Estrutura
- `p1q1.py`: FastAPI + Hugging Face GPT-2 para geração de texto.
- `p1q2.py`: FastAPI + Helsinki-NLP/opus-mt-en-fr para tradução inglês-francês.
- `p2q1.py`: FastAPI + LangChain FakeListLLM para chatbot simulado.
- `p2q2.py`: FastAPI + LangChain + Groq para tradução inglês-francês.
- `p2q3.py`: FastAPI + LangChain + Hugging Face Helsinki-NLP/opus-mt-en-de para tradução inglês-alemão.
- `requirements.txt`: dependências comuns às cinco aplicações.

## Ambiente no Windows / VS Code / IDE escolhida
Abra esta pasta na sua IDE e execute os comandos abaixo:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Na primeira execução, os modelos Hugging Face serão baixados. É necessária conexão com a internet e espaço em disco. A inferência pode ser lenta em CPU.

## Como executar
Execute apenas uma aplicação por vez, em seu próprio terminal:

```powershell
uvicorn p1q1:app --reload --port 8001
uvicorn p1q2:app --reload --port 8002
uvicorn p2q1:app --reload --port 8003
uvicorn p2q2:app --reload --port 8004
uvicorn p2q3:app --reload --port 8005
```

Para cada aplicação, acesse `http://127.0.0.1:PORT/docs`, substituindo PORT pela porta usada (8001 a 8005).

## Exemplos de requisição
POST `/generate` em p1q1:
```json
{"text": "Artificial intelligence is"}
```

POST `/translate` em p1q2, p2q2 ou p2q3:
```json
{"text": "Data science helps organizations make better decisions."}
```

POST `/chat` em p2q1:
```json
{"message": "O que é Python?"}
```

## Chave Groq (p2q2)
Crie uma chave na plataforma Groq e configure-a no PowerShell antes de iniciar o servidor:

```powershell
$env:GROQ_API_KEY="SUA_CHAVE"
uvicorn p2q2:app --reload --port 8004
```

## Observações
- Os modelos são carregados sob demanda e reutilizados durante a execução do processo.
- GPT-2 é um modelo de continuação de texto, não um tradutor nem um chatbot factual.
- FakeListLLM devolve respostas predefinidas e serve para testar o fluxo sem custo de inferência ou API externa.
- Os endpoints podem ser testados pelo Swagger em `/docs`.
