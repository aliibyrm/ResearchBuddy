from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

# FastAPI uygulamasını başlat
app = FastAPI()

# URL'ler ve başlangıç ayarları
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Belgeleri indir ve işle
def process_documents(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    flat_docs = [doc for sublist in docs for doc in sublist]  # Düz liste

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(flat_docs)
    return split_docs

# Vektör deposunu oluştur
def create_vectorstore(split_docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="articles",
        persist_directory="./.chroma"
    )
    return vectorstore

# Makaleleri işleyin ve vektör deposu oluşturun
documents = process_documents(urls)
vectorstore = create_vectorstore(documents)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# GPT-4 LLM'i başlat
llm = ChatOpenAI(model="gpt-4", temperature=0)

# İstek modeli
class QuestionRequest(BaseModel):
    question: str

# Yanıt modeli
class AnswerResponse(BaseModel):
    answer: str

# Soruyu işleyen yardımcı fonksiyon
# Soruyu işleyen yardımcı fonksiyon
def answer_question(question: str, retriever, llm):
    documents = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in documents])
    ai_response = llm.invoke(f"Context: {context}\n\nQuestion: {question}")
    return ai_response.content  # Burada yanıtın sadece metin kısmını döndürüyoruz


# Endpoint tanımı
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        answer = answer_question(request.question, retriever, llm)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
