from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

# Makale linklerini ekleyin
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Belgeleri indir ve işle
def process_documents(urls):
    print("Makaleler indiriliyor ve işleniyor...")
    docs = [WebBaseLoader(url).load() for url in urls]
    flat_docs = [doc for sublist in docs for doc in sublist]  # Düz liste
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(flat_docs)
    return split_docs

# Vektör deposunu oluştur
def create_vectorstore(split_docs):
    print("Vektör deposu oluşturuluyor...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="articles",
        persist_directory="./.chroma"
    )
    return vectorstore

# Soruya yanıt üret
def answer_question(question, retriever, llm):
    print("Soru işleniyor...")
    documents = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in documents])
    response = llm.invoke(f"Context: {context}\n\nQuestion: {question}")
    return response


if __name__ == "__main__":
    # Makaleleri işleyin ve vektör deposu oluşturun
    documents = process_documents(urls)
    vectorstore = create_vectorstore(documents)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # GPT-4 LLM'i başlat
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Kullanıcıdan gelen soruları yanıtla
    while True:
        question = input("Sorunuzu girin (çıkmak için 'exit' yazın): ")
        if question.lower() == "exit":
            print("Çıkış yapılıyor...")
            break
        
        answer = answer_question(question, retriever, llm)
        print("\nYanıt:")
        print(answer)
