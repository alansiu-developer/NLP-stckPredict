from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# Add to ChromaDB vector store
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()


question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
len(docs)

docs[0]


!curl https://ollama.ai/install.sh | sh
!sudo apt install -y neofetch

!neofetch

import subprocess
import time

# Start ollama as a backrgound process
command = "nohup ollama serve&"

# Use subprocess.Popen to start the process in the background
process = subprocess.Popen(command,
                            shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
print("Process ID:", process.pid)
# Let's use fly.io resources
#!OLLAMA_HOST=https://ollama-demo.fly.dev:443
time.sleep(5)  # Makes Python wait for 5 seconds

!ollama pull gemma:7b

print("done done done")
