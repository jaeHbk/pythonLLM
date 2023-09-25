from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from starlette.responses import FileResponse
from serv.ai import generate_review

loader = WebBaseLoader("https://www.ukc.ksea.org/plenary-keynotes/plenary/")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

app = FastAPI()

print("test")
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


@app.get("/api/gpt")
async def gpt(review: str):
    print(qa)
    return generate_review(review, qa)



#while True:
#    query = input("Ask a question about tea\n")
#    print(qa.run(query))