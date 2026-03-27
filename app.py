import os
import uvicorn
from src.utils.logger import logger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import datetime
from langchain_milvus import Milvus, BM25BuiltInFunction
from src.models.model import create_qwen_client, generate_qwen_answer, create_deepseek_client, generate_deepseek_answer, SiliconFlowEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from src.rag_api.vector import generate_milvus_vectorstore
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)
embedding_model = SiliconFlowEmbeddings()
logger.info("创建 Embedding 模型成功......")


psyqa_vectorstore=generate_milvus_vectorstore(config.psyqa_collection_name)
#docstore = InMemoryStore()
#pdf_vectorstore=generate_milvus_vectorstore("pdf_collection")
model,tokenizer=create_qwen_client()
logger.info("创建 model 成功......")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
@app.post("/")
async def chatbot(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('question')
    recall_rerank_milvus = psyqa_vectorstore.similarity_search(
        query,
        k=10,
        ranker_type="rrf",
        ranker_params={"k": 100}    )
    if recall_rerank_milvus:
        context = format_docs(recall_rerank_milvus)
    else:
        context = [] 
    # res = ""
    # retrieved_docs = parent_retriever.invoke(query)
    # if retrieved_docs is not None and len(retrieved_docs) >= 1:
    #     res = retrieved_docs[0].page_content
    #     context = context + "\n" + res
    SYSTEM_PROMPT = """
    System: 你是⼀个⾮常得⼒的医学助⼿, 你可以通过从数据库中检索出的信息找到问题的答案.
    """
    USER_PROMPT = f"""
    User: 利⽤介于<context>和</context>之间的从数据库中检索出的信息来回答问题, 具体的问题介于<question>和</question>之间. 如果提供的信息为空, 则按照你的经验知识来给出尽可能严谨准确的回答, 不知道的时候坦诚的承认不了解, 不要编造不真实的信息.
    <context>
    {context}
    </context>
    <question>
    {query}
    </question>
    """
    response=generate_qwen_answer(model,tokenizer, SYSTEM_PROMPT + USER_PROMPT.format(context, query))
    # client=create_deepseek_client()
    # response=generate_deepseek_answer(client,SYSTEM_PROMPT + USER_PROMPT.format(context, query))
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time}
    return answer

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8103, workers=1)
