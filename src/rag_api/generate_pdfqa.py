import os
import json
import uuid
import pandas as pd
from src.utils.logger import logger
from tqdm import tqdm
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever, MultiVectorRetriever
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from src.models.model import SiliconFlowEmbeddings
from src.rag_api.vector import generate_milvus_vectorstore, prepare_pdf_document
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

import config
##################################创建qa对###################################
# 创建 LangChain 兼容的模型
model = ChatOpenAI(
    api_key=config.deepseek_api_key,
    base_url=config.deepseek_base_url,
    model=config.deepseek_model,
    temperature=0.7,
    max_tokens=2000  # 添加 max_tokens 设置
)
embeddings = SiliconFlowEmbeddings()
question_gen_prompt_str = (
    "你是⼀位AI心理咨询专家。请根据以下⽂档内容,⽣成3个⽤户可能会提出的,⾼度相关的问题。\n"
    "只返回问题列表，每个问题占⼀⾏，不要有其他前缀或编号。\n"
    "⽂档内容:\n"
    "----------\n"
    "{content}\n"
    "----------\n"
)

question_gen_prompt = ChatPromptTemplate.from_template(question_gen_prompt_str)
question_generator_chain = question_gen_prompt | model | StrOutputParser()
# 加载PDF文档数据
logger.info("加载PDF文档数据...")
pdf_file_path = '/hy-tmp/agent_ln/pdf_output/pdf_detailed_text.xlsx'
docs = prepare_pdf_document(pdf_file_path)
logger.info(f"成功加载 {len(docs)} 个文档,{len(docs[:10])}")
# 生成问题
logger.info("生成相关问题...")
sub_docs = []
for i, doc in enumerate(docs):
    doc_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
    try:
        generated_questions = question_generator_chain.invoke({"content": doc.page_content}).split("\n")
        generated_questions = [q.strip() for q in generated_questions if q.strip()]
        for q in generated_questions:
            sub_docs.append(Document( metadata={"doc_id": doc_id, "source": "generated_question"}))
    except Exception as e:
        logger.info(f"为文档 {i} 生成问题时出错: {e}")
        continue

logger.info(f"生成了 {len(sub_docs)} 个相关问题")
########################################加入collection###########################
logger.info("创建milvus向量数据库, 并添加文档...")
milvus = generate_milvus_vectorstore('pdf_collection', client=embeddings)



# 创建检索器
vectorstore = Milvus.from_documents(
            documents=sub_docs,
            embedding=embeddings,
            builtin_function=BM25BuiltInFunction(),
            index_params=[
                {
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                },
                {
                    "metric_type": "BM25",
                    "index_type": "SPARSE_INVERTED_INDEX"
                }            ],
            #index_params=[self.dense_index],
            vector_field=["dense", "sparse"],
            #vector_field=["dense"],
            connection_args={ "uri":     },
            collection_name = config.psyqa_collection_name,
            consistency_level=config.milvus_consistency_level,
            drop_old=True,  # 改为 True 清除旧数据
        )
# 添加文档到检索器
count = 0
temp = []
for doc in tqdm(docs):
    temp.append(doc)
    if len(temp) >= 10:
        retriever.add_documents(temp)
        count += len(temp)
        temp = []
        logger.info(f"已插入 {count} 条数据......")
        # time.sleep(1)

if temp:  # 处理剩余的文档
    retriever.add_documents(temp)
    count += len(temp)

logger.info(f"总共插入 {count} 条数据......")
logger.info("✅基于PDF文档数据的 Milvus 索引完成 ‼")
