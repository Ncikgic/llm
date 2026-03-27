from tqdm import tqdm
import json
import uuid
from src.utils.logger import logger
import time
import pandas as pd
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
import os
import sys
from pymilvus import FieldSchema, DataType
from src.models.model import SiliconFlowEmbeddings
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
class Milvus_vector():
    def __init__(self, client, uri=config.milvus_uri):
        self.URI = uri
        self.embeddings = client
        self.dense_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}}
        self.sparse_index = { 
             "metric_type": "BM25",
             "index_type": "SPARSE_INVERTED_INDEX"}
    def create_psyqa_vector_store(self, docs):
        init_docs = docs[:10]       
        self.vectorstore = Milvus.from_documents(
            documents=init_docs,
            embedding=self.embeddings,
            builtin_function=BM25BuiltInFunction(),
            index_params=[self.dense_index, self.sparse_index],
            #index_params=[self.dense_index],
            vector_field=["dense", "sparse"],
            #vector_field=["dense"],
            connection_args={
                "uri": self.URI,
            },
            collection_name = config.psyqa_collection_name,
            consistency_level=config.milvus_consistency_level,
            drop_old=True,  # 改为 True 清除旧数据
        )
        logger.info("已初始化创建 Milvus ‼")
        count = 10
        temp = []
        for doc in tqdm(docs[10:]):
            temp.append(doc)
            if len(temp) >= 5:
                self.vectorstore.add_documents(temp)
                count += len(temp)
                temp = []
                logger.info(f"已插⼊ {count} 条数据......")
                #time.sleep(1)
        logger.info(f"总共插⼊ {count} 条数据......")
        logger.info("✅已创建 Milvus 索引完成 ‼")
        return self.vectorstore

class Pdf_retriever():
    def __init__(self, client, uri=config.milvus_uri):
        self.URI = uri
        self.embeddings = client
        self.dense_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
        }
        self.sparse_index = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX"
        }
        self.docstore = InMemoryStore()
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def create_pdf_vector_store(self, docs):
        test_vec = self.embeddings.embed_query("test")
        actual_dim = len(test_vec)
        logger.info(f"实际维度: {actual_dim} (设置: 2560)")
        self.milvus_vectorstore = Milvus(
            embedding_function=self.embeddings,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            index_params=[
                {
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                },
                {
                    "metric_type": "BM25",
                    "index_type": "SPARSE_INVERTED_INDEX"
                }
            ],
            collection_name = config.pdf_collection_name,
            connection_args={"uri": self.URI},
            consistency_level=config.milvus_consistency_level,
            drop_old=True,  # 改为 True
        )
        logger.info("✅ Milvus连接成功")
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.milvus_vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )
        count = 0
        temp = []
        for doc in tqdm(docs):
            temp.append(doc)
            if len(temp) >= 10:
                self.retriever.add_documents(temp)
                count += len(temp)
                temp = []
                logger.info(f"已插⼊ {count} 条数据......")
                #time.sleep(1)
        logger.info(f"总共插⼊ {count} 条数据......")
        logger.info("✅基于PDF⽂档数据的 Milvus 索引完成 ‼")
        return self.retriever

def prepare_document(file_path):
    """
    准备文档数据，支持多种数据格式：
    1. psyqa_cleaned.jsonl 格式（包含 text/question/answer_text 字段）
    2. 原有的 query/response 格式
    """
    count = 0
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            content = json.loads(line.strip())
            if 'text' in content and content['text']:
                prompt = content['text']
            else:
                logger.info(f"警告: 第 {count+1} 行数据格式不支持，跳过")
                continue
            # 创建文档对象，包含更多元数据
            temp_doc = Document(
                page_content=prompt, 
                metadata={
                    "doc_id": str(uuid.uuid4()),
                    "question_id": content.get('questionID', ''),
                    "answer_index": content.get('answer_index', 0),
                    "keywords": content.get('keywords', ''),
                    "has_label": content.get('has_label', False),
                    "original_id": content.get('id', '')
                }
            )
            docs.append(temp_doc)
            count += 1
            
            if count % 100 == 0:
                logger.info(f"✅已加载 {count} 条数据!")
    
    logger.info(f"总共加载了 {count} 条数据")
    return docs

def prepare_pdf_document(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(subset=['text_content'])
    documents = []
    for _, row in df.iterrows():
        text_content = str(row['text_content']) if pd.notna(row['text_content']) else ""
        doc = Document(
            page_content=text_content.strip(),
            metadata={"doc_id": str(uuid.uuid4())}
        )
        documents.append(doc)
    logger.info(f"成功加载 {len(documents)} 个⽂档")
    return documents

def generate_milvus_vectorstore(collection_name,
                                uri=config.milvus_uri,
                                client=None):
    """
    创建Milvus向量存储
    Args:
        collection_name: 集合名称
        uri: Milvus数据库URI
        client: 嵌入客户端，如果为None则创建新的SiliconFlowEmbeddings实例
    """

    # 如果client为None，创建新的嵌入客户端
    if client is None:
        client = SiliconFlowEmbeddings()
    
    try:
        # 首先检查集合是否存在，以及是否有sparse字段
        from pymilvus import connections, utility
        connections.connect(uri=uri)
        collections = utility.list_collections()
        
        if collection_name in collections:
            # 检查集合是否有sparse字段
            from pymilvus import Collection
            collection = Collection(collection_name)
            field_names = [field.name for field in collection.schema.fields]
            
            if "sparse" not in field_names:
                # 如果集合没有sparse字段，删除它并重新创建
                logger.info(f"集合 '{collection_name}' 缺少sparse字段，删除并重新创建...")
                utility.drop_collection(collection_name)
        
        # 创建或重新连接集合（确保有sparse字段）
        milvus_vectorstore = Milvus(
            embedding_function=client,
            builtin_function=BM25BuiltInFunction(),
            collection_name=collection_name,
            connection_args={"uri": uri},
            vector_field=["dense", "sparse"],
            index_params=[
                {
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                },
                {
                    "metric_type": "BM25",
                    "index_type": "SPARSE_INVERTED_INDEX"
                }
            ],
            consistency_level=config.milvus_consistency_level,
            drop_old=False,  # 我们已经手动处理了删除
        )
        logger.info(f"连接 Milvus 集合 '{collection_name}' 成功......")
        return milvus_vectorstore
    except Exception as e:
        logger.info(f"连接 Milvus 失败: {e}")
        # import traceback
        # traceback.logger.info_exc()
        raise
def extract_xlsx_files(file_path: str) -> list:
    """
    提取指定路径下所有.xlsx后缀的文件路径
    """
    path = Path(file_path)
    xlsx_files = []
    if path.is_file() and path.suffix.lower() == '.xlsx':
        # 如果是单个.xlsx文件，直接返回
        xlsx_files.append(str(path.absolute()))
    elif path.is_dir():
        # 如果是目录，递归查找所有.xlsx文件
        for file in path.rglob("*.xlsx"):
            xlsx_files.append(str(file.absolute()))
    
    return xlsx_files
if __name__ == "__main__":
    # 创建客户端实例
    client = SiliconFlowEmbeddings()
    
    # 获取所有.xlsx文件
    path_list = extract_xlsx_files(config.pdf_output)
    logger.info(f"找到 {len(path_list)} 个.xlsx文件: {path_list}")
    
    # 只处理包含text_content列的文件
    docs = []
    for path in path_list:
        try:
            # 尝试读取文件并检查是否有text_content列
            import pandas as pd
            df = pd.read_excel(path)
            if 'text_content' in df.columns:
                logger.info(f"处理文件: {path}")
                file_docs = prepare_pdf_document(path)
                docs.extend(file_docs)
                logger.info(f"  成功加载 {len(file_docs)} 个文档")
            else:
                logger.info(f"跳过文件 (无text_content列): {path}")
        except Exception as e:
            logger.info(f"处理文件 {path} 时出错: {e}")
    
    if not docs:
        logger.info("错误: 没有找到包含text_content列的Excel文件!")
        exit(1)
    
    logger.info(f"预处理 PDF ⽂档数据成功，总共 {len(docs)} 个文档......")
    
    pdf_vectorstore = Pdf_retriever(client)
    logger.info("创建 PDF Milvus 连接成功......")
    retriever = pdf_vectorstore.create_pdf_vector_store(docs)
    logger.info("创建基于 Milvus 数据库的⽗⼦⽂档检索器成功......")
    
    milvus = Milvus_vector(client)
    file = config.psyqa_data_path
    vector_docs = prepare_document(file)
    vectorstore = milvus.create_psyqa_vector_store(vector_docs)
    logger.info("全部初始化完成, 可以开始问答了......")
