#!/usr/bin/env python3
"""
测试Milvus连接是否正常
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_api.vector import generate_milvus_vectorstore
import config
from src.utils.logger import logger

def test_milvus_connection():
    """测试Milvus连接"""
    try:
        logger.info("开始测试Milvus连接...")
        
        # 测试psyqa_collection
        logger.info(f"测试集合: {config.psyqa_collection_name}")
        vectorstore = generate_milvus_vectorstore(config.psyqa_collection_name)
        
        # 测试简单的相似性搜索
        test_query = "心理健康"
        logger.info(f"执行测试查询: '{test_query}'")
        results = vectorstore.similarity_search(test_query, k=3)
        
