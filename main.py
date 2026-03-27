#!/usr/bin/env python3
"""
agent_ln 项目主程序入口

使用说明：
1. 在 /hy-tmp 目录下运行：python agent_ln/main.py
2. 或者在 agent_ln 目录下运行：python -m rag_api.vector
"""
from src.utils.logger import logger
import os
import sys

# 添加当前目录到 Python 路径，确保可以导入 agent_ln 包
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logger.info(f"✅ 已添加路径到 sys.path: {current_dir}")

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("agent_ln 项目主程序")
    logger.info("=" * 60)
    
    try:
        # 导入必要的模块
        from rag_api.vector import Milvus_vector, prepare_pdf_document
        from models.model import SiliconFlowEmbeddings
        
        logger.info("✅ 所有模块导入成功")
        
        # 创建客户端
        client = SiliconFlowEmbeddings()
        logger.info("✅ SiliconFlowEmbeddings 客户端创建成功")
        
        # 准备文档
        logger.info("\n准备文档数据...")
        docs = prepare_pdf_document()
        logger.info(f"✅ 已加载 {len(docs)} 个文档")
        
