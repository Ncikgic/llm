import json
import os
import sys
from src.utils.logger import logger
from pymilvus import Collection, connections
import random
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.models.model import generate_deepseek_answer, create_deepseek_client
def get_random_documents_from_milvus(collection_name=config.psyqa_collection_name, 
                                     num_samples=None, 
                                     uri=config.milvus_uri):
    """
    从 Milvus 中随机获取文档
    Args:
        collection_name: collection名称
        num_samples: 要获取的随机文档数量
        uri: 数据库路径
    Returns:
        List[Document]: 随机文档列表
    """
    # 1. 连接 Milvus
    connections.connect(uri=uri,keep_alive_time=30000,keep_alive_timeout=10000)
    # 2. 加载 collection
    collection = Collection(collection_name)
    collection.load()
    # 3. 获取总数据量
    total_count = collection.num_entities
    logger.info(f"📊 总数据量: {total_count} 条")
    if total_count == 0:
        logger.info("⚠️ collection 为空")
        return []
    # 4. 获取所有主键
    # 限制请求数量不超过总数据量
    num_samples = min(num_samples, total_count)
    # 获取所有主键（可能需要分页，但数据量不大）
    # 注意：对于大数据量，这可能需要优化
    all_pks = collection.query(
        expr="pk > 0",
        output_fields=["pk"],
        limit=total_count
    )
    if not all_pks:
        logger.info("⚠️ 无法获取主键列表")
        return []
    # 提取主键值
    pk_values = [item["pk"] for item in all_pks]
    # 随机选择主键
    random_pks = random.sample(pk_values, num_samples)
    # 5. 获取数据
    # 构建查询表达式：pk in [id1, id2, ...]
    id_list_str = ", ".join(str(pk) for pk in random_pks)
    expr = f"pk in [{id_list_str}]"
    results = collection.query(
        expr=expr,
        output_fields=["pk", "text", "doc_id", "question_id", "answer_index", "keywords", "has_label", "original_id"]  # 指定要获取的字段
    )
    # 6. 转换为 Document 格式
    from langchain_core.documents import Document
    documents = []
    for result in results:
        doc = Document(
            page_content=result.get("text", ""),
            metadata={
                "pk": result.get("pk"),
                "doc_id": result.get("doc_id", ""),
                "question_id": result.get("question_id", ""),
                "answer_index": result.get("answer_index", 0),
                "keywords": result.get("keywords", ""),
                "has_label": result.get("has_label", False),
                "original_id": result.get("original_id", "")
            }
        )
        documents.append(doc)
    
    logger.info(f"✅ 成功获取 {len(documents)} 条随机数据")
    return documents
def build_prompt(documents, num_cases):
    """
    根据文档列表构建prompt
    
    Args:
        documents: List[Document] - Document对象列表
        num_cases: int - 需要生成的测试用例数量
    
    Returns:
        str: 构建的prompt字符串
    """
    # 提取所有文档的内容和实际ID
    context_texts = []
    doc_info = []  # 存储文档信息和对应的实际ID
    
    for i, doc in enumerate(documents):
        # 使用文档的page_content作为文本内容
        content = doc.page_content
        # 获取文档的实际ID（pk）
        actual_id = str(doc.metadata.get("pk", ""))
        if not actual_id:
            # 如果没有pk，使用original_id
            actual_id = str(doc.metadata.get("original_id", f"chunk_{i+1:03d}"))
        
        # 存储文档信息
        doc_info.append({
            "index": i + 1,
            "id": actual_id,
            "content": content[:500]  # 只取前500字符作为上下文
        })
        
        # 构建上下文文本，包含实际ID
        context_texts.append(f"=== 文档 {i+1} (ID: {actual_id}) ===\n{content[:500]}...\n")
    
    # 将所有上下文文本连接起来
    context_text = "\n".join(context_texts)
    
    # 构建文档ID列表，供LLM参考
    id_list = ", ".join([f'"{info["id"]}"' for info in doc_info])
    
    prompt = f"""
你是一个心理学知识库专家，需要根据数据库的内容为心理疗愈AI助手生成召回测试数据。

数据库内容如下（每个文档都有唯一的ID）：
{context_text}

每个测试用例必须包含：
- query: 基于给定文档内容生成的问题
- answer: 简洁准确的答案（基于文档内容）
- ID: 必须使用文档的实际ID（来自上面的文档ID）

重要要求：
1. query必须基于给定文档的文本内容生成
2. answer必须简洁准确，基于文档内容
3. ID必须使用文档的实际ID（不能虚构ID）
4. 生成{num_cases}条测试用例，均匀分布在不同的文档上
5. 可用的文档ID有：{id_list}

输出格式必须是严格的JSON数组：
[
{{
 "query": "基于文档内容的问题",
 "answer": "简洁的答案",
 "ID": "文档的实际ID"
}}
]

请生成测试数据，确保每个测试用例的ID都来自上面的文档ID列表。
"""
    return prompt
if __name__ == "__main__":
    collections_list = config.collections_list
    # 读取现有的recalltest.json文件
    output_file = "recalltest.json"
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            recalltest_data = json.load(f)
    except FileNotFoundError:
        # 如果文件不存在，创建初始结构
        recalltest_data = {}
        for collection_name in collections_list:
            recalltest_data[collection_name] = []
    except json.JSONDecodeError:
        # 如果JSON解析错误，创建新的结构
        recalltest_data = {}
        for collection_name in collections_list:
            recalltest_data[collection_name] = []
    
    # 获取Milvus中实际存在的集合
    from pymilvus import connections, utility
    connections.connect(uri=config.milvus_uri)
    
    # 尝试列出所有集合
    try:
        existing_collections = utility.list_collections()
        logger.info(f"📊 Milvus中存在的集合: {existing_collections}")
    except Exception as e:
        logger.info(f"⚠️ 无法获取集合列表: {e}")
        existing_collections = []
    
    # 只处理实际存在的集合
    for collection_name in collections_list:
        if collection_name:
            # 尝试检查集合是否存在
            try:
                # 尝试创建Collection对象，如果集合不存在会抛出异常
                collection = Collection(collection_name)
                # 如果上面的代码没有抛出异常，说明集合存在
                collection_exists = True
            except Exception:
                collection_exists = False
            
            if collection_exists:
                num_cases = 30
                # 使用正确的数据库路径
                texts = get_random_documents_from_milvus(
                    collection_name=collection_name,
                    num_samples=30,
                    uri=config.milvus_uri)
                if not texts:
                    logger.info(f"❌ 无法从集合 {collection_name} 获取文档，请检查数据库连接")
                    continue      
                logger.info(f"✅ 成功从集合 {collection_name} 获取 {len(texts)} 个文档")
                prompt = build_prompt(texts, num_cases)
                logger.info("📝 构建prompt完成")
                client = create_deepseek_client()
                logger.info("🤖 调用deepseek API生成测试数据...")
                try:
                    # 打印prompt的前500个字符用于调试
                    logger.info(f"📋 Prompt预览: {prompt[:500]}...")
                    
                    recalltest_json = generate_deepseek_answer(client, prompt)
                    
                    # 打印API响应用于调试
                    logger.info(f"📥 API响应预览: {recalltest_json[:500] if recalltest_json else '空响应'}...")
                    
                    # 检查响应是否为空
                    if not recalltest_json or recalltest_json.strip() == "":
                        logger.info(f"⚠️ API返回空响应，跳过集合 {collection_name}")
                        continue
                    
                    # 去除可能的Markdown代码块标记
                    cleaned_json = recalltest_json.strip()
                    if cleaned_json.startswith("```json"):
                        cleaned_json = cleaned_json[7:]  # 移除 ```json
                    if cleaned_json.startswith("```"):
                        cleaned_json = cleaned_json[3:]  # 移除 ```
                    if cleaned_json.endswith("```"):
                        cleaned_json = cleaned_json[:-3]  # 移除结尾的 ```
                    cleaned_json = cleaned_json.strip()
                    
                    # 尝试解析JSON
                    recalltest = json.loads(cleaned_json)
                    
                    # 验证数据结构
                    validated = []
                    if isinstance(recalltest, list):
                        for item in recalltest:
                            if (
                                isinstance(item, dict)
                                and "query" in item
                                and "answer" in item
                                and "ID" in item):
                                validated.append(item)
                    elif isinstance(recalltest, dict):
                        # 如果返回的是单个对象而不是数组
                        if ("query" in recalltest and "answer" in recalltest and "ID" in recalltest):
                            validated.append(recalltest)
                    
                    # 将验证后的数据写入recalltest_data中对应的键
                    recalltest_data[collection_name] = validated
                    logger.info(f"✅ 为集合 {collection_name} 生成完成: {len(validated)} 条有效测试用例")
                    
                except json.JSONDecodeError as e:
                    logger.info(f"❌ JSON解析错误: {e}")
                    logger.info(f"📝 原始响应: {recalltest_json[:1000] if recalltest_json else '空响应'}")
                    logger.info(f"⚠️ 为集合 {collection_name} 生成失败，跳过")
                except Exception as e:
                    logger.info(f"❌ 为集合 {collection_name} 生成过程中出现错误: {e}")
                    import traceback
                    traceback.logger.info_exc()
            else:
                logger.info(f"⚠️ 集合 {collection_name} 在Milvus中不存在，跳过")
    
    # 将所有数据写入recalltest.json文件
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(recalltest_data, f, ensure_ascii=False, indent=2)
        logger.info(f"📁 所有结果已保存到: {output_file}")
        logger.info(f"📊 总计: {sum(len(data) for data in recalltest_data.values())} 条测试用例")
    except Exception as e:
        logger.info(f"❌ 保存文件时出现错误: {e}")
