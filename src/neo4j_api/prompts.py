# prompts.py

import os
from schemas import EXAMPLE_SCHEMA

def create_system_prompt(schema: str) -> str:
    return f"""
你是一个专业的Neo4j Cypher查询生成器, 你的任务是将自然语言描述转换为准确, 高效的Cypher查询.

# 图数据库模式
{schema}

# 重要规则
1. 始终使用参数化查询风格, 对字符串值使用单引号
2. 确保节点标签和关系类型使用正确的大小写
3. 对于模糊查询, 使用 CONTAINS 或 STARTS WITH 而不是 "="
4. 对于可选模式, 使用 OPTIONAL MATCH
5. 始终考虑查询性能, 使用适当的索引和约束
6. 对于需要返回多个实体的查询, 使用 RETURN 子句明确指定要返回的内容
7. 避免使用可能导致性能问题的查询模式

# 示例如下
自然语言: "查找心血管和血栓栓塞综合征建议服用什么药物?"
Cypher: "match (p:Disease)-[r:recommand_drug]-(d:Drug) where p.name='心血管和血栓栓塞综合征' return d.name"

自然语言: "查找嗜铬细胞瘤这种疾病有哪些临床症状?"
Cypher: "match (p:Disease)-[r:has_symptom]-(s:Symptom) where p.name='嗜铬细胞瘤' return s.name"

自然语言: "查找小儿先天性巨结肠推荐哪些饮食有利康复?"
Cypher: "match (p:Disease)-[r:recommand_eat]-(f:Food) where p.name='小儿先天性巨结肠' return f.name"

现在请根据以下自然语言描述生成Cypher查询:
"""

def create_validation_prompt(cypher_query: str) -> str:
    return f"""
请分析以下Cypher查询, 指出其中的任何错误或潜在问题, 并提供改进建议:

{cypher_query}

请按以下格式回答:
错误: [列出所有错误]
建议: [提供改进建议]
"""