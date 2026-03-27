# schemas.py

import os
from pydantic import BaseModel
from typing import Dict, List, Optional

class NodeSchema(BaseModel):
    label: str
    properties: Dict[str, str] # 属性名: 类型

class RelationshipSchema(BaseModel):
    type: str
    from_node: str # 起始节点标签
    to_node: str # 目标节点标签
    properties: Dict[str, str] # 属性名: 类型

class GraphSchema(BaseModel):
    nodes: List[NodeSchema]
    relationships: List[RelationshipSchema]

# 示例图模式 (按照你在构建知识图谱neo4j数据库时的定义schema来填充下面的示例)
EXAMPLE_SCHEMA = GraphSchema(
    # 节点的名称一定要严格保持跟neo4j一致
    nodes=[
        NodeSchema(label="Disease", properties={"name": "string"}),
        NodeSchema(label="Drug", properties={"name": "string"}),
        NodeSchema(label="Food", properties={"name": "string"}),
        NodeSchema(label="Symptom", properties={"name": "string"})
    ],
    # 关系的相关字段一定要严格保持跟neo4j一致, 大小写都不能错
    relationships=[
        RelationshipSchema(
            type="has_symptom", 
            from_node="Disease", 
            to_node="Symptom",
            properties={}
        ),
        RelationshipSchema(
            type="recommand_drug", 
            from_node="Disease", 
            to_node="Drug",
            properties={}
        ),
        RelationshipSchema(
            type="recommand_eat", 
            from_node="Disease", 
            to_node="Food",
            properties={}
        ),
    ]
)

if __name__ == '__main__':
    res = str(EXAMPLE_SCHEMA.model_dump())
    logger.info(res)