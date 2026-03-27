# main.py
from src.utils.logger import logger
import os
import sys
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from openai import OpenAI
from dotenv import load_dotenv
from agent_ln.neo4j_api.models import NL2CypherRequest, CypherResponse, ValidationRequest, ValidationResponse
from schemas import EXAMPLE_SCHEMA
from agent_ln.neo4j_api.prompts import create_system_prompt, create_validation_prompt
from agent_ln.neo4j_api.validators import CypherValidator, RuleBasedValidator

# 加载环境变量
load_dotenv()

# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# 获取 DeepSeek-V3.2 的 api key
deepseek_api_key = config.deepseek_api_key

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if all([neo4j_uri, neo4j_user, neo4j_password]):
        app.state.validator = CypherValidator(neo4j_uri, neo4j_user, neo4j_password)
    else:
        app.state.validator = RuleBasedValidator()
        
    yield
    
    # 关闭时清理
    if hasattr(app.state.validator, 'close'):
        app.state.validator.close()

# 创建FastAPI应用
app = FastAPI(title="NL2Cypher API", lifespan=lifespan)

# 初始化 DeepSeek 模型
client = OpenAI(
    api_key=deepseek_api_key, # 你的 DeepSeek API 密钥
    base_url=config.deepseek_base_url, # DeepSeek 的 API 端点
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_cypher_query(natural_language: str, query_type: str = None) -> str:
    """使用 DeepSeek 生成 Cypher 查询"""
    system_prompt = create_system_prompt(str(EXAMPLE_SCHEMA.model_dump()))
    
    user_prompt = natural_language
    if query_type:
        user_prompt = f"{query_type}查询: {natural_language}"
        
    try:
        response = client.chat.completions.create(
            model=config.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2048,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepSeek API错误: {str(e)}")

def explain_cypher_query(cypher_query: str) -> str:
    """解释Cypher查询"""
    try:
        response = client.chat.completions.create(
            model=config.deepseek_model,
            messages=[
                {"role": "system", "content": "你是一个Neo4j专家, 请用简单明了的语言解释Cypher查询."},
                {"role": "user", "content": f"请解释以下Cypher查询: \n{cypher_query}"}
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"无法生成解释: {str(e)}"

@app.post("/generate", response_model=CypherResponse)
async def generate_cypher(request: NL2CypherRequest):
    """生成Cypher查询端点"""
    # 利用 DeepSeek 生成 Cypher 查询
    cypher_query = generate_cypher_query(
        request.natural_language_query, 
        request.query_type.value if request.query_type else None
    )
    
    # 利用 DeepSeek 生成解释
    explanation = explain_cypher_query(cypher_query)
    
    # 验证查询
    is_valid, errors = app.state.validator.validate_against_schema(cypher_query, EXAMPLE_SCHEMA)
    
    # 计算置信度, 将基础置信度设置为0.9
    confidence = 0.9
    
    # 如果有潜在错误, 重新计算置信度 confidence
    if errors:
        confidence = max(0.3, confidence - len(errors) * 0.1)
        
    return CypherResponse(
        cypher_query=cypher_query,
        explanation=explanation,
        confidence=confidence,
        validated=is_valid,
        validation_errors=errors
    )

@app.post("/validate", response_model=ValidationResponse)
async def validate_cypher(request: ValidationRequest):
    """验证Cypher查询端点"""
    is_valid, errors = app.state.validator.validate_against_schema(request.cypher_query, EXAMPLE_SCHEMA)
    
    # 生成改进建议
    suggestions =[]
    if errors:
        try:
        response = client.chat.completions.create(
            model=config.deepseek_model,
                messages=[
                    {"role": "system", "content": "你是一个Neo4j专家, 请提供Cypher查询的改进建议."},
                    {"role": "user", "content": create_validation_prompt(request.cypher_query)}
                ],
                temperature=0.1,
                max_tokens=1024,
                stream=False
            )
            suggestions = [response.choices[0].message.content.strip()]
        except:
            suggestions =["无法生成建议"]
            
    return ValidationResponse(
        is_valid=is_valid,
        errors=errors,
        suggestions=suggestions
    )

@app.get("/schema")
async def get_schema():
    """获取图模式端点"""
    return EXAMPLE_SCHEMA.model_dump()

if __name__ == "__main__":
    # 因为我们项目中的主服务Agent启动在8103端口, 所以这个neo4j的服务端口另选一个8101即可
    uvicorn.run(app, host="0.0.0.0", port=8101)
