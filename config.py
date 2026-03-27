# 应用配置
app_name = "agent_ln"
app_version = "1.1.0"
# API密钥配置（敏感信息建议通过环境变量设置）
import os
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "sk-9636939aa42c48cda0eabe7a4298418d")
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY", "sk-jthrzcfbrpwofhwmgiessefyzxorsliqbcbpykgbsfpugmxx")
claude_api_key = os.getenv("CLAUDE_API_KEY", "sk-4TUfM4nFoDfAyq3dOIMQSgbFIMe50G6RZerv0Q2ZbDnyhDgX")
claude_base_url = os.getenv("CLAUDE_BASE_URL", "https://api.whatai.cc/")
# 嵌入模型配置
embedding_provider = "siliconflow"
embedding_model = "Qwen/Qwen3-Embedding-4B"
embedding_batch_size = 32
embedding_dimensions = 2560
# LLM配置
llm_default_provider = "qwen"
# Qwen配置
qwen_model_path = os.getenv("QWEN_MODEL_PATH", "/hy-tmp/agent_ln/src/models/Qwen3-14B-bnb-4bit")
qwen_model_id = "Qwen3-14B-bnb-4bit"
qwen_device = "cuda"
qwen_max_new_tokens = 512
qwen_temperature = 0.7
qwen_top_p = 0.9
qwen_do_sample = True
# DeepSeek配置
deepseek_model = "deepseek-chat"
deepseek_base_url = "https://api.deepseek.com/v1"
# Claude配置
claude_model = "claude-sonnet-4-6"
claude_max_tokens = 1000
# 向量数据库配置
vector_db_provider = "milvus"
milvus_uri = os.getenv("MILVUS_URI", "/hy-tmp/agent_ln/milvus_agent.db")
milvus_consistency_level = "Bounded"
collections_list=['psyqa_collection','pdf_collection','neo4j_collection']
pdf_output="/hy-tmp/agent_ln/data"
pdf_input="/hy-tmp/agent_ln/data/texts"
# 集合配置
psyqa_collection_name = "psyqa_collection"
pdf_collection_name = "pdf_collection"
# 数据文件路径
psyqa_data_path = os.getenv("PSYQA_DATA_PATH", "/hy-tmp/agent_ln/data/psyqa_cleaned.jsonl")
pdf_data_path = os.getenv("PDF_DATA_PATH", "./pdf_output/pdf_detailed_text.xlsx")
# 日志配置
log_file = "agent_ln.log"
