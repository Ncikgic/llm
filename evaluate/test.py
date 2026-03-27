#!/usr/bin/env python3
"""
Agent_LN 综合评测系统
整合：召回测试（使用 recall_test.py）、压力测试、LLM裁判测试
"""
from src.utils.logger import logger
import json
import time
import os
import sys
import requests
import ast
from datetime import datetime

# 设置路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, '/hy-tmp')

# 导入主程序模块
from src.models.model import generate_claude_answer, create_claude_client
import config

HISTORY_FILE = os.path.join(ROOT_DIR, "metrics_history.jsonl")

# ================= 压力测试 ===========================================
def run_stress_test(concurrency: int = 50, test_url: str = "http://127.0.0.1:8103/", 
                    duration: float = None) -> dict:
    """
    运行压力测试
    如果指定duration，则运行指定时长（秒）的压力测试，最长40秒
    否则运行固定请求数的压力测试
    """
    logger.info("\n" + "="*40)
    
    payload = {"question": "晚上睡不着觉，心里很焦虑怎么办？",
               "stream": True }
    response_times = []
    ttft_times = []
    failed_count = 0
    success_count = 0
    total_requests = 0
    timeout = 40  # 添加timeout变量定义
    
    def send_req():
        nonlocal failed_count, success_count, total_requests, response_times, ttft_times
        start = time.time()
        ttft = None
        try:
            res = requests.post(test_url, json=payload, timeout=timeout, stream=True)
            total_requests += 1
            if res.status_code == 200:
                for chunk in res.iter_content(chunk_size=1024):
                    if chunk:  # 收到第一个有效数据
                        first_token_time = time.time()
                        ttft = (first_token_time - start) * 1000  # 转为毫秒
                        ttft_times.append(ttft)
                        break  # 只取第一个token
                for _ in res.iter_content(chunk_size=1024):
                    pass  # 消耗剩余数据
                lat = (time.time() - start) * 1000
                response_times.append(lat)
                success_count += 1
                return lat
        except Exception:
            failed_count += 1
            total_requests += 1
        return -1  # 只返回一个值
    
    if duration is not None:
        if duration > 40:
            duration = 40        
        logger.info(f" 开始时长压力测试 (并发数: {concurrency}, 持续时间: {duration}秒)")        
        import threading
        from threading import Event        
        stop_event = Event()        
        def worker():
            while not stop_event.is_set():
                send_req()       
        # 启动工作线程
        threads = []
        for _ in range(concurrency):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)        
        # 运行指定时长
        start_total = time.time()
        time.sleep(duration)
        stop_event.set()        
        # 等待线程结束
        for t in threads:
            t.join(timeout=1)           
        total_time = time.time() - start_total      
        # 计算统计信息
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        avg_latency = sum(response_times) / len(response_times) if response_times else 0
        avg_ttft = sum(ttft_times) / len(ttft_times) if ttft_times else 0  # 平均TTFT
        throughput = total_requests / total_time if total_time > 0 else 0        
        logger.info(f"🔥 压测完成!")
        logger.info(f"   总请求数: {total_requests}")
        logger.info(f"   成功率: {success_rate:.1f}%")
        logger.info(f"   平均延迟: {avg_latency:.2f} ms")
        logger.info(f"   平均TTFT: {avg_ttft:.2f} ms")
        logger.info(f"   QPS: {throughput:.2f} 请求/秒")
        logger.info(f"   总耗时: {total_time:.2f} 秒")     
        return {
            "concurrency": concurrency,
            "duration_sec": duration,
            "total_requests": total_requests,
            "success_rate_pct": round(success_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_ttft_ms": round(avg_ttft, 2),
            "throughput_rps": round(throughput, 2),
            "total_cost_sec": round(total_time, 2)
        }

# ================= LLM 裁判测试 ===========================================
def run_llm_judge(test_queries: list, timeout: int = 40) -> dict:
    logger.info("\n" + "="*40)
    logger.info("开始 LLM-as-a-Judge 自动化打分")
    total_scores = {
        "reasoning": 0, 
        "empathy": 0,
        "faithfulness": 0,
        "helpfulness": 0
    }
    valid_count = 0
    
    def clean_json_response(response_text: str) -> str:
        """清理JSON响应，移除Markdown代码块标记和多余的空格"""
        # 移除 ```json 和 ``` 标记
        response_text = response_text.replace('```json', '').replace('```', '')
        # 移除前后的空白字符
        response_text = response_text.strip()
        # 尝试找到第一个 { 和最后一个 }
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end != 0:
            response_text = response_text[start:end]
        return response_text
    
    for query in test_queries:
        try:
            agent_answer_response = requests.post("http://127.0.0.1:8103/", json={"question": query}, timeout=timeout)
            agent_answer = agent_answer_response.json()
            logger.info(agent_answer)
            
            # 将 agent_answer 转换为字符串，确保在提示词中正确显示
            if isinstance(agent_answer, dict):
                # 如果响应是字典，尝试提取文本内容
                answer_text = agent_answer.get('response', str(agent_answer))
            else:
                answer_text = str(agent_answer)
            
            judge_prompt = f"""
        你是一个极其严厉、挑剔的资深心理医生和AI评测专家。你的任务是评估一个心理疗愈AI助手的回答质量。
        你必须对AI的回答进行无情的审视，只要存在"废话"、"套话"、"像维基百科一样冰冷"的问题，就必须狠狠扣分！
        【评测维度与严格扣分规则】（基础分均为100分）

        1. 共情与温度 (Empathy)
        - 扣分项：没有先安抚用户情绪（-20分）；像机器一样冷冰冰地罗列知识（-30分）；语气带有高高在上的说教意味（-40分）。
        - 要求：必须像真人心理咨询师一样温暖。

        2. 忠实度 (Faithfulness)
        - 扣分项：；出现医学幻觉或不严谨的承诺（-50分）;给出未经科学验证的治疗承诺（如"保证治愈"）（-40分）;建议可能有害的非专业干预措施（-50分）;未基于召回的专业文档输出建议，仅凭模型自身知识生成（-35分）

        3. 针对性与专业性 (Helpfulness)
        - 扣分项：回答了与用户具体痛点无关的周边废话（-20分）；车轱辘话来回说（-15分）；建议太大太空（如"建议放松心态"）毫无实操性（-25分）;只讲废话没有给出任何心理学派（如CBT、正念）的具体干预动作（-30分）；长篇大论不分段，用户看不下去（-15分）。
        【评分基准线（严格遵守）】
        - 60分（及格线）：回答安全、准确，但像"百度百科"或"说明书"，缺乏共情，给出的是"多喝水、早点睡"等机械的通用建议。
        - 75分（良好线）：有一定的人性化语气，建议稍微具体，但仍有轻微的AI AI说教感（"爹味"）。
        - 90分+（优秀线）：极强的共情能力（接纳用户情绪），像真实的心理医生一样温暖，且给出了极其具体、可操作的心理学干预技巧（如CBT、正念等）。
        - 60分以下（不及格）：答非所问、存在医疗幻觉、出现危险建议或严重忽略用户的负面情绪。
    
        【评估参考示例 (Few-Shot)】
    
        示例 1：
        用户提问：我最近每天晚上都在被窝里哭，觉得压力特别大，快崩溃了。
        AI回答：压力大和失眠是由多种原因引起的。建议您：1. 保持规律作息；2. 睡前不要喝咖啡；3. 适当运动放松心情。如果情况严重，请及时就医咨询专业医生。
        评分输出：
        {{
            "reasoning": "回答准确但极其机械。没有任何对用户'在被窝里哭'的悲伤情绪进行共情和接纳。直接抛出说明书式的1234条建议，缺乏温度，但没有常识性错误，刚够及格线。",
            "empathy": 40,
            "faithfulness": 65,
            "helpfulness": 60        }}       
        示例 2：
        用户提问：我最近每天晚上都在被窝里哭，觉得压力特别大，快崩溃了。
        AI回答：听到你这么说，我能感受到你现在有多疲惫和无助。一个人扛着这么大的压力，每天晚上还要独自面对情绪的崩溃，真的非常辛苦（拥抱）。你不需要强迫自己立刻坚强起来，允许自己哭一会儿是可以的。如果你愿意的话，今晚我们可以尝试一个简单的'4-7-8呼吸法'来让身体先放松下来：吸气4秒，憋气7秒，呼气8秒。我就在这里陪着你，你绝对不是一个人在面对这些。
        评分输出：
        {{  
            "reasoning": "极好的共情与情绪接纳，没有冰冷的说教。用温暖的语言安抚了用户，并且给出了马上就能在被窝里操作的'4-7-8呼吸法'，专业且具有陪伴感。",
            "empathy": 85,
            "faithfulness": 85,
            "helpfulness": 80        }}
        【现在请你评估以下真实对话】
        用户提问: {query}
        AI回答: {answer_text}
        请严格按照以上基准线和示例的尺度进行打分。
        你必须且只能输出合法的JSON格式，严禁输出其他任何解释性文字！
        必须先输出思考扣分过程，再输出最终得分！
        {{
            "reasoning": "用一句话指出AI回答中最致命的缺点和扣分理由",
            "empathy": <根据扣分规则打分，0-100的整数>,
            "faithfulness": <根据扣分规则打分，0-100的整数>,
            "helpfulness": <根据扣分规则打分，0-100的整数>
        }}
        """
            client = create_claude_client()
            scores_text = generate_claude_answer(client, judge_prompt)
            
            # 清理响应文本
            cleaned_scores_text = clean_json_response(scores_text)
            
            try:
                scores = ast.literal_eval(cleaned_scores_text)
            except (SyntaxError, ValueError) as e:
                # 如果ast.literal_eval失败，尝试使用json.loads
                try:
                    import json as json_module
                    scores = json_module.loads(cleaned_scores_text)
                except json_module.JSONDecodeError:
                    logger.info(f"无法解析JSON响应: {cleaned_scores_text}")
                    logger.info(f"原始响应: {scores_text}")
                    continue
            
            logger.info(scores)
            total_scores["empathy"] += scores.get("empathy", 0)
            total_scores["faithfulness"] += scores.get("faithfulness", 0)
            total_scores["helpfulness"] += scores.get("helpfulness", 0)
            valid_count += 1
            time.sleep(0.5) # 防止API限流
        
        except Exception as e:
            logger.info(f"处理查询 '{query}' 时出错: {e}")
            continue
        
    if valid_count == 0: 
        logger.info("警告: 没有有效的评测结果")
        return {}    
    
    avg_scores = {k: round(v / valid_count, 2) for k, v in total_scores.items() if k != "reasoning"}
    avg_scores["overall"] = round(sum(avg_scores.values()) / len(avg_scores), 2)   
    logger.info(f"评测完成! 综合得分: {avg_scores['overall']}/100.0")
    logger.info(f"共情度: {avg_scores.get('empathy', 0)}")
    logger.info(f"专业度: {avg_scores.get('faithfulness', 0)}")
    logger.info(f"帮助度: {avg_scores.get('helpfulness', 0)}")    
    return avg_scores

# ================= 主函数 ===========================================
def main():
    """主函数：运行所有测试"""
    version = config.app_version
    logger.info(f"\n{'='*60}")
    logger.info(f"启动 Agent_LN 综合评测系统 | 当前版本: {version}")
    logger.info(f"{'='*60}")
    
    # 准备测试数据
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "version": version,
        "results": {}
    }
    
    # 1. 运行召回测试（使用 recall_test.py）
    try:
        import evaluate.recall_test as recall_test
        logger.info("\n" + "="*40)
        logger.info("开始召回测试（使用 recall_test.py）")
        
        # 调用 recall_test 的主函数
        recall_results = recall_test.main()
        
        if recall_results and "recall" in recall_results:
            metrics["results"]["recall"] = recall_results["recall"]
            logger.info(f"✓ 召回测试完成")
        else:
            logger.info("✗ 召回测试失败或没有结果")
            metrics["results"]["recall"] = {}
            
    except ImportError as e:
        logger.info(f"✗ 导入 recall_test 模块失败: {e}")
        metrics["results"]["recall"] = {}
    except Exception as e:
        logger.info(f"✗ 召回测试过程中出现错误: {e}")
        metrics["results"]["recall"] = {}
    
    # 2. 运行压力测试
    logger.info("\n" + "="*40)
    logger.info("开始压力测试")
    try:
        stress_results = run_stress_test(concurrency=10, duration=10)  # 10并发，持续10秒
        metrics["results"]["stress"] = stress_results
        logger.info(f"✓ 压力测试完成")
    except Exception as e:
        logger.info(f"✗ 压力测试过程中出现错误: {e}")
        metrics["results"]["stress"] = {}
    
    # 3. 运行 LLM 裁判测试
    logger.info("\n" + "="*40)
    logger.info("开始 LLM 裁判测试")
    try:
        # 读取 testqa.txt 中的查询
        testqa_path = os.path.join(ROOT_DIR, "evaluate", "testqa.txt")
        if os.path.exists(testqa_path):
            with open(testqa_path, 'r', encoding='utf-8') as f:
                testqa_queries = [line.strip().split(' ', 1)[1] for line in f if line.strip()]
            
            if testqa_queries:
                judge_queries = testqa_queries[:5]  # 限制数量，避免测试时间过长
                judge_results = run_llm_judge(judge_queries, timeout=40)
                metrics["results"]["llm_judge"] = judge_results
                logger.info(f"✓ LLM裁判测试完成")
            else:
                logger.info("✗ testqa.txt 中没有有效的查询")
                metrics["results"]["llm_judge"] = {}
        else:
            logger.info(f"✗ 找不到 testqa.txt 文件: {testqa_path}")
            metrics["results"]["llm_judge"] = {}
    except Exception as e:
        logger.info(f"✗ LLM裁判测试过程中出现错误: {e}")
        metrics["results"]["llm_judge"] = {}
    
    # 4. 汇总结果
    logger.info(f"\n{'='*60}")
    logger.info("评测汇总")
    logger.info(f"{'='*60}")
    
    if metrics["results"].get("recall"):
        logger.info("\n召回测试结果:")
        recall_data = metrics["results"]["recall"]
        if isinstance(recall_data, dict) and "recall" in recall_data:
            for collection, metrics_data in recall_data["recall"].items():
                logger.info(f"  {collection}:")
                logger.info(f"    测试数: {metrics_data.get('count', 0)}")
                logger.info(f"    Recall@1: {metrics_data.get('r1', 0):.4f}")
                logger.info(f"    Recall@3: {metrics_data.get('r3', 0):.4f}")
                logger.info(f"    Recall@5: {metrics_data.get('r5', 0):.4f}")
                logger.info(f"    MRR: {metrics_data.get('mrr', 0):.4f}")
    
    if metrics["results"].get("stress"):
        logger.info("\n压力测试结果:")
        stress_data = metrics["results"]["stress"]
        logger.info(f"  总请求数: {stress_data.get('total_requests', 0)}")
        logger.info(f"  成功率: {stress_data.get('success_rate_pct', 0):.1f}%")
        logger.info(f"  平均延迟: {stress_data.get('avg_latency_ms', 0):.2f} ms")
        logger.info(f"  平均TTFT: {stress_data.get('avg_ttft_ms', 0):.2f} ms")
        logger.info(f"  QPS: {stress_data.get('throughput_rps', 0):.2f} 请求/秒")
    
    if metrics["results"].get("llm_judge"):
        logger.info("\nLLM裁判测试结果:")
        judge_data = metrics["results"]["llm_judge"]
        logger.info(f"  综合得分: {judge_data.get('overall', 0):.2f}/100")
        logger.info(f"  共情度: {judge_data.get('empathy', 0):.2f}")
        logger.info(f"  专业度: {judge_data.get('faithfulness', 0):.2f}")
        logger.info(f"  帮助度: {judge_data.get('helpfulness', 0):.2f}")
    
    # 5. 保存结果到历史文件
    try:
        with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        logger.info(f"\n✓ 评测结果已保存到: {HISTORY_FILE}")
    except Exception as e:
        logger.info(f"\n✗ 保存结果失败: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info("所有测试完成!")
    logger.info(f"{'='*60}")
    
    return metrics

if __name__ == "__main__":
    main()
