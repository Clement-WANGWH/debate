import json
import asyncio
import sys
import os
from typing import List, Dict, Any, Optional
from async_llm import AsyncQwenLLM, LLMConfig
from config import get_qwen_config
import time

class QwenEvaluator:
    """基于Qwen的LLM评估器"""
    
    def __init__(self, llm_config: Optional[Dict] = None, system_prompt: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            llm_config: LLM配置，如果为None则使用全局配置
            system_prompt: 系统提示词
        """
        if llm_config is None:
            llm_config = get_qwen_config()
        
        self.llm_config = LLMConfig(llm_config)
        
        # 默认系统提示词
        if system_prompt is None:
            system_prompt = """
你是一个专业的答案评估专家，负责判断答案的正确性。
你的任务是将提供的答案与标准答案进行比较，并判断其是否正确。

评估规则：
1. 答案应该包含标准答案中的关键信息
2. 答案可以用不同的方式表述，但必须传达相同的含义
3. 部分正确的答案应该获得部分分数
4. 完全错误的答案得分为0

请提供0到1之间的分数，其中：
- 1: 完全正确的答案
- 0.8-0.9: 基本正确，有轻微瑕疵
- 0.5-0.7: 部分正确
- 0.1-0.4: 有一定道理但错误较多
- 0: 完全错误的答案

回答格式必须严格按照以下格式：
分数: [你的分数，必须是0-1之间的小数]
解释: [你的简短解释，1-2句话]
"""
        
        self.llm = AsyncQwenLLM(self.llm_config, system_prompt)
    
    async def evaluate_single(self, query: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        评估单个答案
        
        Args:
            query: 原始问题
            answer: 待评估的答案
            ground_truth: 标准答案
            
        Returns:
            评估结果字典
        """
        prompt = f"""
问题: {query}

标准答案: {ground_truth}

待评估答案: {answer}

请评估待评估答案相对于标准答案的正确性，并提供分数和解释。
"""
        
        try:
            response = await self.llm(prompt)
            
            # 解析评估结果
            score = self._extract_score(response)
            explanation = self._extract_explanation(response)
            
            return {
                "score": score,
                "explanation": explanation,
                "raw_response": response,
                "success": True
            }
            
        except Exception as e:
            print(f"评估出错: {e}")
            return {
                "score": 0.0,
                "explanation": f"评估过程中出现错误: {str(e)}",
                "raw_response": "",
                "success": False
            }
    
    def _extract_score(self, response: str) -> float:
        """从响应中提取分数"""
        import re
        
        # 尝试多种模式匹配分数
        patterns = [
            r'分数[：:]\s*([0-1](?:\.[0-9]+)?)',
            r'得分[：:]\s*([0-1](?:\.[0-9]+)?)',
            r'评分[：:]\s*([0-1](?:\.[0-9]+)?)',
            r'SCORE[：:]\s*([0-1](?:\.[0-9]+)?)',
            r'(?:^|\n)([0-1](?:\.[0-9]+)?)(?:\s|$)',  # 单独的数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 1:
                        return score
                except ValueError:
                    continue
        
        # 如果无法提取分数，尝试基于关键词启发式判断
        response_lower = response.lower()
        if any(word in response_lower for word in ['完全正确', '完全对', '正确', '对的']):
            return 0.9
        elif any(word in response_lower for word in ['基本正确', '基本对', '大致正确']):
            return 0.7
        elif any(word in response_lower for word in ['部分正确', '部分对']):
            return 0.5
        elif any(word in response_lower for word in ['错误', '不对', '不正确']):
            return 0.1
        else:
            return 0.3  # 默认分数
    
    def _extract_explanation(self, response: str) -> str:
        """从响应中提取解释"""
        import re
        
        # 尝试提取解释
        patterns = [
            r'解释[：:]\s*(.*?)(?:\n|$)',
            r'说明[：:]\s*(.*?)(?:\n|$)',
            r'原因[：:]\s*(.*?)(?:\n|$)',
            r'EXPLANATION[：:]\s*(.*?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                explanation = match.group(1).strip()
                if explanation:
                    return explanation
        
        # 如果没有找到标准格式，返回整个响应
        lines = response.split('\n')
        for line in lines:
            if line.strip() and '分数' not in line and 'SCORE' not in line.upper():
                return line.strip()
        
        return "未能提取详细解释"
    
    async def evaluate_batch(self, results: List[Dict[str, Any]], 
                           batch_size: int = 10, delay: float = 0.5) -> Dict[str, Any]:
        """
        批量评估结果
        
        Args:
            results: 结果列表，每个结果包含query, answer, ground_truth
            batch_size: 批处理大小
            delay: 请求间延迟（秒）
            
        Returns:
            评估结果汇总
        """
        evaluated_results = []
        total_score = 0.0
        success_count = 0
        
        print(f"开始批量评估 {len(results)} 个结果...")
        
        for i in range(0, len(results), batch_size):
            batch = results[i:i+batch_size]
            print(f"\n处理批次 {i//batch_size + 1}/{(len(results) + batch_size - 1)//batch_size}")
            
            # 并行处理一个批次
            batch_tasks = []
            for j, result in enumerate(batch):
                task = self.evaluate_single(
                    query=result.get("query", ""),
                    answer=result.get("answer", ""),
                    ground_truth=result.get("ground_truth", "")
                )
                batch_tasks.append(task)
            
            # 等待批次完成
            batch_evaluations = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理批次结果
            for j, (result, evaluation) in enumerate(zip(batch, batch_evaluations)):
                idx = i + j + 1
                
                if isinstance(evaluation, Exception):
                    print(f"结果 {idx} 评估失败: {evaluation}")
                    evaluation = {
                        "score": 0.0,
                        "explanation": f"评估异常: {str(evaluation)}",
                        "success": False
                    }
                
                # 合并结果
                evaluated_result = result.copy()
                evaluated_result.update(evaluation)
                evaluated_results.append(evaluated_result)
                
                if evaluation["success"]:
                    total_score += evaluation["score"]
                    success_count += 1
                
                print(f"结果 {idx}: 分数={evaluation['score']:.3f}, 解释={evaluation['explanation'][:50]}...")
            
            # 批次间延迟
            if i + batch_size < len(results):
                await asyncio.sleep(delay)
        
        # 计算汇总统计
        average_score = total_score / max(1, success_count)
        
        summary = {
            "total_results": len(results),
            "successful_evaluations": success_count,
            "failed_evaluations": len(results) - success_count,
            "average_score": average_score,
            "total_score": total_score,
            "success_rate": success_count / len(results),
            "results": evaluated_results
        }
        
        return summary
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """获取使用摘要"""
        return self.llm.get_usage_summary()

async def evaluate_results_file(file_path: str, output_path: Optional[str] = None,
                               batch_size: int = 10, delay: float = 0.5) -> Dict[str, Any]:
    """
    评估结果文件
    
    Args:
        file_path: 输入文件路径
        output_path: 输出文件路径，如果为None则自动生成
        batch_size: 批处理大小
        delay: 请求间延迟
        
    Returns:
        评估结果汇总
    """
    # 加载结果文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载文件出错: {e}")
        return {"error": f"Failed to load file: {e}"}
    
    if "results" not in data or not data["results"]:
        print("错误: JSON文件必须包含非空的'results'数组")
        return {"error": "Invalid file format"}
    
    # 创建评估器
    evaluator = QwenEvaluator()
    
    # 批量评估
    summary = await evaluator.evaluate_batch(
        results=data["results"],
        batch_size=batch_size,
        delay=delay
    )
    
    # 添加使用摘要
    summary["llm_usage"] = evaluator.get_usage_summary()
    summary["evaluation_time"] = time.time()
    
    # 生成输出文件路径
    if output_path is None:
        base_name = os.path.splitext(file_path)[0]
        output_path = f"{base_name}_qwen_evaluated.json"
    
    # 保存结果
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存至: {output_path}")
    except Exception as e:
        print(f"保存文件出错: {e}")
    
    # 打印汇总信息
    print(f"\n=== 评估汇总 ===")
    print(f"总结果数: {summary['total_results']}")
    print(f"成功评估: {summary['successful_evaluations']}")
    print(f"失败评估: {summary['failed_evaluations']}")
    print(f"平均分数: {summary['average_score']:.4f}")
    print(f"成功率: {summary['success_rate']:.2%}")
    
    # 打印使用信息
    usage = summary["llm_usage"]
    print(f"\n=== 使用统计 ===")
    print(f"总token数: {usage['total_tokens']}")
    print(f"总成本: ${usage['total_cost']:.6f}")
    print(f"调用次数: {usage['call_count']}")
    
    return summary

async def main():
    """主程序"""
    if len(sys.argv) < 2:
        print("用法: python qwen_evaluator.py <results_file.json> [output_file.json] [batch_size] [delay]")
        print("示例: python qwen_evaluator.py results.json evaluated_results.json 5 1.0")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    delay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    
    print(f"开始评估文件: {input_file}")
    print(f"批次大小: {batch_size}, 延迟: {delay}秒")
    
    summary = await evaluate_results_file(
        file_path=input_file,
        output_path=output_file,
        batch_size=batch_size,
        delay=delay
    )
    
    if "error" in summary:
        print(f"评估失败: {summary['error']}")
        return 1
    
    print("评估完成！")
    return 0

if __name__ == "__main__":
    async def test_evaluator():
        """测试评估器"""
        print("测试Qwen评估器...")
        
        evaluator = QwenEvaluator()
        
        result = await evaluator.evaluate_single(
            query="3 + 5 = ?",
            answer="答案是8",
            ground_truth="8"
        )
        
        print(f"评估结果: {result}")
        print(f"使用摘要: {evaluator.get_usage_summary()}")
    
    if len(sys.argv) > 1:
        asyncio.run(main())
    else:
        print("提示: 请使用命令行参数运行评估，或配置API密钥后取消注释测试代码")