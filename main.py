#!/usr/bin/env python3

import asyncio
import json
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback

from enhanced_debate import EnhancedMultiagentDebate, quick_debate
from code.debate.benchmarks.qwen_evaluator import evaluate_results_file, QwenEvaluator
from config import ConfigManager, get_config, get_qwen_config, get_debate_config
from utils import dataset_2_process_fn
import time

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化实验运行器"""
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        print(f"=== 多智能体辩论系统 v2.0 ===")
        print(f"模型: {self.config.qwen.model}")
        print(f"智能体数量: {self.config.debate.num_agents}")
        print(f"最大轮数: {self.config.debate.max_rounds}")
        print(f"剪枝: {'启用' if self.config.debate.enable_prune else '禁用'}")
        print(f"反思: {'启用' if self.config.debate.enable_reflect else '禁用'}")
        print(f"缓存: {'启用' if self.config.debate.enable_cache else '禁用'}")
        print("=" * 40)
    
    def _load_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """加载数据集"""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        print(f"加载数据集: {dataset_path}")
        
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            if dataset_path.endswith('.jsonl'):
                for line_num, line in enumerate(f):
                    if max_samples and len(data) >= max_samples:
                        break
                    try:
                        item = json.loads(line.strip())
                        if 'query' not in item and 'question' in item:
                            item['query'] = item['question']
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"警告: 第{line_num+1}行JSON格式错误: {e}")
            else:  # .json
                json_data = json.load(f)
                if isinstance(json_data, list):
                    data = json_data[:max_samples] if max_samples else json_data
                elif isinstance(json_data, dict) and 'data' in json_data:
                    data = json_data['data'][:max_samples] if max_samples else json_data['data']
                else:
                    raise ValueError(f"不支持的JSON格式: {dataset_path}")
        
        print(f"加载了 {len(data)} 个样本")
        return data
    
    async def run_single_debate(self, question: str, question_id: Optional[str] = None) -> Dict[str, Any]:
        """运行单次辩论"""
        debate_config = self.config_manager.get_debate_config()
        debate_system = EnhancedMultiagentDebate(config=debate_config)
        
        if question_id is None:
            question_id = str(hash(question))
        
        data = {"query": question, "id": question_id}
        
        print(f"\n问题: {question}")
        print("-" * 60)
        
        start_time = time.time()
        result = await debate_system.debate(data)
        end_time = time.time()
        
        result["execution_time"] = end_time - start_time
        result["cost_summary"] = debate_system.get_cost_summary()
        result["system_stats"] = debate_system.get_system_stats()
        
        print(f"\n结果: {result['answer']}")
        print(f"轮数: {result['rounds']}")
        print(f"共识: {'达成' if result['consensus'] else '未达成'}")
        print(f"缓存: {'命中' if result.get('cached', False) else '未命中'}")
        print(f"执行时间: {result['execution_time']:.2f}秒")
        print(f"成本: ${result['cost_summary']['total_cost']:.6f}")
        
        return result
    
    async def run_batch_experiments(self, dataset_path: str, dataset_type: str = "default", 
                                  max_samples: Optional[int] = None,
                                  output_dir: str = "results") -> str:
        """运行批量实验"""
        
        # 加载数据集
        data = self._load_dataset(dataset_path, max_samples)
        if not data:
            raise ValueError("数据集为空")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_file = os.path.join(output_dir, f"{dataset_name}_{len(data)}samples_{timestamp}.json")
        
        print(f"\n开始批量实验")
        print(f"数据集: {dataset_path}")
        print(f"样本数: {len(data)}")
        print(f"输出文件: {output_file}")
        print("=" * 60)
        
        # 创建辩论系统
        debate_config = self.config_manager.get_debate_config()
        debate_system = EnhancedMultiagentDebate(dataset=dataset_type, config=debate_config)
        
        results = []
        total_start_time = time.time()
        
        for i, item in enumerate(data):
            print(f"\n处理样本 {i+1}/{len(data)}")
            
            try:
                start_time = time.time()
                result = await debate_system.debate(item)
                end_time = time.time()
                
                # 添加额外信息
                result["sample_index"] = i
                result["execution_time"] = end_time - start_time
                result["ground_truth"] = item.get("answer", item.get("ground_truth", ""))
                
                results.append(result)
                
                print(f"问题: {result['query'][:100]}...")
                print(f"答案: {result['answer']}")
                print(f"标准答案: {result.get('ground_truth', 'N/A')}")
                print(f"时间: {result['execution_time']:.2f}秒")
                
            except Exception as e:
                print(f"样本 {i+1} 处理失败: {e}")
                traceback.print_exc()
                
                # 记录失败的样本
                results.append({
                    "sample_index": i,
                    "query": item.get("query", ""),
                    "error": str(e),
                    "answer": "处理失败",
                    "ground_truth": item.get("answer", item.get("ground_truth", ""))
                })
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # 生成实验报告
        experiment_summary = {
            "experiment_info": {
                "dataset_path": dataset_path,
                "dataset_type": dataset_type,
                "total_samples": len(data),
                "successful_samples": len([r for r in results if "error" not in r]),
                "failed_samples": len([r for r in results if "error" in r]),
                "total_time": total_time,
                "average_time_per_sample": total_time / len(data),
                "timestamp": timestamp,
                "config": self.config_manager.get_debate_config()
            },
            "system_stats": debate_system.get_system_stats(),
            "cost_summary": debate_system.get_cost_summary(),
            "results": results
        }
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, ensure_ascii=False, indent=2)
        
        # 打印实验总结
        print(f"\n{'='*60}")
        print(f"实验完成！")
        print(f"总样本数: {len(data)}")
        print(f"成功处理: {experiment_summary['experiment_info']['successful_samples']}")
        print(f"失败处理: {experiment_summary['experiment_info']['failed_samples']}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均耗时: {total_time/len(data):.2f}秒/样本")
        print(f"总成本: ${experiment_summary['cost_summary']['total_cost']:.6f}")
        print(f"结果保存至: {output_file}")
        
        return output_file
    
    async def evaluate_results(self, results_file: str, output_file: Optional[str] = None) -> str:
        """评估实验结果"""
        print(f"\n开始评估结果文件: {results_file}")
        
        summary = await evaluate_results_file(
            file_path=results_file,
            output_path=output_file,
            batch_size=5,  # 较小的批次避免频率限制
            delay=1.0      # 1秒延迟
        )
        
        if "error" in summary:
            raise Exception(f"评估失败: {summary['error']}")
        
        output_path = output_file or results_file.replace('.json', '_evaluated.json')
        print(f"评估完成，结果保存至: {output_path}")
        
        return output_path

def create_sample_config():
    """创建示例配置文件"""
    config_content = """# Qwen多智能体辩论系统配置文件

# Qwen API配置
qwen:
  model: "qwen-turbo"  # 可选: qwen-turbo, qwen-plus, qwen-max, qwen-flash
  api_key: "sk-3e0c2ad56ad540948ceee1a469e05115"  # 请替换为你的Qwen API密钥
  temperature: 0.7
  top_p: 0.9
  max_tokens: 2000
  base_url: "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 辩论系统配置  
debate:
  num_agents: 3          # 智能体数量 (1-10)
  max_rounds: 3          # 最大辩论轮数 (1-10)
  enable_prune: true     # 启用剪枝机制 (true/false)
  enable_reflect: true   # 启用反思机制 (true/false)
  strict_mode: true      # 严格模式 (true/false)
  num_injections: 0      # 错误注入数量，仅用于测试 (0-5)
  enable_cache: true     # 启用EPIC缓存机制 (true/false)
  cache_size: 10000      # 缓存大小 (1000-100000)
"""
    
    with open("config.yaml", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("示例配置文件 config.yaml 已创建，请编辑后使用")

async def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="Qwen多智能体辩论系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 创建配置文件模板
  python main.py --create-config
  
  # 单次辩论
  python main.py --single "小明有10个苹果，给了小红3个，还剩多少个？"
  
  # 批量实验
  python main.py --batch data/gsm8k_test.jsonl --dataset gsm8k --max-samples 100
  
  # 评估结果
  python main.py --evaluate results/gsm8k_100samples_20241201_120000.json
  
  # 完整流程（实验+评估）
  python main.py --batch data/gsm8k_test.jsonl --dataset gsm8k --evaluate --max-samples 50
        """
    )
    
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--create-config", action="store_true",
                       help="创建示例配置文件")
    
    parser.add_argument("--single", type=str,
                       help="单次辩论模式，提供问题文本")
    parser.add_argument("--batch", type=str,
                       help="批量实验模式，提供数据集文件路径")
    parser.add_argument("--evaluate", type=str, nargs="?", const=True,
                       help="评估结果文件（可单独使用或与--batch组合）")
    
    parser.add_argument("--dataset", type=str, default="default",
                       choices=["gsm8k", "hotpotqa", "math", "mmlu", "default"],
                       help="数据集类型 (默认: default)")
    parser.add_argument("--max-samples", type=int,
                       help="最大样本数（用于限制实验规模）")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="输出目录 (默认: results)")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    if not (args.single or args.batch or args.evaluate):
        parser.print_help()
        return
    
    try:
        runner = ExperimentRunner(args.config)
        
        if args.single:
            result = await runner.run_single_debate(args.single)
            
            output_file = os.path.join(args.output_dir, f"single_debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存至: {output_file}")
        
        elif args.batch:
            results_file = await runner.run_batch_experiments(
                dataset_path=args.batch,
                dataset_type=args.dataset,
                max_samples=args.max_samples,
                output_dir=args.output_dir
            )
            
            if args.evaluate is True:
                evaluated_file = await runner.evaluate_results(results_file)
                print(f"\n完整流程完成！")
                print(f"实验结果: {results_file}")
                print(f"评估结果: {evaluated_file}")
        
        elif args.evaluate and args.evaluate is not True:
            evaluated_file = await runner.evaluate_results(args.evaluate)
            print(f"\n评估完成: {evaluated_file}")
        
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
        return 1
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        traceback.print_exc()
        return 1
    
    print("\n程序执行完成!")
    return 0

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)