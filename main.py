#!/usr/bin/env python3

import asyncio
import json
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from enhanced_debate import ProgressiveMultiagentDebate, quick_debate
from config import ConfigManager

class ExperimentRunner:
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        print(f"多智能体辩论系统 v2.0")
        print(f"模型: {self.config.qwen.model}")
        print(f"智能体: {self.config.debate.num_agents}")
        print("-" * 40)
    
    def _load_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")
        
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            if dataset_path.endswith('.jsonl'):
                for line in f:
                    if max_samples and len(data) >= max_samples:
                        break
                    try:
                        item = json.loads(line.strip())
                        if 'query' not in item and 'question' in item:
                            item['query'] = item['question']
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
            else:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    data = json_data[:max_samples] if max_samples else json_data
                elif isinstance(json_data, dict) and 'data' in json_data:
                    data = json_data['data'][:max_samples] if max_samples else json_data['data']
        
        print(f"加载 {len(data)} 个样本")
        return data
    
    async def run_single_debate(self, question: str) -> Dict[str, Any]:
        debate_config = self.config_manager.get_debate_config()
        debate_system = ProgressiveMultiagentDebate(config=debate_config)
        
        data = {"query": question, "id": str(hash(question))}
        
        print(f"\n问题: {question}")
        print("-" * 60)
        
        result = await debate_system.debate(data)
        
        print(f"\n答案: {result['answer']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"轮次: {result['rounds']}")
        print(f"共识: {'达成' if result['consensus_reached'] else '未达成'}")
        
        cost_summary = debate_system.get_cost_summary()
        print(f"成本: ${cost_summary['total_cost']:.6f}")
        
        return result
    
    async def run_batch_experiments(self, dataset_path: str, dataset_type: str = "default",
                                  max_samples: Optional[int] = None,
                                  output_dir: str = "results") -> str:
        data = self._load_dataset(dataset_path, max_samples)
        if not data:
            raise ValueError("数据集为空")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_file = os.path.join(output_dir, f"{dataset_name}_{len(data)}samples_{timestamp}.json")
        
        print(f"\n开始批量实验")
        print(f"样本数: {len(data)}")
        print("=" * 60)
        
        debate_config = self.config_manager.get_debate_config()
        debate_system = ProgressiveMultiagentDebate(dataset=dataset_type, config=debate_config)
        
        results = []
        successful = 0
        
        for i, item in enumerate(data):
            print(f"\n[{i+1}/{len(data)}] 处理中...")
            
            try:
                result = await debate_system.debate(item)
                result["sample_index"] = i
                result["ground_truth"] = item.get("answer", item.get("ground_truth", ""))
                results.append(result)
                successful += 1
                
                print(f"  答案: {result['answer'][:50]}...")
                print(f"  置信度: {result['confidence']:.2f}")
                
            except Exception as e:
                print(f"  失败: {str(e)[:50]}")
                results.append({
                    "sample_index": i,
                    "query": item.get("query", ""),
                    "error": str(e),
                    "answer": "处理失败",
                    "ground_truth": item.get("answer", "")
                })
        
        experiment_summary = {
            "experiment_info": {
                "dataset_path": dataset_path,
                "dataset_type": dataset_type,
                "total_samples": len(data),
                "successful_samples": successful,
                "timestamp": timestamp
            },
            "cost_summary": debate_system.get_cost_summary(),
            "system_stats": debate_system.get_system_stats(),
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n实验完成")
        print(f"成功: {successful}/{len(data)}")
        print(f"总成本: ${experiment_summary['cost_summary']['total_cost']:.6f}")
        print(f"结果: {output_file}")
        
        return output_file

async def main():
    parser = argparse.ArgumentParser(description="多智能体辩论系统")
    
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--single", type=str,
                       help="单次辩论模式")
    parser.add_argument("--batch", type=str,
                       help="批量实验模式")
    parser.add_argument("--dataset", type=str, default="default",
                       choices=["gsm8k", "hotpotqa", "math", "mmlu", "default"],
                       help="数据集类型")
    parser.add_argument("--max-samples", type=int,
                       help="最大样本数")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    if not (args.single or args.batch):
        parser.print_help()
        return
    
    try:
        runner = ExperimentRunner(args.config)
        
        if args.single:
            result = await runner.run_single_debate(args.single)
            
            output_file = os.path.join(args.output_dir, f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
        elif args.batch:
            await runner.run_batch_experiments(
                dataset_path=args.batch,
                dataset_type=args.dataset,
                max_samples=args.max_samples,
                output_dir=args.output_dir
            )
        
    except KeyboardInterrupt:
        print("\n用户中断")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)