import asyncio
import json
import numpy as np
import hashlib
import time
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque, Counter
from pathlib import Path
from agent import create_debate_agents, BidirectionalDebateAgent
from config import get_debate_config, get_qwen_config
from utils import if_reach_consensus, dataset_2_process_fn

@dataclass
class DebateRound:
    """辩论轮次记录"""
    round_num: int
    responses: List[Dict[str, Any]]
    consensus: bool
    consensus_answer: Optional[str]
    avg_confidence: float
    timestamp: float

class ConsensusStrategy:
    """渐进式共识策略"""
    
    def __init__(self, initial_threshold: float = 0.7, decay_rate: float = 0.05):
        self.initial_threshold = initial_threshold
        self.decay_rate = decay_rate
        self.min_threshold = 0.4
    
    def get_threshold(self, round_num: int) -> float:
        threshold = self.initial_threshold - (round_num * self.decay_rate)
        return max(threshold, self.min_threshold)
    
    def evaluate_consensus(self, answers: List[str], confidences: List[float], 
                          round_num: int) -> Tuple[bool, Optional[str], float]:
        if not answers:
            return False, None, 0.0
        
        threshold = self.get_threshold(round_num)
        answer_weights = {}
        
        for answer, confidence in zip(answers, confidences):
            if answer not in answer_weights:
                answer_weights[answer] = 0
            answer_weights[answer] += confidence
        
        total_weight = sum(answer_weights.values())
        if total_weight == 0:
            return False, None, 0.0
        
        for answer, weight in answer_weights.items():
            if weight / total_weight >= threshold:
                consensus_confidence = weight / total_weight
                return True, answer, consensus_confidence
        
        best_answer = max(answer_weights.items(), key=lambda x: x[1])
        return False, best_answer[0], best_answer[1] / total_weight

class AdaptiveRoundController:
    """自适应轮次控制器"""
    
    def __init__(self, min_rounds: int = 2, max_rounds: int = 5):
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds
        self.confidence_history = []
        self.convergence_rate = 0
    
    def should_continue(self, round_num: int, avg_confidence: float, 
                       consensus: bool) -> bool:
        if round_num < self.min_rounds:
            return True
        
        if round_num >= self.max_rounds:
            return False
        
        if consensus and avg_confidence > 0.8:
            return False
        
        self.confidence_history.append(avg_confidence)
        
        if len(self.confidence_history) >= 2:
            recent_improvement = self.confidence_history[-1] - self.confidence_history[-2]
            if recent_improvement < 0.01:
                return False
        
        return True
    
    def update_convergence(self, rounds: List[DebateRound]):
        if len(rounds) < 2:
            self.convergence_rate = 0
            return
        
        improvements = []
        for i in range(1, len(rounds)):
            improvement = rounds[i].avg_confidence - rounds[i-1].avg_confidence
            improvements.append(improvement)
        
        self.convergence_rate = np.mean(improvements) if improvements else 0

class ProgressiveMultiagentDebate:
    """渐进式多智能体辩论系统"""
    
    def __init__(self, dataset: str = "default", config: Optional[Dict] = None):
        if config is None:
            config = get_debate_config()
        
        self.dataset = dataset
        self.num_agents = config.get("num_agents", 3)
        self.enable_cache = config.get("enable_cache", True)
        
        self.agents = self._create_agents()
        self.consensus_strategy = ConsensusStrategy()
        self.round_controller = AdaptiveRoundController(
            min_rounds=2,
            max_rounds=config.get("max_rounds", 5)
        )
        
        self.process_fn = dataset_2_process_fn(dataset)
        self.debate_history = []
        
        self.memory_dir = Path("memory")
        self.memory_dir.mkdir(exist_ok=True)
        self.session_file = self.memory_dir / f"session_{int(time.time())}.json"
    
    def _create_agents(self) -> List[BidirectionalDebateAgent]:
        diverse_profiles = [
            "你擅长系统性分析，注重逻辑推理的完整性",
            "你擅长创造性思维，善于发现非常规解决方案", 
            "你擅长批判性思考，专注于验证和纠错",
            "你擅长综合分析，善于整合不同观点",
            "你擅长细节把控，注重计算准确性"
        ]
        
        agents = create_debate_agents(
            num_agents=self.num_agents,
            profiles=diverse_profiles[:self.num_agents],
            enable_prune=True,
            enable_reflect=True
        )
        
        return agents
    
    def _introduce_structural_diversity(self, round_num: int, agent_idx: int) -> str:
        """引入结构化多样性"""
        templates = [
            "让我们用不同的角度分析这个问题",
            "从另一个维度思考",
            "换个思路来看",
            "重新审视这个问题",
            "让我们深入探讨"
        ]
        
        noise_level = 0.1 + (round_num * 0.05)
        if random.random() < noise_level:
            return random.choice(templates) + "\n"
        return ""
    
    async def progressive_debate_round(self, query: str, question_id: str, 
                                      round_num: int, prev_round: Optional[DebateRound]) -> DebateRound:
        """执行一轮渐进式辩论"""
        print(f"\n轮次 {round_num + 1}")
        
        round_responses = []
        round_confidences = []
        round_answers = []
        
        for idx, agent in enumerate(self.agents):
            diversity_prefix = self._introduce_structural_diversity(round_num, idx)
            modified_query = diversity_prefix + query
            
            if prev_round:
                context = []
                for i, resp in enumerate(prev_round.responses):
                    if i == idx:
                        context.append(f"你上轮的分析:\n答案: {resp.get('answer', '')}\n置信度: {resp.get('confidence', 0):.2f}")
                    else:
                        context.append(f"{self.agents[i].name}的分析:\n答案: {resp.get('answer', '')}\n置信度: {resp.get('confidence', 0):.2f}")
            else:
                context = None
            
            try:
                result, metadata = await agent(modified_query, question_id, context)
                round_responses.append(result)
                round_confidences.append(result.get('confidence', 0.5))
                round_answers.append(result.get('answer', ''))
                
                print(f"  {agent.name}: {result.get('answer', '')[:50]}... (置信度: {result.get('confidence', 0):.2f})")
                
            except Exception as e:
                print(f"  {agent.name}: 错误 - {str(e)[:50]}")
                round_responses.append({"answer": "", "confidence": 0})
                round_confidences.append(0)
                round_answers.append("")
        
        avg_confidence = np.mean(round_confidences) if round_confidences else 0
        
        consensus, consensus_answer, consensus_conf = self.consensus_strategy.evaluate_consensus(
            round_answers, round_confidences, round_num
        )
        
        debate_round = DebateRound(
            round_num=round_num,
            responses=round_responses,
            consensus=consensus,
            consensus_answer=consensus_answer,
            avg_confidence=avg_confidence,
            timestamp=time.time()
        )
        
        return debate_round
    
    async def debate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行完整辩论流程"""
        query = data["query"]
        question_id = data.get("id", str(hash(query)))
        
        print(f"开始辩论: {query[:100]}...")
        
        rounds = []
        final_answer = None
        final_confidence = 0
        
        for round_num in range(self.round_controller.max_rounds):
            prev_round = rounds[-1] if rounds else None
            
            debate_round = await self.progressive_debate_round(
                query, question_id, round_num, prev_round
            )
            
            rounds.append(debate_round)
            
            print(f"  共识: {'是' if debate_round.consensus else '否'}, 平均置信度: {debate_round.avg_confidence:.2f}")
            
            if debate_round.consensus:
                final_answer = debate_round.consensus_answer
                final_confidence = debate_round.avg_confidence
            
            if not self.round_controller.should_continue(
                round_num, debate_round.avg_confidence, debate_round.consensus
            ):
                print(f"  提前结束辩论")
                break
        
        self.round_controller.update_convergence(rounds)
        
        if final_answer is None:
            best_round = max(rounds, key=lambda r: r.avg_confidence)
            if best_round.consensus_answer:
                final_answer = best_round.consensus_answer
                final_confidence = best_round.avg_confidence
            else:
                all_answers = []
                all_confidences = []
                for r in rounds:
                    for resp in r.responses:
                        if resp.get('answer'):
                            all_answers.append(resp['answer'])
                            all_confidences.append(resp.get('confidence', 0))
                
                if all_answers:
                    weighted_answers = Counter()
                    for ans, conf in zip(all_answers, all_confidences):
                        weighted_answers[ans] += conf
                    final_answer = weighted_answers.most_common(1)[0][0]
                    final_confidence = np.mean(all_confidences)
                else:
                    final_answer = "无法确定答案"
                    final_confidence = 0
        
        result = {
            "query": query,
            "answer": final_answer,
            "confidence": final_confidence,
            "rounds": len(rounds),
            "consensus_reached": any(r.consensus for r in rounds),
            "convergence_rate": self.round_controller.convergence_rate,
            "debate_history": [
                {
                    "round": r.round_num,
                    "consensus": r.consensus,
                    "avg_confidence": r.avg_confidence,
                    "answer": r.consensus_answer
                }
                for r in rounds
            ]
        }
        
        self._save_session(result)
        
        print(f"\n最终答案: {final_answer}")
        print(f"置信度: {final_confidence:.2f}")
        print(f"轮次: {len(rounds)}")
        
        return result
    
    def _save_session(self, result: Dict[str, Any]):
        """保存会话记录"""
        try:
            sessions = []
            if self.session_file.exists():
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    sessions = json.load(f)
            
            sessions.append({
                "timestamp": time.time(),
                "result": result
            })
            
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存会话失败: {e}")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        agent_summaries = [agent.get_usage_summary() for agent in self.agents]
        
        total_cost = sum(summary["total_cost"] for summary in agent_summaries)
        total_tokens = sum(summary["total_tokens"] for summary in agent_summaries)
        total_calls = sum(summary["call_count"] for summary in agent_summaries)
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "average_cost_per_call": total_cost / max(1, total_calls)
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        agent_states = {}
        for i, agent in enumerate(self.agents):
            agent_states[f"agent_{i}"] = agent.export_state()
        
        return {
            "total_debates": len(self.debate_history),
            "convergence_rate": self.round_controller.convergence_rate,
            "agent_states": agent_states
        }

async def quick_debate(question: str, dataset: str = "default", **kwargs) -> Dict[str, Any]:
    debate_system = ProgressiveMultiagentDebate(dataset=dataset, config=kwargs)
    data = {"query": question, "id": str(hash(question))}
    result = await debate_system.debate(data)
    return result

if __name__ == "__main__":
    async def test():
        question = "如果一个水池有3个进水管和2个出水管，每个进水管每小时注入10升水，每个出水管每小时排出8升水，水池初始有100升水，问5小时后水池有多少水？"
        result = await quick_debate(question)
        print(f"\n测试完成: {result['answer']}")
    
    asyncio.run(test())