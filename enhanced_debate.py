import asyncio
import json
import numpy as np
import hashlib
import time
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque, Counter
from agent import create_debate_agents, DebateAgent
from config import get_debate_config, get_qwen_config
from utils import if_reach_consensus, dataset_2_process_fn, get_random_unique_names

@dataclass
class KVCache:
    """键值缓存类，实现SnapKV的缓存压缩机制"""
    key_vectors: np.ndarray = field(default_factory=lambda: np.array([]))
    value_vectors: np.ndarray = field(default_factory=lambda: np.array([]))
    importance_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    content_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    max_cache_size: int = 1024
    
    def add_entry(self, key_vector: np.ndarray, value_vector: np.ndarray, importance: float):
        """添加缓存条目"""
        if self.key_vectors.size == 0:
            self.key_vectors = key_vector.reshape(1, -1)
            self.value_vectors = value_vector.reshape(1, -1)
            self.importance_scores = np.array([importance])
        else:
            self.key_vectors = np.vstack([self.key_vectors, key_vector])
            self.value_vectors = np.vstack([self.value_vectors, value_vector])
            self.importance_scores = np.append(self.importance_scores, importance)
        
        if len(self.importance_scores) > self.max_cache_size:
            self.compress()
    
    def compress(self, threshold: float = 0.5):
        """基于重要性分数压缩缓存"""
        if len(self.importance_scores) <= self.max_cache_size:
            return
        
        important_indices = np.where(self.importance_scores > threshold)[0]
        
        if len(important_indices) > self.max_cache_size:
            top_k_indices = np.argsort(self.importance_scores)[-self.max_cache_size:]
            important_indices = top_k_indices
        
        if len(important_indices) < self.max_cache_size // 2:
            top_half_indices = np.argsort(self.importance_scores)[-self.max_cache_size//2:]
            important_indices = top_half_indices
        
        if len(important_indices) > 0:
            self.key_vectors = self.key_vectors[important_indices]
            self.value_vectors = self.value_vectors[important_indices]
            self.importance_scores = self.importance_scores[important_indices]
    
    def update_access(self):
        self.access_count += 1
        self.timestamp = time.time()

class PositionIndependentCache:
    
    def __init__(self, cache_size: int = 10000, ttl: float = 3600):
        self.cache: Dict[str, KVCache] = {}
        self.cache_size = cache_size
        self.ttl = ttl
        self.access_history = deque(maxlen=cache_size)
    
    def _compute_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_expired(self, kv_cache: KVCache) -> bool:
        return time.time() - kv_cache.timestamp > self.ttl
    
    def _evict_lru(self):
        if len(self.cache) < self.cache_size:
            return
        min_access_key = min(self.cache.keys(), 
                           key=lambda k: (self.cache[k].access_count, self.cache[k].timestamp))
        del self.cache[min_access_key]
    
    def _cleanup_expired(self):
        expired_keys = [key for key, cache in self.cache.items() if self._is_expired(cache)]
        for key in expired_keys:
            del self.cache[key]
    
    def store(self, content: str, response: str, metadata: Dict[str, Any]):
        cache_key = self._compute_hash(content)
        
        self._cleanup_expired()
        
        self._evict_lru()
        
        key_vector = self._content_to_vector(content)
        value_vector = self._content_to_vector(response)
        importance = self._calculate_importance(content, response, metadata)
        
        kv_cache = KVCache(content_hash=cache_key)
        kv_cache.add_entry(key_vector, value_vector, importance)
        
        self.cache[cache_key] = kv_cache
        self.access_history.append(cache_key)
    
    def retrieve(self, content: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        cache_key = self._compute_hash(content)
        
        if cache_key in self.cache:
            kv_cache = self.cache[cache_key]
            
            if self._is_expired(kv_cache):
                del self.cache[cache_key]
                return None
            kv_cache.update_access()
            
            return f"[缓存响应] 基于相似问题的推理结果", {"cached": True, "cache_key": cache_key}
        
        return None
    
    def _content_to_vector(self, content: str) -> np.ndarray:
        hash_val = int(hashlib.md5(content.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**31))
        return np.random.randn(128)
    
    def _calculate_importance(self, content: str, response: str, metadata: Dict[str, Any]) -> float:
        """计算重要性分数"""
        importance = 0.5
        
        importance += min(0.3, len(content) / 1000)
        
        if metadata.get("confidence"):
            importance += metadata["confidence"] * 0.3
        
        complex_keywords = ["计算", "分析", "推理", "证明", "解释"]
        keyword_count = sum(1 for kw in complex_keywords if kw in content)
        importance += min(0.2, keyword_count * 0.05)
        
        return min(1.0, importance)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_access = sum(cache.access_count for cache in self.cache.values())
        avg_importance = np.mean([np.mean(cache.importance_scores) 
                                for cache in self.cache.values() 
                                if cache.importance_scores.size > 0])
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "total_accesses": total_access,
            "average_importance": avg_importance if not np.isnan(avg_importance) else 0.0,
            "hit_rate": total_access / max(1, len(self.access_history))
        }

class EnhancedMultiagentDebate:
    """增强的多智能体辩论系统"""
    
    def __init__(self, dataset: str = "default", config: Optional[Dict] = None):
        # 加载
        if config is None:
            config = get_debate_config()
        
        self.dataset = dataset
        self.num_agents = config.get("num_agents", 3)
        self.max_rounds = config.get("max_rounds", 3)
        self.enable_prune = config.get("enable_prune", True)
        self.enable_reflect = config.get("enable_reflect", True)
        self.strict_mode = config.get("strict_mode", True)
        self.num_injections = config.get("num_injections", 0)
        self.enable_cache = config.get("enable_cache", True)
        
        # 初始化
        self.agents = self._create_agents()
        if self.enable_cache:
            self.global_cache = PositionIndependentCache(cache_size=config.get("cache_size", 10000))
        else:
            self.global_cache = None
        self.process_fn = dataset_2_process_fn(dataset)
        
        self.wrong_samples = self._load_wrong_samples() if self.num_injections > 0 else None
        
        self.stats = {
            "total_debates": 0,
            "consensus_reached": 0,
            "cache_hits": 0,
            "total_rounds": 0
        }
    
    def _create_agents(self) -> List[DebateAgent]:
        """创建智能体"""
        profiles = [
            "你是一个善于逻辑分析的AI助手，喜欢从理论角度思考问题",
            "你是一个注重实践的AI助手，善于从实际应用角度分析问题", 
            "你是一个批判性思维很强的AI助手，善于发现问题和漏洞",
            "你是一个创新思维很强的AI助手，喜欢从多个角度思考问题",
            "你是一个谨慎细致的AI助手，注重细节和准确性"
        ]
        
        agents = create_debate_agents(
            num_agents=self.num_agents,
            profiles=profiles[:self.num_agents],
            enable_prune=self.enable_prune,
            enable_reflect=self.enable_reflect,
            strict_mode=self.strict_mode
        )
        
        return agents
    
    def _load_wrong_samples(self) -> Optional[Dict]:
        """加载错误样本（用于错误注入测试）"""
        try:
            wrong_samples_path = f"data/{self.dataset}/{self.dataset}_train_new_wrong_samples_woxml.json"
            with open(wrong_samples_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Wrong samples file not found: {wrong_samples_path}")
            return None
    
    def _sample_injections(self, question_id: str, k: int) -> List[Dict]:
        """采样错误注入样本"""
        if not self.wrong_samples or question_id not in self.wrong_samples:
            return []
        
        samples = random.sample(self.wrong_samples[question_id], min(k, len(self.wrong_samples[question_id])))
        return [{"think": sample["trajectory"], "answer": sample["answer"]} for sample in samples]
    
    async def debate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行辩论过程"""
        query = data["query"]
        question_id = data.get("id", str(hash(query)))
        
        cached_result = None
        if self.enable_cache and self.global_cache:
            cached_result = self.global_cache.retrieve(query)
            if cached_result:
                self.stats["cache_hits"] += 1
                return {
                    "query": query,
                    "answer": cached_result[0],
                    "cached": True,
                    "cache_metadata": cached_result[1]
                }
        
        history = []
        mask_history = []
        reflection_history = []
        
        # 多轮辩论
        for round_num in range(self.max_rounds):
            print(f"\n=== 第{round_num + 1}轮辩论 ===")
            
            round_responses = []
            round_masks = []
            round_reflections = []
            answer_list = []
            
            injected_indices = []
            if round_num == 0 and self.num_injections > 0:
                injected_indices = random.sample(range(self.num_agents), 
                                               min(self.num_injections, self.num_agents))
                injected_samples = self._sample_injections(question_id, len(injected_indices))
            
            for idx, agent in enumerate(self.agents):
                if round_num == 0:
                    # 第一轮：独立思考或注入错误
                    if idx in injected_indices and injected_samples:
                        result = injected_samples.pop()
                        metadata = {"injected": True, "masks": [], "reflection": ""}
                    else:
                        result, metadata = await agent(query, question_id)
                else:
                    # 后续轮：基于上下文辩论
                    context = [f"你自己上一轮的解决方案:\n{history[round_num-1][i]}" 
                             if i == idx 
                             else f"来自{self.agents[i].name}的解决方案:\n{history[round_num-1][i]}"
                             for i in range(self.num_agents)]
                    
                    result, metadata = await agent(query, question_id, context)
                
                round_responses.append(result)
                round_masks.append(metadata.get("masks", []))
                round_reflections.append(metadata.get("reflection", ""))
                answer_list.append(result.get("answer", ""))
                
                print(f"{agent.name}: {result.get('answer', '')[:100]}...")
            
            history.append(round_responses)
            mask_history.append(round_masks)
            reflection_history.append(round_reflections)
            consensus_reached, common_answer = if_reach_consensus(answer_list, self.process_fn)
            if consensus_reached:
                print(f"\n达成共识！答案: {common_answer}")
                break
        
        # 最终答案处理
        if not consensus_reached:
            final_answer = self._vote_for_final_answer(answer_list)
            print(f"\n未达成共识，投票结果: {final_answer}")
        else:
            final_answer = common_answer
            self.stats["consensus_reached"] += 1
        
        result = {
            "query": query,
            "answer": final_answer,
            "consensus": consensus_reached,
            "rounds": len(history),
            "history": history,
            "mask_history": mask_history,
            "reflection_history": reflection_history,
            "injected_indices": injected_indices,
            "cached": False
        }
        
        if self.enable_cache and self.global_cache:
            self.global_cache.store(query, final_answer, {
                "consensus": consensus_reached,
                "rounds": len(history),
                "confidence": 1.0 if consensus_reached else 0.7
            })
        
        self.stats["total_debates"] += 1
        self.stats["total_rounds"] += len(history)
        
        return result
    
    def _vote_for_final_answer(self, answers: List[str]) -> str:
        """投票决定最终答案"""
        if not answers:
            return "无法确定答案"
        
        answer_counter = Counter(answers)
        most_common = answer_counter.most_common(1)[0]
        
        return most_common[0]
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """获取成本摘要"""
        agent_summaries = [agent.get_usage_summary() for agent in self.agents]
        
        total_cost = sum(summary["total_cost"] for summary in agent_summaries)
        total_tokens = sum(summary["total_tokens"] for summary in agent_summaries)
        total_calls = sum(summary["call_count"] for summary in agent_summaries)
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "average_cost_per_call": total_cost / max(1, total_calls),
            "agent_details": agent_summaries
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = self.stats.copy()
        
        if self.enable_cache and self.global_cache:
            stats["cache_stats"] = self.global_cache.get_stats()
        
        memory_stats = {}
        for i, agent in enumerate(self.agents):
            memory_stats[f"agent_{i}_{agent.name}"] = agent.get_memory_summary()
        stats["memory_stats"] = memory_stats
        
        return stats
    
    def export_knowledge(self) -> Dict[str, Any]:
        knowledge = {
            "agent_memories": {},
            "cache_data": {},
            "system_stats": self.get_system_stats()
        }
        
        for i, agent in enumerate(self.agents):
            knowledge["agent_memories"][f"agent_{i}_{agent.name}"] = agent.export_memory()
        
        if self.enable_cache and self.global_cache:
            knowledge["cache_data"] = {
                "cache_keys": list(self.global_cache.cache.keys()),
                "cache_stats": self.global_cache.get_stats()
            }
        
        return knowledge
    
    def clear_all_memory(self):
        for agent in self.agents:
            agent.clear_memory()
        
        if self.enable_cache and self.global_cache:
            self.global_cache.cache.clear()
        
        self.stats = {
            "total_debates": 0,
            "consensus_reached": 0,
            "cache_hits": 0,
            "total_rounds": 0
        }

async def quick_debate(question: str, dataset: str = "default", **kwargs) -> Dict[str, Any]:
    """快速辩论函数"""
    debate_system = EnhancedMultiagentDebate(dataset=dataset, config=kwargs)
    data = {"query": question, "id": str(hash(question))}
    result = await debate_system.debate(data)
    
    print(f"\n=== 辩论摘要 ===")
    print(f"问题: {question}")
    print(f"答案: {result['answer']}")
    print(f"轮数: {result['rounds']}")
    print(f"共识: {'是' if result['consensus'] else '否'}")
    print(f"缓存: {'是' if result['cached'] else '否'}")
    
    return result

if __name__ == "__main__":
    async def test_enhanced_debate():
        """测试增强辩论系统"""
        print("测试增强多智能体辩论系统...")
        
        debate_system = EnhancedMultiagentDebate(
            dataset="gsm8k",
            config={
                "num_agents": 3,
                "max_rounds": 2,
                "enable_cache": True,
                "cache_size": 1000
            }
        )
        
        questions = [
            "小明有24个苹果，给了小红1/3，又给了小华1/4，请问小明还剩多少个苹果？",
            "一个班级有40名学生，其中女生占60%，请问男生有多少人？"
        ]
        
        for i, question in enumerate(questions):
            print(f"\n{'='*60}")
            print(f"测试问题 {i+1}: {question}")
            print('='*60)
            
            data = {"query": question, "id": f"test_{i+1}"}
            result = await debate_system.debate(data)
            
            print(f"\n最终结果:")
            print(f"答案: {result['answer']}")
            print(f"轮数: {result['rounds']}")
            print(f"共识: {'达成' if result['consensus'] else '未达成'}")
        
        print(f"\n系统统计: {debate_system.get_system_stats()}")
        print(f"成本摘要: {debate_system.get_cost_summary()}")
    