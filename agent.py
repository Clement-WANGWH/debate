from async_llm import AsyncQwenLLM, LLMConfig
from config import get_qwen_config
from utils import extract_with_label, normalize_answer
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import re
import json
import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

class CoT(BaseModel):
    hypothesis: str = Field(default="", description="初始假设")
    forward_reasoning: str = Field(default="", description="前向推理过程")
    backward_verification: str = Field(default="", description="反向验证过程")
    answer: str = Field(default="", description="最终答案")
    confidence: float = Field(default=0.5, description="置信度0-1")

class TaskTracker:
    """任务追踪器"""
    def __init__(self, base_dir: str = "memory"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.task_file = self.base_dir / "tasks.md"
        self.current_tasks = []
        
    def update_tasks(self, question_id: str, status: str, details: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task_entry = f"- [{status}] {question_id}: {details} ({timestamp})\n"
        
        with open(self.task_file, "a", encoding="utf-8") as f:
            f.write(task_entry)
        
        self.current_tasks.append({
            "id": question_id,
            "status": status,
            "details": details,
            "timestamp": timestamp
        })
        
        if len(self.current_tasks) > 10:
            self.current_tasks = self.current_tasks[-10:]
    
    def get_context(self) -> str:
        if not self.current_tasks:
            return ""
        
        context = "当前任务状态:\n"
        for task in self.current_tasks[-5:]:
            context += f"- {task['status']}: {task['details'][:50]}...\n"
        return context

class MemoryManager:
    """外部记忆管理器"""
    def __init__(self, base_dir: str = "memory"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.observations_dir = self.base_dir / "observations"
        self.observations_dir.mkdir(exist_ok=True)
        self.failures_dir = self.base_dir / "failures"
        self.failures_dir.mkdir(exist_ok=True)
        
    def store_observation(self, content: str, question_id: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        filename = f"{question_id}_{content_hash}.txt"
        filepath = self.observations_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return str(filepath)
    
    def store_failure(self, error: str, context: Dict[str, Any], question_id: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{question_id}_error_{timestamp}.json"
        filepath = self.failures_dir / filename
        
        failure_data = {
            "error": error,
            "context": context,
            "timestamp": timestamp
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(failure_data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def load_observation(self, filepath: str) -> Optional[str]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return None
    
    def compress_context(self, context: List[str], max_length: int = 1000) -> Tuple[str, List[str]]:
        """可逆上下文压缩"""
        if not context:
            return "", []
        
        compressed = []
        references = []
        
        for item in context:
            if len(item) > max_length:
                ref = self.store_observation(item, f"compress_{hashlib.md5(item.encode()).hexdigest()[:8]}")
                compressed.append(f"[长内容已存储: {ref}]")
                references.append(ref)
            else:
                compressed.append(item)
        
        return "\n".join(compressed), references

def get_field_names(model_class):
    return model_class.model_fields.keys()

def look_up_description(model_class, field_name):
    return model_class.model_fields[field_name].description

def check_format_and_extract(text: str, format_class: BaseModel) -> Dict[str, Any]:
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    
    found_fields = {match[0]: match[1].strip() for match in matches}
    
    result = {}
    for field_name in get_field_names(format_class):
        if field_name == "confidence":
            try:
                result[field_name] = float(found_fields.get(field_name, 0.5))
            except:
                result[field_name] = 0.5
        else:
            result[field_name] = found_fields.get(field_name, "")
    
    return result

def fill_with_xml_suffix(text: str, format_class: BaseModel) -> str:
    field_names = get_field_names(format_class)
    examples = []
    
    for field_name in field_names:
        description = look_up_description(format_class, field_name)
        examples.append(f"<{field_name}>{description}</{field_name}>")
    
    example_str = "\n".join(examples)
    text += f"\n\n{example_str}\n"
    return text

class BidirectionalDebateAgent:
    """双向辩论智能体"""
    
    def __init__(self, name: str, profile: Optional[str] = None,
                 enable_prune: bool = True, enable_reflect: bool = True,
                 strict_mode: bool = True, llm_config: Optional[Dict] = None):
        self.name = name
        self.profile = profile
        self.enable_prune = enable_prune
        self.enable_reflect = enable_reflect
        self.strict_mode = strict_mode
        
        self.memory_manager = MemoryManager()
        self.task_tracker = TaskTracker()
        self.error_history = []
        self.reasoning_chain = []
        
        if llm_config is None:
            llm_config = get_qwen_config()
        
        self.llm_config = LLMConfig(llm_config)
        self.llm = AsyncQwenLLM(self.llm_config)
        self.cot_format = CoT
    
    async def _get_response_with_format(self, prompt: str, format_class: BaseModel = None) -> Dict[str, Any]:
        if format_class is None:
            format_class = self.cot_format
        
        role_prompt = f"**你是{self.name}**"
        if self.profile:
            role_prompt += f"\n{self.profile}"
        
        task_context = self.task_tracker.get_context()
        if task_context:
            role_prompt += f"\n\n{task_context}"
        
        formatted_prompt = role_prompt + '\n\n' + fill_with_xml_suffix(prompt, format_class)
        
        try:
            response = await self.llm(formatted_prompt)
            return check_format_and_extract(response, format_class)
        except Exception as e:
            error_ref = self.memory_manager.store_failure(
                str(e), 
                {"prompt": prompt[:500]}, 
                "response_error"
            )
            self.error_history.append(error_ref)
            raise
    
    async def forward_reasoning(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """前向推理"""
        prompt = f"""问题: {query}

请进行前向推理：
1. 提出初始假设
2. 逐步推导
3. 得出初步答案"""
        
        if context and len(context) > 0:
            compressed_context, refs = self.memory_manager.compress_context(context)
            prompt += f"\n\n参考上下文:\n{compressed_context}"
        
        response = await self._get_response_with_format(prompt)
        self.reasoning_chain.append(("forward", response))
        return response
    
    async def backward_verification(self, query: str, forward_result: Dict[str, Any]) -> Dict[str, Any]:
        """反向验证"""
        prompt = f"""问题: {query}
初步答案: {forward_result.get('answer', '')}
推理过程: {forward_result.get('forward_reasoning', '')}

请进行反向验证：
1. 从答案出发，反推是否能得到题目条件
2. 检查推理链中的每个步骤
3. 评估答案的可靠性"""
        
        response = await self._get_response_with_format(prompt)
        self.reasoning_chain.append(("backward", response))
        return response
    
    async def synthesize_result(self, forward: Dict[str, Any], backward: Dict[str, Any]) -> Dict[str, Any]:
        """综合前向和反向结果"""
        confidence = (forward.get('confidence', 0.5) + backward.get('confidence', 0.5)) / 2
        
        if abs(forward.get('confidence', 0.5) - backward.get('confidence', 0.5)) > 0.3:
            confidence *= 0.8
        
        return {
            "hypothesis": forward.get('hypothesis', ''),
            "forward_reasoning": forward.get('forward_reasoning', ''),
            "backward_verification": backward.get('backward_verification', ''),
            "answer": forward.get('answer', ''),
            "confidence": confidence
        }
    
    async def generate_response(self, query: str, question_id: str,
                              context: Optional[List[str]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """生成双向推理响应"""
        self.task_tracker.update_tasks(question_id, "开始", query[:50])
        
        try:
            forward = await self.forward_reasoning(query, context)
            self.task_tracker.update_tasks(question_id, "前向完成", f"答案: {forward.get('answer', '')[:30]}")
            
            backward = await self.backward_verification(query, forward)
            self.task_tracker.update_tasks(question_id, "反向完成", f"置信度: {backward.get('confidence', 0)}")
            
            result = await self.synthesize_result(forward, backward)
            
            if len(self.reasoning_chain) > 100:
                old_chain = self.reasoning_chain[:50]
                for direction, content in old_chain:
                    self.memory_manager.store_observation(
                        json.dumps(content, ensure_ascii=False),
                        f"chain_{question_id}"
                    )
                self.reasoning_chain = self.reasoning_chain[50:]
            
            metadata = {
                "forward_confidence": forward.get('confidence', 0.5),
                "backward_confidence": backward.get('confidence', 0.5),
                "final_confidence": result['confidence'],
                "reasoning_steps": len(self.reasoning_chain),
                "errors": len(self.error_history)
            }
            
            self.task_tracker.update_tasks(question_id, "完成", f"最终置信度: {result['confidence']:.2f}")
            
            return result, metadata
            
        except Exception as e:
            self.task_tracker.update_tasks(question_id, "失败", str(e)[:50])
            raise
    
    async def __call__(self, query: str, question_id: str,
                      context: Optional[List[str]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return await self.generate_response(query, question_id, context)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        return self.llm.get_usage_summary()
    
    def export_state(self) -> Dict[str, Any]:
        return {
            "reasoning_chain_length": len(self.reasoning_chain),
            "error_count": len(self.error_history),
            "task_history": self.task_tracker.current_tasks
        }

def create_agent(name: str, profile: Optional[str] = None, **kwargs) -> BidirectionalDebateAgent:
    return BidirectionalDebateAgent(name=name, profile=profile, **kwargs)

def create_debate_agents(num_agents: int, profiles: Optional[List[str]] = None, **kwargs) -> List[BidirectionalDebateAgent]:
    from utils import get_random_unique_names
    
    names = get_random_unique_names(num_agents)
    agents = []
    
    for i in range(num_agents):
        name = names[i]
        profile = profiles[i] if profiles and i < len(profiles) else None
        agent = create_agent(name, profile, **kwargs)
        agents.append(agent)
    
    return agents