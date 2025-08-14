from async_llm import AsyncQwenLLM, LLMConfig
from config import get_qwen_config
from utils import extract_with_label, normalize_answer
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import re
import json
from typing import List, Dict, Any, Optional, Tuple

# 提示词模板
COT_PROMPT = """
现在你正在与其他智能体进行辩论讨论。
请逐步思考并解决这个问题。
"""

DEBATE_PROMPT = """
{question}

现在你正在与其他智能体进行辩论讨论。
请逐步思考并解决这个问题。

请仔细参考以下其他智能体的观点作为额外建议：
{context}
"""

PRUNE_PROMPT = """
问题: {question}

解决方案: {solution}

请分析这个解决方案，指出其中的错误或缺陷（如果存在的话）。

** 你的回答必须以以下格式结束: <label>YES</label> 或 <label>NO</label> 或 <label>NOT SURE</label> **
- 如果解决方案完全正确，返回YES
- 如果解决方案的任何部分不正确，返回NO  
- 如果你不确定，返回NOT SURE
"""

REFLECT_PROMPT = """
问题: {question}

错误解决方案: {solutions}

这些是错误的解决方案。请简洁地告诉自己如何避免这些错误。
"""

REFLECTION_HEAD = """
考虑以下反思来避免错误的解决方案：
"""

class CoT(BaseModel):
    """思维链响应格式"""
    think: str = Field(default="", description="逐步思考过程")
    answer: str = Field(default="", description="最终答案")

class PruneResult(BaseModel):
    """剪枝结果格式"""
    analysis: str = Field(default="", description="分析解决方案并指出错误或缺陷")
    label: str = Field(default="", description="如果解决方案完全正确返回YES，任何部分不正确返回NO，不确定返回NOT SURE")

def get_field_names(model_class):
    """获取模型字段名"""
    return model_class.model_fields.keys()

def look_up_description(model_class, field_name):
    """获取字段描述"""
    return model_class.model_fields[field_name].description

def check_format_and_extract(text: str, format_class: BaseModel) -> Dict[str, str]:
    """检查XML格式并提取内容"""
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    
    found_fields = {match[0]: match[1].strip() for match in matches}
    
    for field_name in get_field_names(format_class):
        if field_name not in found_fields or not found_fields[field_name]:
            raise ValueError(f"Field '{field_name}' is missing or empty.")
    
    return found_fields

def fill_with_xml_suffix(text: str, format_class: BaseModel) -> str:
    field_names = get_field_names(format_class)
    examples = []
    
    for field_name in field_names:
        description = look_up_description(format_class, field_name)
        examples.append(f"<{field_name}>{description}</{field_name}>")
    
    example_str = "\n".join(examples)
    text += f"""

{example_str}
"""
    return text

class DebateAgent:
    
    def __init__(self, name: str, profile: Optional[str] = None, 
                 enable_prune: bool = True, enable_reflect: bool = True, 
                 strict_mode: bool = True, llm_config: Optional[Dict] = None):
        """
        初始化辩论智能体
        
        Args:
            name: 智能体名字
            profile: 智能体人设
            enable_prune: 是否启用剪枝
            enable_reflect: 是否启用反思
            strict_mode: 严格模式
            llm_config: LLM配置，如果为None则使用全局配置
        """
        self.name = name
        self.profile = profile
        self.enable_prune = enable_prune
        self.enable_reflect = enable_reflect
        self.strict_mode = strict_mode
        
        # 内存系统：记录错误答案
        self.memory: Dict[str, List[str]] = {}
        
        # 初始化LLM
        if llm_config is None:
            llm_config = get_qwen_config()
        
        self.llm_config = LLMConfig(llm_config)
        self.llm = AsyncQwenLLM(self.llm_config)
        
        # 默认响应格式
        self.cot_format = CoT
    
    async def _get_response(self, prompt: str) -> str:
        """获取LLM响应"""
        role_prompt = f"**你是{self.name}**"
        if self.profile:
            role_prompt += f"\n你的人设: {self.profile}"
        
        full_prompt = role_prompt + '\n\n' + prompt
        response = await self.llm(full_prompt)
        return response
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(Exception))
    async def _get_response_with_format(self, prompt: str, format_class: BaseModel = None) -> Dict[str, str]:
        """获取格式化的LLM响应"""
        if format_class is None:
            format_class = self.cot_format
        
        role_prompt = f"**你是{self.name}**"
        if self.profile:
            role_prompt += f"\n你的人设: {self.profile}"
        
        formatted_prompt = role_prompt + '\n\n' + fill_with_xml_suffix(prompt, format_class)
        response = await self.llm(formatted_prompt)
        
        return check_format_and_extract(response, format_class)
    
    async def _prune_solutions(self, query: str, solutions: List[str]) -> List[bool]:
        """剪枝：评估解决方案质量"""
        if not self.enable_prune or not solutions:
            return [True] * len(solutions)
        
        masks = []
        for solution in solutions:
            prompt = PRUNE_PROMPT.format(question=query, solution=solution)
            
            try:
                response = await self._get_response(prompt)
                label = extract_with_label(response, "label")
                
                if label is None:
                    label_match = re.search(r'<label>(.*?)</label>', response, re.IGNORECASE)
                    if label_match:
                        label = label_match.group(1)
                
                if label:
                    normalized_label = normalize_answer(label)
                    if normalized_label == "yes":
                        masks.append(True)
                    elif normalized_label == "no":
                        masks.append(False)
                    else:  # "not sure"
                        masks.append(not self.strict_mode)
                else:
                    masks.append(not self.strict_mode)
                    
            except Exception as e:
                print(f"Error in pruning: {e}")
                masks.append(not self.strict_mode)
        
        return masks
    
    async def _reflect_on_failures(self, query: str, failed_solutions: List[str]) -> List[str]:
        """反思失败的解决方案"""
        if not self.enable_reflect or not failed_solutions:
            return []
        
        failed_answers = []
        for solution in failed_solutions:
            answer_match = re.search(r"'answer':\s*'([^']*)'", solution)
            if answer_match:
                failed_answers.append(answer_match.group(1))
            else:
                lines = solution.split('\n')
                for line in lines:
                    if '答案' in line or '结果' in line:
                        failed_answers.append(line.strip())
                        break
        
        return list(set(failed_answers))  # 去重
    
    def _update_memory(self, question_id: str, failed_answers: List[str]):
        if question_id in self.memory:
            self.memory[question_id].extend(failed_answers)
            self.memory[question_id] = list(set(self.memory[question_id]))
        else:
            self.memory[question_id] = failed_answers.copy()
    
    def _get_memory_context(self, question_id: str) -> str:
        if question_id in self.memory and self.memory[question_id]:
            return "之前的错误答案: " + ", ".join(self.memory[question_id])
        return ""
    
    async def generate_response(self, query: str, question_id: str, 
                              context: Optional[List[str]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        生成响应
        
        Args:
            query: 问题
            question_id: 问题ID
            context: 其他智能体的解决方案上下文
            
        Returns:
            (response, metadata)
        """
        masks = []
        reflection_summary = ""
        
        if context and len(context) > 0:
            masks = await self._prune_solutions(query, context)
            
            if False in masks and self.enable_reflect:
                failed_solutions = [sol for sol, mask in zip(context, masks) if not mask]
                failed_answers = await self._reflect_on_failures(query, failed_solutions)
                
                if failed_answers:
                    self._update_memory(question_id, failed_answers)
                    reflection_summary = f"学习到的错误答案: {', '.join(failed_answers)}"
            
            valid_context = [sol for sol, mask in zip(context, masks) if mask]
            
            if valid_context:
                prompt = DEBATE_PROMPT.format(
                    question=query, 
                    context='\n'.join([f"观点 {i+1}: {ctx}" for i, ctx in enumerate(valid_context)])
                )
            else:
                prompt = f"{query}\n{COT_PROMPT}"
            
            memory_context = self._get_memory_context(question_id)
            if memory_context:
                prompt += f"\n\n{memory_context}"
                
        else:
            prompt = f"{query}\n{COT_PROMPT}"
            
            memory_context = self._get_memory_context(question_id)
            if memory_context:
                prompt += f"\n\n{memory_context}"
        
        try:
            response = await self._get_response_with_format(prompt)
        except Exception as e:
            print(f"Error generating response for {self.name}: {e}")
            response = {"think": "生成响应时出现错误", "answer": "无法确定"}
        
        metadata = {
            "masks": masks,
            "reflection": reflection_summary,
            "memory_size": len(self.memory.get(question_id, [])),
            "valid_context_count": len([m for m in masks if m]) if masks else 0
        }
        
        return response, metadata
    
    async def __call__(self, query: str, question_id: str, 
                      context: Optional[List[str]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """调用智能体进行推理"""
        return await self.generate_response(query, question_id, context)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """获取使用摘要"""
        return self.llm.get_usage_summary()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        total_memories = sum(len(memories) for memories in self.memory.values())
        return {
            "total_questions": len(self.memory),
            "total_memories": total_memories,
            "memory_details": {qid: len(memories) for qid, memories in self.memory.items()}
        }
    
    def clear_memory(self, question_id: Optional[str] = None):
        if question_id:
            if question_id in self.memory:
                del self.memory[question_id]
        else:
            self.memory.clear()
    
    def export_memory(self) -> Dict[str, List[str]]:
        return self.memory.copy()
    
    def import_memory(self, memory_data: Dict[str, List[str]]):
        self.memory.update(memory_data)

def create_agent(name: str, profile: Optional[str] = None, **kwargs) -> DebateAgent:
    """创建智能体的工厂函数"""
    return DebateAgent(name=name, profile=profile, **kwargs)

def create_debate_agents(num_agents: int, profiles: Optional[List[str]] = None, **kwargs) -> List[DebateAgent]:
    """创建多个辩论智能体"""
    from utils import get_random_unique_names
    
    names = get_random_unique_names(num_agents)
    agents = []
    
    for i in range(num_agents):
        name = names[i]
        profile = profiles[i] if profiles and i < len(profiles) else None
        agent = create_agent(name, profile, **kwargs)
        agents.append(agent)
    
    return agents

if __name__ == "__main__":
    # 测试智能体
    import asyncio
    
    async def test_agent():
        """测试智能体功能"""
        print("Testing DebateAgent...")
        
        agent = create_agent("测试智能体", "你是一个善于逻辑推理的AI助手")
        
        query = "小明有10个苹果，给了小红3个，又给了小华2个，请问小明还剩多少个苹果？"
        question_id = "test_001"
        
        print("第一轮思考:")
        response1, meta1 = await agent(query, question_id)
        print(f"思考过程: {response1.get('think', '')}")
        print(f"答案: {response1.get('answer', '')}")
        print(f"元数据: {meta1}")
        
        wrong_context = [
            "我认为答案是7个苹果，因为10-3=7",
            "答案应该是5个苹果，因为10-3-2=5"
        ]
        
        print("\n第二轮辩论:")
        response2, meta2 = await agent(query, question_id, wrong_context)
        print(f"思考过程: {response2.get('think', '')}")
        print(f"答案: {response2.get('answer', '')}")
        print(f"元数据: {meta2}")
        
        print(f"\n使用摘要: {agent.get_usage_summary()}")
        print(f"记忆摘要: {agent.get_memory_summary()}")
    