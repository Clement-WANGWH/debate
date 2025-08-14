import aiohttp
import json
import asyncio
from typing import Dict, Any, Optional
import time

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "qwen-flash")
        self.temperature = config.get("temperature", 0.7)
        self.api_key = config.get("api_key", None)
        self.base_url = config.get("base_url", "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
        self.top_p = config.get("top_p", 0.9)
        self.max_tokens = config.get("max_tokens", 2000)

class ModelPricing:
    PRICES = {
        "qwen-turbo": {"input": 0.0003, "output": 0.003},
        "qwen-plus": {"input": 0.0008, "output": 0.002},
        "qwen-max": {"input": 0.0024, "output": 0.0096},
        "qwen-long": {"input": 0.0005, "output": 0.002},
        "qwen-flash": {"input": 0.00015, "output": 0.0015},
        "default": {"input": 0.00015, "output": 0.0015}
    }
    
    @classmethod
    def get_price(cls, model_name: str, token_type: str) -> float:
        """获取特定模型和token类型的价格"""
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]
        
        # 尝试部分匹配
        for key in cls.PRICES:
            if key in model_name.lower():
                return cls.PRICES[key][token_type]
        
        return cls.PRICES["default"][token_type]

class TokenUsageTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
        self.call_count = 0
    
    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "timestamp": time.time()
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.call_count += 1
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self) -> Dict[str, Any]:
        """获取使用摘要"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "average_cost_per_call": self.total_cost / max(1, self.call_count)
        }

class AsyncQwenLLM:
    
    def __init__(self, config: LLMConfig, system_msg: Optional[str] = None):
        self.config = config
        self.system_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        
        if not self.config.api_key:
            raise ValueError("API key is required for Qwen API")
    
    async def __call__(self, prompt: str, **kwargs) -> str:
        # 构建消息
        messages = []
        if self.system_msg:
            messages.append({"role": "system", "content": self.system_msg})
        messages.append({"role": "user", "content": prompt})
        
        # 构建请求数据
        data = {
            "model": self.config.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "result_format": "message"
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.config.base_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Qwen API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    # 解析响应
                    if "output" not in result:
                        raise Exception(f"Invalid response format: {result}")
                    
                    output = result["output"]
                    content = output.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    usage = output.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    
                    if input_tokens == 0 and output_tokens == 0:
                        input_tokens = self._estimate_tokens(prompt)
                        output_tokens = self._estimate_tokens(content)
                    
                    usage_record = self.usage_tracker.add_usage(
                        self.config.model, input_tokens, output_tokens
                    )
                    
                    print(f"[{self.config.model}] Token usage: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total")
                    print(f"Cost: ${usage_record['total_cost']:.6f}")
                    
                    return content
                    
            except asyncio.TimeoutError:
                raise Exception("Request timeout")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """估算token数量（粗略估计）"""
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len(text) - chinese_chars
        english_words = english_chars / 5
        
        estimated_tokens = int(chinese_chars * 1.5 + english_words * 1.3)
        return max(1, estimated_tokens)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """获取使用摘要"""
        return self.usage_tracker.get_summary()

AsyncLLM = AsyncQwenLLM