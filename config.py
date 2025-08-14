import yaml
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class QwenConfig:
    """Qwen模型配置"""
    model: str = "qwen-flash"
    api_key: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2000
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

@dataclass
class DebateConfig:
    """辩论系统配置"""
    num_agents: int = 3
    max_rounds: int = 3
    enable_prune: bool = True
    enable_reflect: bool = True
    strict_mode: bool = True
    num_injections: int = 0
    enable_cache: bool = True
    cache_size: int = 10000

@dataclass
class SystemConfig:
    """系统总配置"""
    qwen: QwenConfig
    debate: DebateConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """从字典创建配置"""
        qwen_config = QwenConfig(**config_dict.get('qwen', {}))
        debate_config = DebateConfig(**config_dict.get('debate', {}))
        return cls(qwen=qwen_config, debate=debate_config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SystemConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'SystemConfig':
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'qwen': {
                'model': self.qwen.model,
                'api_key': self.qwen.api_key,
                'temperature': self.qwen.temperature,
                'top_p': self.qwen.top_p,
                'max_tokens': self.qwen.max_tokens,
                'base_url': self.qwen.base_url
            },
            'debate': {
                'num_agents': self.debate.num_agents,
                'max_rounds': self.debate.max_rounds,
                'enable_prune': self.debate.enable_prune,
                'enable_reflect': self.debate.enable_reflect,
                'strict_mode': self.debate.strict_mode,
                'num_injections': self.debate.num_injections,
                'enable_cache': self.debate.enable_cache,
                'cache_size': self.debate.cache_size
            }
        }
    
    def save_yaml(self, yaml_path: str):
        """保存为YAML文件"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def save_json(self, json_path: str):
        """保存为JSON文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """加载配置"""
        if os.path.exists(self.config_path):
            try:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return SystemConfig.from_yaml(self.config_path)
                elif self.config_path.endswith('.json'):
                    return SystemConfig.from_json(self.config_path)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
            except Exception as e:
                print(f"Error loading config from {self.config_path}: {e}")
                print("Using default configuration...")
                return self._create_default_config()
        else:
            print(f"Config file {self.config_path} not found, creating default...")
            config = self._create_default_config()
            self._save_default_config(config)
            return config
    
    def _create_default_config(self) -> SystemConfig:
        """创建默认配置"""
        # 尝试从环境变量获取API密钥
        api_key = os.getenv('QWEN_API_KEY', '')
        
        qwen_config = QwenConfig(
            model="qwen-flash",
            api_key=api_key,
            temperature=0.7,
            top_p=0.9,
            max_tokens=2000
        )
        
        debate_config = DebateConfig(
            num_agents=3,
            max_rounds=3,
            enable_prune=True,
            enable_reflect=True,
            strict_mode=True,
            enable_cache=True
        )
        
        return SystemConfig(qwen=qwen_config, debate=debate_config)
    
    def _save_default_config(self, config: SystemConfig):
        """保存默认配置"""
        try:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                config.save_yaml(self.config_path)
            elif self.config_path.endswith('.json'):
                config.save_json(self.config_path)
            print(f"Default config saved to {self.config_path}")
        except Exception as e:
            print(f"Failed to save default config: {e}")
    
    def get_qwen_config(self) -> Dict[str, Any]:
        """获取Qwen API配置"""
        return {
            'model': self.config.qwen.model,
            'api_key': self.config.qwen.api_key,
            'temperature': self.config.qwen.temperature,
            'top_p': self.config.qwen.top_p,
            'max_tokens': self.config.qwen.max_tokens,
            'base_url': self.config.qwen.base_url
        }
    
    def get_debate_config(self) -> Dict[str, Any]:
        """获取辩论系统配置"""
        return {
            'num_agents': self.config.debate.num_agents,
            'max_rounds': self.config.debate.max_rounds,
            'enable_prune': self.config.debate.enable_prune,
            'enable_reflect': self.config.debate.enable_reflect,
            'strict_mode': self.config.debate.strict_mode,
            'num_injections': self.config.debate.num_injections,
            'enable_cache': self.config.debate.enable_cache,
            'cache_size': self.config.debate.cache_size
        }
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config.qwen, key):
                setattr(self.config.qwen, key, value)
            elif hasattr(self.config.debate, key):
                setattr(self.config.debate, key, value)
            else:
                print(f"Unknown config key: {key}")
    
    def save_config(self):
        """保存当前配置"""
        self._save_default_config(self.config)

# 全局配置管理器实例
config_manager = ConfigManager()

def get_config() -> SystemConfig:
    """获取全局配置"""
    return config_manager.config

def get_qwen_config() -> Dict[str, Any]:
    """获取Qwen配置"""
    return config_manager.get_qwen_config()

def get_debate_config() -> Dict[str, Any]:
    """获取辩论配置"""
    return config_manager.get_debate_config()

# 创建默认配置文件模板
DEFAULT_CONFIG_YAML = """
# Qwen API配置
qwen:
  model: "qwen-flash"  # 可选: qwen-turbo, qwen-plus, qwen-max, qwen-flash等
  api_key: ""  # 你的Qwen API密钥
  temperature: 0.7
  top_p: 0.9
  max_tokens: 2000
  base_url: "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 辩论系统配置  
debate:
  num_agents: 3          # 智能体数量
  max_rounds: 3          # 最大辩论轮数
  enable_prune: true     # 启用剪枝机制
  enable_reflect: true   # 启用反思机制
  strict_mode: true      # 严格模式（不确定答案视为错误）
  num_injections: 0      # 错误注入数量（仅用于测试）
  enable_cache: true     # 启用缓存机制
  cache_size: 10000      # 缓存大小
"""

if __name__ == "__main__":
    # 测试配置管理器
    print("Testing ConfigManager...")
    
    # 创建配置管理器
    cm = ConfigManager("test_config.yaml")
    
    # 打印配置
    print("Qwen Config:", cm.get_qwen_config())
    print("Debate Config:", cm.get_debate_config())
    
    # 更新配置
    cm.update_config(temperature=0.8, num_agents=5)
    
    # 保存配置
    cm.save_config()
    
    print("Configuration test completed!")