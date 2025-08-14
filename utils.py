import re
import random
import hashlib
from collections import Counter
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

# ===============================
# 答案提取和处理函数
# ===============================

def extract_with_label(text: str, pattern: str = "answer") -> Optional[str]:
    """
    从文本中提取带标签的内容
    
    Args:
        text: 输入文本
        pattern: 标签模式
        
    Returns:
        提取的内容，如果没有找到则返回None
    """
    pattern = rf'<{pattern}>(.*?)</{pattern}>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_number(text: str) -> Optional[float]:
    """
    从文本中提取最后一个数字
    
    Args:
        text: 输入文本
        
    Returns:
        提取的数字，如果没有找到则返回None
    """
    # 改进的数字提取正则，支持更多格式
    patterns = [
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?",
        r"[-+]?\d+(?:\.\d+)?",
        r"[-+]?\d+",
    ]
    
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, str(text))
        all_matches.extend(matches)
    
    if all_matches:
        last_number = all_matches[-1].replace(",", "")
        try:
            return float(last_number)
        except ValueError:
            return None
    
    return None

def normalize_answer(s: str) -> str:
    """
    标准化答案用于比较
    
    Args:
        s: 输入答案
        
    Returns:
        标准化后的答案
    """
    if not isinstance(s, str):
        s = str(s)
    
    # 移除选项标记 (A), [A], A), A.
    s = re.sub(r'[\(\[\{]([A-Za-z])[\)\]\}]|([A-Za-z])[\.:\)]', r'\1\2', s)
    
    # 移除多余空格和换行
    s = re.sub(r'\s+', ' ', s)
    
    # 转为小写并去除首尾空格
    return s.lower().strip()

def normalize_char(char: str) -> str:
    """
    统一字符格式：全角转半角，提取第一个有效字符
    
    Args:
        char: 输入字符
        
    Returns:
        标准化后的字符
    """
    if not char:
        return ""
    
    if len(char) == 1 and 0xFF01 <= ord(char) <= 0xFF5E:
        char = chr(ord(char) - 0xFEE0)
    
    char = char.replace("：", ":")
    
    cleaned = re.sub(r'[^\w]', '', char)
    return cleaned.lower()[0] if cleaned else ""

def extract_final_answer(text: str, patterns: Optional[List[str]] = None) -> str:
    """
    从文本中提取最终答案
    
    Args:
        text: 输入文本
        patterns: 自定义提取模式
        
    Returns:
        提取的答案
    """
    if patterns is None:
        patterns = [
            r"最终答案[是为]?[:：]?\s*(.+?)(?:\n|$|。)",
            r"答案[是为]?[:：]?\s*(.+?)(?:\n|$|。)",
            r"结果[是为]?[:：]?\s*(.+?)(?:\n|$|。)",
            r"所以[答案结果]?[是为]?[:：]?\s*(.+?)(?:\n|$|。)",
            r"因此[答案结果]?[是为]?[:：]?\s*(.+?)(?:\n|$|。)",
        ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('#') and len(line) > 1:
            return line
    
    return text.strip()

# ===============================
# 共识判断算法
# ===============================

def if_reach_consensus(answers: List[str], process_fn: Optional[Callable] = None, 
                      threshold: float = 0.5, similarity_threshold: float = 0.8) -> Tuple[bool, Optional[str]]:
    """
    改进的共识判断算法
    
    Args:
        answers: 答案列表
        process_fn: 答案处理函数
        threshold: 共识阈值（需要达到的比例）
        similarity_threshold: 相似度阈值
        
    Returns:
        (是否达成共识, 共识答案)
    """
    if not answers:
        return False, None
    
    # 处理答案
    if process_fn:
        processed_answers = [process_fn(answer) for answer in answers]
        processed_answers = [ans for ans in processed_answers if ans is not None]
    else:
        processed_answers = answers
    
    if not processed_answers:
        return False, None
    
    # 计算答案频率
    answer_counts = Counter(processed_answers)
    total_answers = len(processed_answers)
    
    # 检查是否有答案达到阈值
    for answer, count in answer_counts.items():
        if count / total_answers >= threshold:
            return True, answer
    
    # 如果没有直接匹配，尝试相似度匹配（用于文本答案）
    if isinstance(processed_answers[0], str):
        return _consensus_by_similarity(processed_answers, threshold, similarity_threshold)
    
    # 返回最频繁的答案
    most_common = answer_counts.most_common(1)[0]
    return False, most_common[0]

def _consensus_by_similarity(answers: List[str], threshold: float, similarity_threshold: float) -> Tuple[bool, Optional[str]]:
    """
    基于相似度的共识判断
    
    Args:
        answers: 字符串答案列表
        threshold: 共识阈值
        similarity_threshold: 相似度阈值
        
    Returns:
        (是否达成共识, 共识答案)
    """
    if not answers:
        return False, None
    
    # 计算答案间的相似度
    similarity_groups = []
    
    for answer in answers:
        # 找到相似的组
        found_group = False
        for group in similarity_groups:
            if any(_text_similarity(answer, existing) >= similarity_threshold for existing in group):
                group.append(answer)
                found_group = True
                break
        
        # 如果没有找到相似组，创建新组
        if not found_group:
            similarity_groups.append([answer])
    
    # 检查是否有组达到共识阈值
    total_answers = len(answers)
    for group in similarity_groups:
        if len(group) / total_answers >= threshold:
            # 返回组内最常见的答案
            group_counter = Counter(group)
            return True, group_counter.most_common(1)[0][0]
    
    # 返回最大组的代表答案
    largest_group = max(similarity_groups, key=len)
    return False, Counter(largest_group).most_common(1)[0][0]

def _text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度
    
    Args:
        text1, text2: 要比较的文本
        
    Returns:
        相似度分数 (0-1)
    """
    # 简化的相似度计算
    text1 = normalize_answer(text1)
    text2 = normalize_answer(text2)
    
    if text1 == text2:
        return 1.0
    
    # 计算Jaccard相似度
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

# ===============================
# 评估指标函数
# ===============================

def f1_score(prediction: str, ground_truth: str) -> float:
    """
    计算F1分数
    
    Args:
        prediction: 预测答案
        ground_truth: 标准答案
        
    Returns:
        F1分数 (0-1)
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens and not truth_tokens:
        return 1.0
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def exact_match(prediction: str, ground_truth: str) -> bool:
    """
    精确匹配
    
    Args:
        prediction: 预测答案
        ground_truth: 标准答案
        
    Returns:
        是否精确匹配
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def contains_match(prediction: str, ground_truth: str) -> bool:
    """
    包含匹配
    
    Args:
        prediction: 预测答案
        ground_truth: 标准答案
        
    Returns:
        预测答案是否包含标准答案
    """
    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(ground_truth)
    return truth_norm in pred_norm

# ===============================
# 数据集处理函数
# ===============================

def dataset_2_process_fn(dataset: str) -> Optional[Callable]:
    """
    根据数据集返回对应的处理函数
    
    Args:
        dataset: 数据集名称
        
    Returns:
        处理函数
    """
    dataset_map = {
        "gsm8k": lambda x: extract_number(x),
        "hotpotqa": lambda x: normalize_answer(x),
        "math": lambda x: x.strip(),
        "mmlu": lambda x: normalize_char(x),
        "aqua": lambda x: normalize_char(x),
        "commonsense_qa": lambda x: normalize_char(x),
        "arc": lambda x: normalize_char(x),
        "default": lambda x: normalize_answer(x)
    }
    
    return dataset_map.get(dataset.lower(), dataset_map["default"])

def load_dataset_sample(dataset_path: str, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    加载数据集样本
    
    Args:
        dataset_path: 数据集路径
        sample_size: 样本大小
        
    Returns:
        数据样本列表
    """
    import json
    import os
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    data = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        if dataset_path.endswith('.jsonl'):
            for line in f:
                if sample_size and len(data) >= sample_size:
                    break
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    continue
        else:
            json_data = json.load(f)
            if isinstance(json_data, list):
                data = json_data
            elif isinstance(json_data, dict) and 'data' in json_data:
                data = json_data['data']
    
    if sample_size and len(data) > sample_size:
        data = random.sample(data, sample_size)
    
    return data

# ===============================
# 名字生成函数
# ===============================

def create_name_library() -> List[str]:
    """
    创建名字库
    
    Returns:
        名字列表
    """
    names = [
        # 中文名字
        "小明", "小红", "小华", "小李", "小王", "小张", "小刘", "小陈", "小杨", "小赵",
        "晓东", "晓西", "晓南", "晓北", "志强", "志明", "志华", "志勇", "志国", "志军",
        "美丽", "美华", "美玲", "美芳", "美娟", "美红", "美英", "美莲", "美凤", "美霞",
        
        # 英文名字
        "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery", "Blake",
        "Cameron", "Devon", "Ellis", "Finley", "Gray", "Harper", "Hayden", "Indigo", "Jamie", "Kelly",
        "Lane", "Max", "Noah", "Ocean", "Parker", "Quinn", "River", "Sage", "Tate", "Unity",
        "Vale", "West", "Xander", "Yael", "Zion", "Aria", "Brook", "Cedar", "Drew", "Echo",
        
        # 创意名字
        "智者", "思考者", "分析师", "探索者", "创新者", "批判家", "实践家", "理论家", "观察者", "推理专家",
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"
    ]
    
    return names

def get_random_unique_names(k: int) -> List[str]:
    """
    获取k个唯一的随机名字
    
    Args:
        k: 需要的名字数量
        
    Returns:
        唯一名字列表
        
    Raises:
        ValueError: 如果k大于可用名字数量
    """
    name_library = create_name_library()
    
    if k <= 0:
        return []
    
    if k > len(name_library):
        raise ValueError(f"无法从 {len(name_library)} 个名字中选择 {k} 个唯一名字")
    
    return random.sample(name_library, k)

# ===============================
# 性能和调试工具
# ===============================

def hash_content(content: str) -> str:
    """
    计算内容哈希
    
    Args:
        content: 输入内容
        
    Returns:
        哈希值
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本
    
    Args:
        text: 输入文本
        max_length: 最大长度
        suffix: 后缀
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m{secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h{minutes:.0f}m"

def safe_json_dump(obj: Any, file_path: str, encoding: str = 'utf-8', indent: int = 2):
    """
    安全的JSON保存
    
    Args:
        obj: 要保存的对象
        file_path: 文件路径
        encoding: 编码
        indent: 缩进
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    def json_serializer(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'isoformat'):  # datetime对象
            return obj.isoformat()
        else:
            return str(obj)
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent, default=json_serializer)
    except Exception as e:
        print(f"保存JSON文件失败 {file_path}: {e}")
        raise

# ===============================
# 测试和验证函数
# ===============================

def test_answer_extraction():
    """测试答案提取功能"""
    test_cases = [
        ("答案是42", "42"),
        ("The answer is 3.14", "3.14"),
        ("结果: 100个苹果", "100个苹果"),
        ("最终答案：选择A", "选择A"),
        ("<answer>Hello World</answer>", "Hello World"),
        ("因此答案为 -5.5", "-5.5"),
    ]
    
    print("测试答案提取:")
    for text, expected in test_cases:
        extracted = extract_final_answer(text)
        print(f"输入: {text}")
        print(f"期望: {expected}, 实际: {extracted}, 匹配: {expected in extracted}")
        print()

def test_number_extraction():
    """测试数字提取功能"""
    test_cases = [
        ("答案是42", 42.0),
        ("共有3.14个", 3.14),
        ("总计1,234,567.89元", 1234567.89),
        ("科学记数法: 1.23e-4", 1.23e-4),
        ("没有数字", None),
    ]
    
    print("测试数字提取:")
    for text, expected in test_cases:
        extracted = extract_number(text)
        print(f"输入: {text}")
        print(f"期望: {expected}, 实际: {extracted}, 匹配: {extracted == expected}")
        print()

def test_consensus():
    """测试共识算法"""
    test_cases = [
        (["A", "A", "B"], True, "A"),
        (["42", "42.0", "42"], True, "42"),
        (["红色", "蓝色", "绿色"], False, None),
        ([], False, None),
    ]
    
    print("测试共识算法:")
    for answers, expected_consensus, expected_answer in test_cases:
        consensus, answer = if_reach_consensus(answers)
        print(f"答案: {answers}")
        print(f"共识: {consensus} (期望: {expected_consensus})")
        print(f"答案: {answer} (期望: {expected_answer})")
        print()

if __name__ == "__main__":
    print("=== 工具函数测试 ===")
    test_answer_extraction()
    test_number_extraction()
    test_consensus()
    
    print("随机名字:", get_random_unique_names(5))
    
    print("测试完成!")