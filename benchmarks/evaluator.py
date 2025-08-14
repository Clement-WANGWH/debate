import json
import numpy as np
import datetime
import asyncio
import re
import inspect
from math import isclose
from typing import Any, Callable, List, Tuple
from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections import OrderedDict

# Try to import optional dependencies
try:
    import regex
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self, data_path: str, save_path: str):
        self.data_path = data_path
        self.save_path = save_path

    def load_data(self):
        data = []
        with open(self.data_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                data.append(entry)
        return data
    
    @abstractmethod
    def calculate_score(self, prediction, expected_output) -> float:
        pass
    
    def prepare_data_item(self, data_item):
        """Prepare data item for evaluation. Override in subclasses if needed."""
        return data_item
    
    async def eval(self, mad, samples: int = 1000, seed: int = 42, args = None):
        data = self.load_data()
        np.random.seed(seed)
        data = np.random.choice(data, samples)

        results = await self._eval(mad, data)

        avg_score = np.mean([result["score"] for result in results])
        
        # Get cost information
        cost_summary = {}
        if hasattr(mad, 'get_cost_summary'):
            cost_summary = mad.get_cost_summary()
        
        file_results = {
            "avg_score": avg_score, 
            "cost_summary": cost_summary,
            "results": results
        }
        
        # Add cost info to file name
        task_name = self.__class__.__name__.replace("Eval", "").lower()
        file_name = f"{self.save_path}/{task_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        self.save_results(file_results, file_name)

        print(f"Average score: {avg_score}")
        
        # Print cost summary
        # if cost_summary:
        #     print(f"Total cost: {cost_summary.get('all_cost', 'N/A')}")
        #     print(f"Total call count: {cost_summary.get('all_call_count', 'N/A')}")
        #     print(f"Total tokens: {cost_summary.get('all_tokens', 'N/A')}")
        
        return avg_score, cost_summary

    async def _eval(self, mad, data, max_concurrent: int = 50):
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        # Create progress bar
        pbar = tqdm(total=len(data), desc="Evaluating", position=0, leave=True)
        
        async def sem_eval(mad, data_item):
            async with semaphore:
                try:
                    prepared_data = self.prepare_data_item(data_item)
                    
                    # Support both MATH benchmark and other benchmarks
                    # MATH expects result to be tuple of (output, cost)
                    # Others expect result to be dict with "answer" key
                    try:
                        result = await mad(prepared_data)
                        
                        # Handle different return types
                        if isinstance(result, tuple) and len(result) == 2:
                            # For MATH benchmark compatibility
                            output, cost = result
                            result = {
                                "answer": output,
                                "cost": cost
                            }
                        
                        # Calculate score based on evaluator type
                        answer_key = "answer"
                        truth_key = "answer"
                                                
                        score = self.calculate_score(result[answer_key], data_item[truth_key])
                        
                        # Determine query information
                        query = ""
                        if "init_query" in data_item:
                            query = data_item["init_query"]
                        elif "problem" in data_item:
                            query = data_item["problem"]
                        else:
                            query = data_item.get("query", "Unknown query")
                            
                        # Get ground truth
                        ground_truth = data_item.get(truth_key, "Unknown ground truth")
                        
                        # Create ordered result dict with specified key order
                        ordered_result = OrderedDict([
                            ("query", query),
                            ("ground_truth", ground_truth),
                            ("answer", result[answer_key]),
                            ("score", score)
                        ])
                        
                        # Add history if exists
                        if "history" in result:
                            ordered_result["history"] = result["history"]
                        
                        # Add remaining keys from the result
                        for key, value in result.items():
                            if key not in ordered_result and key != answer_key:
                                ordered_result[key] = value
                        
                    except Exception as e:
                        # In case of error during evaluation
                        ordered_result = OrderedDict([
                            ("query", data_item.get("query", data_item.get("problem", "Unknown query"))),
                            ("ground_truth", data_item.get("answer", data_item.get("solution", "Unknown truth"))),
                            ("answer", f"Error: {str(e)}"),
                            ("score", 0.0),
                            ("error", str(e))
                        ])
                    
                    return ordered_result
                finally:
                    # Update progress bar regardless of success or failure
                    pbar.update(1)
        
        # Create all tasks
        tasks = [sem_eval(mad, data_item) for data_item in data]
        
        # Process tasks in chunks to maintain semaphore limits while showing progress
        for i in range(0, len(tasks), max_concurrent):
            chunk = tasks[i:i+max_concurrent]
            chunk_results = await asyncio.gather(*chunk)
            results.extend(chunk_results)
        
        # Close progress bar
        pbar.close()
        
        return results
    
    def save_results(self, results, file_name):
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)

class MMLUEval(BaseEvaluator):
    def calculate_score(self, prediction: str, expected_output: str) -> float:
        """支持全/半角字母、大小写不敏感的匹配"""
        # 提取预测中的第一个有效选项（兼容全角字符）
        pred_match = re.search(r'([Ａ-ＤA-Da-d])', prediction)
        if not pred_match:
            return 0.0
        
        # 统一转换为半角大写
        pred_char = self.normalize_char(pred_match.group(1))
        correct_char = self.normalize_char(expected_output.strip())
        
        return 1.0 if pred_char == correct_char else 0.0
    
    def normalize_char(self, char: str) -> str:
        """统一字符格式：全角转半角，小写转大写"""
        # 全角字母转换（Ａ→A）
        if len(char) == 1 and 0xFF01 <= ord(char) <= 0xFF5E:
            char = chr(ord(char) - 0xFEE0)
        
        # 处理中文冒号等特殊字符
        char = char.replace("：", ":")  # 中文冒号转英文
        
        # 提取第一个有效字符并大写
        return char.strip().lower()[0] if char else ""
    
    def prepare_data_item(self, data_item):
        """标准化MMLU数据格式"""
        # 创建标准化问题格式
        data_item["query"] = f"**ONLY include option letter in your final answer**\n{data_item['query']}"
        # 确保ID格式统一
        data_item["id"] = data_item.get("id", "unknown-id")
        return data_item
    
class MATHEval(BaseEvaluator):
    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, prediction: str, expected_output: str) -> float:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = prediction.strip()

        if self.math_equal(predicted_answer, expected_answer):
            return 1.0
        else:
            # Log mismatch for debugging
            # if hasattr(self, 'log_mismatch'):
            #     self.log_mismatch(
            #         "Question", 
            #         expected_output, 
            #         prediction,
            #         f"Expected: {expected_answer}, Predicted: {predicted_answer}",
            #         extract_answer_code=inspect.getsource(self.extract_model_answer),
            #     )
            return 0.0

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        if SYMPY_AVAILABLE:
            try:
                return self.symbolic_equal(prediction, reference)
            except:
                pass

        return False

    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        if not regex or not SYMPY_AVAILABLE:
            return None
            
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        if not SYMPY_AVAILABLE:
            return False
            
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False
    
    def log_mismatch(self, input_text, expected_output, output, extracted_output, extract_answer_code=None):
        """Log mismatched answers for debugging"""
        print("\n==== MATH MISMATCH ====")
        print(f"Input: {input_text[:100]}...")
        print(f"Expected: {expected_output[:100]}...")
        print(f"Output: {output[:100]}...")
        print(f"Extraction: {extracted_output}")
        print("=======================\n")


from utils import extract_number

class GSM8KEval(BaseEvaluator):
    def calculate_score(self, prediction, expected_output) -> float:
        expected_output = extract_number(expected_output)
        prediction = extract_number(prediction)

        if prediction is None:
            return 0.0
        
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0
    
    def prepare_data_item(self, data_item):
        # GSM8K doesn't need special preparation
        return data_item


from utils import f1_score

class HotpotQAEval(BaseEvaluator):
    def calculate_score(self, prediction, expected_output) -> float:
        return f1_score(prediction, expected_output)
    
    def prepare_data_item(self, data_item):
        paragraphs = [item[1] for item in data_item["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        
        # Store original query for reporting
        data_item["init_query"] = data_item["query"]
        # Add context to query
        data_item["query"] = f"{data_item['query']}\n\nRevelant Context:{context_str}"
        # Ensure ID field is standardized
        data_item["id"] = data_item.get("_id", data_item.get("id", "unknown"))
        
        return data_item