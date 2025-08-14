import json
import asyncio
import sys
from async_llm import AsyncLLM, LLMConfig

async def evaluate_results(json_file_path):
    """
    Evaluate answers in a JSON file using LLM as a judge.
    
    Args:
        json_file_path (str): Path to the JSON file containing results
        api_key (str): API key for OpenAI
    
    Returns:
        float: Average score across all evaluated results
    """
    # Load the JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return 0
    
    if "results" not in data or not data["results"]:
        print("Error: JSON file must contain a 'results' array with at least one item")
        return 0
    
    import yaml
    def load_configs_from_yaml(file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    configs = load_configs_from_yaml("configs.yaml")
    config = LLMConfig(configs["llm_configs"]["dpwm"])
    
    # System message for evaluation
    system_message = """
    You are an expert evaluator judging the correctness of answers.
    Your task is to compare a provided answer against the ground truth and determine if it's correct.
    
    Rules for evaluation:
    1. The answer should contain the key information found in the ground truth
    2. The answer can be phrased differently but must convey the same meaning
    3. Partial answers should receive partial scores
    4. Completely wrong answers get a score of 0
    
    Please provide a score 0 or 1, where:
    - 1: Correct answer
    - 0: Incorrect answer
    
    After giving your score, provide a 1-2 sentence explanation.
    Format your response exactly like this:
    SCORE: [your score as a decimal number]
    EXPLANATION: [your brief explanation]
    """
    
    llm = AsyncLLM(config, system_message)
    
    # Process each result
    total_score = 0
    total_items = len(data["results"])
    results_with_scores = []
    
    print(f"Evaluating {total_items} results...")
    
    for i, result in enumerate(data["results"]):
        query = result.get("query", "")
        ground_truth = result.get("ground_truth", "")
        answer = result.get("answer", "")
        
        # Skip if missing essential fields
        if not ground_truth or not answer:
            print(f"Skipping result {i+1}: Missing ground_truth or answer")
            continue
        
        prompt = f"""
        Query: {query}
        
        Ground Truth: {ground_truth}
        
        Answer to Evaluate: {answer}
        
        Evaluate the answer against the ground truth and provide your score and brief explanation.
        """
        
        print(f"\nEvaluating result {i+1}/{total_items}...")
        
        try:
            # Get evaluation from LLM
            evaluation = await llm(prompt)
            
            # Parse score from response
            score_line = [line for line in evaluation.split('\n') if line.startswith("SCORE:")][0]
            score = float(score_line.split("SCORE:")[1].strip())
            
            # Store result with score
            result_with_score = result.copy()
            result_with_score["score"] = score
            result_with_score["evaluation"] = evaluation
            results_with_scores.append(result_with_score)
            
            total_score += score
            print(f"Result {i+1} - Score: {score}")
            
        except Exception as e:
            print(f"Error evaluating result {i+1}: {e}")
    
    # Calculate average score
    average_score = total_score / len(results_with_scores) if results_with_scores else 0
    
    # Save results with evaluations
    output_path = json_file_path.replace('.json', '_evaluated.json')
    output_data = {
        "average_score": average_score,
        "results": results_with_scores,
        "usage_summary": llm.get_usage_summary()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation complete. Average score: {average_score:.4f}")
    print(f"Detailed results saved to: {output_path}")
    
    # Print usage summary
    usage = llm.get_usage_summary()
    print(f"\nToken Usage Summary:")
    print(f"Total tokens: {usage['total_tokens']} (Input: {usage['total_input_tokens']}, Output: {usage['total_output_tokens']})")
    print(f"Total cost: ${usage['total_cost']:.6f}")
    
    return average_score

if __name__ == "__main__":
    
    json_file_path = "results/hotpotqa_1000_sc6_2025-04-03_15-43-53.json"
    
    # Run the evaluation
    asyncio.run(evaluate_results(json_file_path))