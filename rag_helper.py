import json
import re
import os
import time
import numpy as np
from google import genai
from tqdm import tqdm
from pathlib import Path
import faiss
from typing import List, Dict, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_ID = "gemma-3-1b-it"

# Paths
EMBEDDINGS_DIR = Path("test_precomputed_embeddings")
METHOD_DB_DIR = Path("method_databases")
OUTPUT_DIR = Path("results/gemma3_2b_single_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RAG Parameters
TOP_K_METHODS = 3
SIMILARITY_THRESHOLD = 0.4
RATE_LIMIT_DELAY = 1
MAX_RETRIES = 3

# Method text strategy
USE_FULL_METHOD = False  # False: [:200], True: tam method
MAX_METHOD_LENGTH = 200  # if USE_FULL_METHOD=False 

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def normalize_number(n_str):
    if not n_str:
        return None
    try:
        clean = re.sub(r'[^\d\.-]', '', str(n_str))
        return float(clean)
    except:
        return None


def extract_single_answer(text: str) -> str:
    if not text or len(text.strip()) == 0:
        return None
    
    text = text.strip()
    
    # 1. #### format
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # 2. The answer is X
    patterns = [
        r'[Tt]he (?:final )?answer is:?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Ff]inal answer:?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Tt]herefore,?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(',', '')
    
    # 3. \\boxed{}
    match = re.search(r'\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}', text)
    if match:
        return match.group(1).replace(',', '')
    
    # 4. last number in text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return None


# =============================================================================
# DATABASE LOADER
# =============================================================================
class VectorStoreClient:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        index_path = db_path / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        
        with open(db_path / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 1, threshold: float = 0.45):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold and idx < len(self.metadata):
                method_data = self.metadata[idx].copy()
                method_data['similarity_score'] = float(sim)
                results.append(method_data)
        
        return results


# =============================================================================
# RAG EVALUATOR WITH DETAILED METRICS
# =============================================================================
class RAGEvaluator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_id = MODEL_ID
        
        print("\n📥 Loading embeddings and metadata...")
        self.embeddings = np.load(EMBEDDINGS_DIR / "gsm8k_socratic_embeddings.npy")
        
        with open(EMBEDDINGS_DIR / "gsm8k_socratic_metadata.json", "r") as f:
            self.test_metadata = json.load(f)
        
        print(f"✅ Loaded {len(self.test_metadata)} test samples")
        
        self.databases = {}
        if METHOD_DB_DIR.exists():
            for db_dir in METHOD_DB_DIR.iterdir():
                if db_dir.is_dir() and (db_dir / "faiss_index.bin").exists():
                    self.databases[db_dir.name] = VectorStoreClient(db_dir)
        
        print(f"✅ Loaded {len(self.databases)} RAG databases")
    
    def _create_prompt(self, question: str, methods: List[Dict] = None) -> str:
        """Create prompt with method strategy"""
        
        if methods:
            # RAG prompt
            if USE_FULL_METHOD:
                # Full method text
                methods_text = "\n\n".join([
                    f"Reference Method {i+1}:\n{m['method_text']}"
                    for i, m in enumerate(methods)
                ])
            else:
                # Kısaltılmış method (kesilmiş cümleler olabilir)
                methods_text = "\n".join([
                    f"Reference: {m['method_text'][:MAX_METHOD_LENGTH]}..."
                    for m in methods
                ])
            
            prompt = f"""Reference solution approach:
{methods_text}

Now solve this problem step by step. End with #### NUMBER

Problem: {question}

Solution:"""
        else:
            # Baseline
            prompt = f"""Solve step by step. End with #### NUMBER

Problem: {question}

Solution:"""
        
        return prompt
    
    def _solve_single_with_retry(self, question: str, methods: List[Dict] = None, max_retries: int = MAX_RETRIES) -> tuple:
        prompt = self._create_prompt(question, methods)
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config={
                        'temperature': 0.1,
                        'max_output_tokens': 512,
                        'top_p': 0.95,
                        'top_k': 40
                    }
                )
                
                output = response.text if response.text else ""
                return output, None
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  ⚠️ Retry {attempt+1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return "", str(e)
        
        return "", "Max retries exceeded"
    
    def evaluate_baseline(self, max_samples: Optional[int] = None):
        """Baseline evaluation"""
        print(f"\n{'='*80}")
        print("🚀 BASELINE EVALUATION")
        print(f"{'='*80}\n")
        
        data = self.test_metadata[:max_samples] if max_samples else self.test_metadata
        all_results = []
        correct = 0
        error_count = 0
        failed_items = []
        
        print(f"📝 Processing {len(data)} questions...")
        
        for idx, item in enumerate(tqdm(data, desc="Baseline")):
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            output, error = self._solve_single_with_retry(question)
            
            if error:
                error_count += 1
                failed_items.append({"item": item, "error": error})
                all_results.append({
                    "index": item["index"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_output": "",
                    "prediction": None,
                    "is_correct": False,
                    "error": error
                })
            else:
                prediction = extract_single_answer(output)
                
                gt_val = normalize_number(ground_truth)
                pred_val = normalize_number(prediction)
                
                is_correct = (
                    gt_val is not None and
                    pred_val is not None and
                    abs(gt_val - pred_val) < 1e-6
                )
                
                if is_correct:
                    correct += 1
                
                all_results.append({
                    "index": item["index"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_output": output,
                    "prediction": prediction,
                    "is_correct": is_correct
                })
            
            time.sleep(RATE_LIMIT_DELAY)
            
            if (idx + 1) % 50 == 0:
                acc = 100 * correct / (idx + 1)
                print(f"\n  Progress: {idx+1}/{len(data)} | Acc: {acc:.1f}% | Errors: {error_count}")
        
        # RETRY
        if failed_items:
            print(f"\n🔄 Retrying {len(failed_items)} failed items...")
            for retry_data in tqdm(failed_items, desc="Retrying"):
                item = retry_data["item"]
                
                output, error = self._solve_single_with_retry(item["question"], max_retries=2)
                
                if not error:
                    prediction = extract_single_answer(output)
                    gt_val = normalize_number(item["ground_truth"])
                    pred_val = normalize_number(prediction)
                    
                    is_correct = (
                        gt_val is not None and
                        pred_val is not None and
                        abs(gt_val - pred_val) < 1e-6
                    )
                    
                    if is_correct:
                        correct += 1
                    
                    for r in all_results:
                        if r["index"] == item["index"]:
                            r["model_output"] = output
                            r["prediction"] = prediction
                            r["is_correct"] = is_correct
                            if "error" in r:
                                del r["error"]
                            break
                
                time.sleep(RATE_LIMIT_DELAY)
        
        accuracy = 100 * correct / len(data) if len(data) > 0 else 0
        
        print(f"\n✅ Baseline Complete: {accuracy:.2f}% ({correct}/{len(data)})")
        print(f"   Final errors: {sum(1 for r in all_results if 'error' in r)}")
        
        # Save
        save_path = OUTPUT_DIR / "baseline"
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        summary = {
            "model": self.model_id,
            "total_samples": len(data),
            "correct": correct,
            "accuracy": f"{accuracy:.2f}%"
        }
        
        with open(save_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return accuracy
    
    def evaluate_specific_db(self, db_name: str, max_samples: Optional[int] = None):
        """RAG evaluation with DETAILED METRICS"""
        if db_name not in self.databases:
            print(f"❌ Database '{db_name}' not found!")
            return None
        
        print(f"\n{'='*80}")
        print(f"🚀 RAG EVALUATION: {db_name}")
        print(f"{'='*80}\n")
        
        database = self.databases[db_name]
        data = self.test_metadata[:max_samples] if max_samples else self.test_metadata
        
        # DETAILED TRACKING
        all_results = []
        error_count = 0
        failed_items = []
        
        # Track separately: with_methods vs without_methods
        with_methods_correct = 0
        with_methods_total = 0
        without_methods_correct = 0
        without_methods_total = 0
        
        print(f"📝 Processing {len(data)} questions with RAG...")
        print(f"   Method strategy: {'FULL TEXT' if USE_FULL_METHOD else f'TRUNCATED ({MAX_METHOD_LENGTH} chars)'}\n")
        
        for idx, item in enumerate(tqdm(data, desc=f"RAG-{db_name}")):
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            # Retrieve methods
            query_embedding = self.embeddings[item["index"]]
            methods = database.retrieve(
                query_embedding,
                top_k=TOP_K_METHODS,
                threshold=SIMILARITY_THRESHOLD
            )
            
            has_methods = len(methods) > 0
            
            # Solve
            output, error = self._solve_single_with_retry(question, methods if has_methods else None)
            
            if error:
                error_count += 1
                failed_items.append({"item": item, "methods": methods, "error": error})
                all_results.append({
                    "index": item["index"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_output": "",
                    "prediction": None,
                    "is_correct": False,
                    "has_methods": has_methods,
                    "methods_used": len(methods),
                    "error": error
                })
                
                # Count for stats (consider as incorrect)
                if has_methods:
                    with_methods_total += 1
                else:
                    without_methods_total += 1
            else:
                prediction = extract_single_answer(output)
                
                gt_val = normalize_number(ground_truth)
                pred_val = normalize_number(prediction)
                
                is_correct = (
                    gt_val is not None and
                    pred_val is not None and
                    abs(gt_val - pred_val) < 1e-6
                )
                
                # Track by category
                if has_methods:
                    with_methods_total += 1
                    if is_correct:
                        with_methods_correct += 1
                else:
                    without_methods_total += 1
                    if is_correct:
                        without_methods_correct += 1
                
                all_results.append({
                    "index": item["index"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_output": output,
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "has_methods": has_methods,
                    "methods_used": len(methods),
                    "method_similarities": [m["similarity_score"] for m in methods] if methods else []
                })
            
            time.sleep(RATE_LIMIT_DELAY)
            
            if (idx + 1) % 50 == 0:
                overall_correct = with_methods_correct + without_methods_correct
                overall_acc = 100 * overall_correct / (idx + 1)
                with_methods_acc = 100 * with_methods_correct / with_methods_total if with_methods_total > 0 else 0
                without_methods_acc = 100 * without_methods_correct / without_methods_total if without_methods_total > 0 else 0
                
                print(f"\n  Progress: {idx+1}/{len(data)}")
                print(f"  Overall Acc: {overall_acc:.1f}%")
                print(f"  With Methods: {with_methods_acc:.1f}% ({with_methods_correct}/{with_methods_total})")
                print(f"  Without Methods: {without_methods_acc:.1f}% ({without_methods_correct}/{without_methods_total})")
                print(f"  Errors: {error_count}")
        
        # RETRY
        if failed_items:
            print(f"\n🔄 Retrying {len(failed_items)} failed items...")
            for retry_data in tqdm(failed_items, desc="Retrying"):
                item = retry_data["item"]
                methods = retry_data["methods"]
                has_methods = len(methods) > 0
                
                output, error = self._solve_single_with_retry(item["question"], methods if has_methods else None, max_retries=2)
                
                if not error:
                    prediction = extract_single_answer(output)
                    gt_val = normalize_number(item["ground_truth"])
                    pred_val = normalize_number(prediction)
                    
                    is_correct = (
                        gt_val is not None and
                        pred_val is not None and
                        abs(gt_val - pred_val) < 1e-6
                    )
                    
                    # Update category stats
                    if has_methods and is_correct:
                        with_methods_correct += 1
                    elif not has_methods and is_correct:
                        without_methods_correct += 1
                    
                    # Update result
                    for r in all_results:
                        if r["index"] == item["index"]:
                            r["model_output"] = output
                            r["prediction"] = prediction
                            r["is_correct"] = is_correct
                            if "error" in r:
                                del r["error"]
                            break
                
                time.sleep(RATE_LIMIT_DELAY)
        
        # FINAL METRICS
        total_correct = with_methods_correct + without_methods_correct
        total_samples = len(data)
        
        overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
        with_methods_accuracy = 100 * with_methods_correct / with_methods_total if with_methods_total > 0 else 0
        without_methods_accuracy = 100 * without_methods_correct / without_methods_total if without_methods_total > 0 else 0
        methods_usage_rate = 100 * with_methods_total / total_samples if total_samples > 0 else 0
        
        # Print detailed summary
        print(f"\n{'='*80}")
        print(f"✅ {db_name} COMPLETE - DETAILED METRICS")
        print(f"{'='*80}")
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
        print(f"\n🎯 METHOD USAGE:")
        print(f"   Usage Rate: {methods_usage_rate:.1f}% ({with_methods_total}/{total_samples} questions)")
        print(f"\n📈 ACCURACY BREAKDOWN:")
        print(f"   With Methods:    {with_methods_accuracy:.2f}% ({with_methods_correct}/{with_methods_total})")
        print(f"   Without Methods: {without_methods_accuracy:.2f}% ({without_methods_correct}/{without_methods_total})")
        print(f"\n💡 METHOD IMPACT:")
        if with_methods_total > 0 and without_methods_total > 0:
            impact = with_methods_accuracy - without_methods_accuracy
            print(f"   Difference: {impact:+.2f} percentage points")
            if impact > 0:
                print(f"   ✅ Methods IMPROVE performance")
            elif impact < 0:
                print(f"   ⚠️  Methods DECREASE performance")
            else:
                print(f"   ➡️  No significant difference")
        print(f"\n❌ Errors: {sum(1 for r in all_results if 'error' in r)}")
        print(f"{'='*80}")
        
        # Save
        save_path = OUTPUT_DIR / db_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Detailed summary
        summary = {
            "model": self.model_id,
            "db_name": db_name,
            "method_strategy": "full_text" if USE_FULL_METHOD else f"truncated_{MAX_METHOD_LENGTH}",
            "total_samples": total_samples,
            
            # Overall metrics
            "overall": {
                "correct": total_correct,
                "accuracy": f"{overall_accuracy:.2f}%"
            },
            
            # Method usage
            "method_usage": {
                "questions_with_methods": with_methods_total,
                "questions_without_methods": without_methods_total,
                "usage_rate": f"{methods_usage_rate:.1f}%"
            },
            
            # Performance breakdown
            "performance_breakdown": {
                "with_methods": {
                    "correct": with_methods_correct,
                    "total": with_methods_total,
                    "accuracy": f"{with_methods_accuracy:.2f}%"
                },
                "without_methods": {
                    "correct": without_methods_correct,
                    "total": without_methods_total,
                    "accuracy": f"{without_methods_accuracy:.2f}%"
                }
            },
            
            # Method impact
            "method_impact": {
                "accuracy_difference": f"{with_methods_accuracy - without_methods_accuracy:+.2f}%",
                "is_beneficial": with_methods_accuracy > without_methods_accuracy
            },
            
            # Config
            "config": {
                "top_k": TOP_K_METHODS,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "max_method_length": MAX_METHOD_LENGTH if not USE_FULL_METHOD else "full"
            }
        }
        
        with open(save_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return overall_accuracy