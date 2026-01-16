import json
import re
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_ID = "ytu-ce-cosmos/Turkish-Gemma-9b-T1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 4096
DTYPE = torch.bfloat16

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def extract_method_from_response(text: str) -> str:
    """Extract method from model response (JSON or plain text)"""
    
    # 1. Try to parse JSON
    try:
        # Look for JSON object
        json_match = re.search(r'\{[^}]*"method"\s*:\s*"([^"]+)"[^}]*\}', text)
        if json_match:
            return json_match.group(1)
        
        # Try full JSON parse
        parsed = json.loads(text)
        if "method" in parsed:
            return parsed["method"]
    except:
        pass
    
    # 2. Look for text after </think>
    think_end = text.rfind('</think>')
    if think_end != -1:
        after_think = text[think_end+8:].strip()
        
        # Remove <end_of_turn> if present
        after_think = after_think.replace('<end_of_turn>', '').strip()
        
        # Try JSON again
        try:
            json_match = re.search(r'\{[^}]*"method"\s*:\s*"([^"]+)"[^}]*\}', after_think)
            if json_match:
                return json_match.group(1)
        except:
            pass
        
        # Return cleaned text if no JSON
        if len(after_think) > 20:
            return after_think[:500]  # Max 500 chars
    
    # 3. Return full text (cleaned)
    cleaned = text.replace('<think>', '').replace('</think>', '')
    cleaned = cleaned.replace('<end_of_turn>', '').strip()
    return cleaned[:500] if cleaned else "No method extracted"


class MethodDatabaseBuilder:
    def __init__(self, model_id: str, embedding_model_name: str):
        print("🔧 Initializing Method Database Builder...")
        
        # Load reasoning model
        print(f"\n📥 Loading reasoning model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=DTYPE,
            device_map="balanced",
            low_cpu_mem_usage=True,
            max_memory={i: "14GB" for i in range(torch.cuda.device_count())}
        )
        self.model.eval()
        
        self.end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        
        # Load embedding model
        print(f"📥 Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Embedding dimension: {self.embedding_dim}")
    
    def generate_method(self, question: str, solution: str, prompt_template: str) -> Tuple[str, float]:
        """Generate method using reasoning model"""
        
        # Build prompt
        prompt = prompt_template.format(
            question=question,
            solution=solution
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda:0")
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=[self.tokenizer.eos_token_id, self.end_of_turn_id],
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        elapsed = time.time() - start_time
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        # Extract method
        method_text = extract_method_from_response(generated_text)
        
        return method_text, elapsed
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts"""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )
        return embeddings.astype('float32')
    
    def build_database(self, prompt_name: str, prompt_template: str, 
                      questions_data: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
        """Build FAISS database for one prompt type"""
        
        print(f"\n{'='*80}")
        print(f"Building database for prompt: {prompt_name}")
        print(f"{'='*80}\n")
        
        methods_data = []
        method_texts = []
        
        for idx, item in enumerate(tqdm(questions_data, desc=f"Generating methods ({prompt_name})")):
            question_id = item["id"]
            question = item["question"]
            solution = item.get("answer", "")  # If solution exists
            
            # Generate method
            method_text, gen_time = self.generate_method(question, solution, prompt_template)
            
            # Create metadata
            metadata = {
                "id": f"{question_id}_{prompt_name}",
                "question_id": question_id,
                "prompt_type": prompt_name,
                "original_question": question,
                "original_solution": solution,
                "method_text": method_text,
                "generation_time": round(gen_time, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            methods_data.append(metadata)
            method_texts.append(method_text)
            
            # Show first 10 examples
            if idx < 5:
                print(f"\nExample {idx+1}:")
                print(f"Solution: {solution}")
                print(f"Method: {method_text}")
                print()
        
        # Create embeddings
        print(f"\n🔢 Creating embeddings for {len(method_texts)} methods...")
        embeddings = self.create_embeddings(method_texts)
        
        # Add embeddings to metadata (for debugging/inspection)
        for i, metadata in enumerate(methods_data):
            metadata["method_embedding"] = embeddings[i].tolist()
        
        # Build FAISS index
        print(f"🔍 Building FAISS index...")
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity)
        index.add(embeddings)
        
        print(f"✅ Database built: {index.ntotal} vectors indexed")
        
        return index, methods_data
    
    def save_database(self, prompt_name: str, index: faiss.Index, 
                     metadata: List[Dict], output_dir: Path):
        """Save FAISS index and metadata"""
        
        db_dir = output_dir / prompt_name
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = db_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = db_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary = {
            "prompt_type": prompt_name,
            "total_methods": len(metadata),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": self.embedding_dim,
            "reasoning_model": MODEL_ID,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_path = db_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Database saved to: {db_dir}")
        print(f"   - FAISS index: {index_path}")
        print(f"   - Metadata: {metadata_path}")
        print(f"   - Summary: {summary_path}")
