#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Qwen3-8B on TAB graph reasoning tasks.

Pipeline:
  1. Load base LLM + LoRA adapter + graph bias weights from ``output/``.
  2. Encode each ``test.json`` sample into a Question/Graph/Answer prompt.
  3. Generate an answer and compare against ground truth via ``answer_evaluator``.
  4. Save per-task results to ``tasks/<task>/test_results_qwen3_8b_*.json``.

Run from the ``testing/`` directory: ``python test_qwen3_8b_finetuned.py``
"""
import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Any
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'finetuning'))
from encoder.graph_sequence_encoder import GraphSequenceEncoder
from answer_evaluator import compare_answers, extract_first_sentence, build_edge_set_from_edge_index, build_directed_edges_from_edge_index, build_edge_set, build_edge_set_from_question, build_directed_edges_from_question
from datetime import datetime
from simple_qwen3_8b_finetuning import GraphEnhancedLoRALLM


def _compute_graph_density(item: Dict[str, Any]) -> float:
    """Estimate edge density (|E| / |E_max|) for filtering dense-graph ablation samples."""
    num_nodes = int(item.get("num_nodes", 0) or 0)
    if num_nodes <= 1:
        return 0.0
    edge_index = item.get("edge_index")
    if not edge_index:
        return 0.0
    try:
        if isinstance(edge_index, list) and len(edge_index) >= 2:
            m = min(len(edge_index[0]), len(edge_index[1]))
        elif isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            m = int(edge_index.shape[1])
        else:
            return 0.0
        # Count undirected edges by removing mirrored duplicates if present.
        undirected = bool(not item.get("is_directed", False))
        if undirected:
            m = max(1, m // 2)
            max_edges = num_nodes * (num_nodes - 1) / 2.0
        else:
            max_edges = num_nodes * (num_nodes - 1)
        if max_edges <= 0:
            return 0.0
        return float(m) / float(max_edges)
    except Exception:
        return 0.0


def _evaluate_single_item(
    model,
    tokenizer,
    graph_encoder: GraphSequenceEncoder,
    task_name: str,
    item: Dict[str, Any],
    device: str,
    use_graph_bias: bool = True,
):
    """Run one test sample end-to-end and return correctness plus timing metrics."""
    t_pre0 = time.perf_counter()
    input_data = convert_test_item_to_input(item, graph_encoder)
    input_text = f"Question: {input_data['question']}\nGraph: {input_data['graph_sequence']}\nAnswer:"
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        truncation=True,
        max_length=512
    ).to(device)
    t_pre1 = time.perf_counter()

    with torch.no_grad():
        if use_graph_bias and isinstance(model, GraphEnhancedLoRALLM):
            graph_data = {
                'graph_features': input_data['graph_features'].to(device),
                'edge_index': input_data['edge_index'].to(device)
            }
            t_gen0 = time.perf_counter()
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                graph_data=graph_data,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            t_gen1 = time.perf_counter()
        else:
            t_gen0 = time.perf_counter()
            if isinstance(model, GraphEnhancedLoRALLM):
                outputs = model.lora_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            t_gen1 = time.perf_counter()

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_answer = extract_answer_from_generation(response, len(input_text))
    edge_set = None
    directed_edges = None
    # Task-specific graph structure needed by answer_evaluator for validation.
    if task_name == "topological_sort":
        if input_data.get('edge_index') is not None:
            directed_edges = build_directed_edges_from_edge_index(input_data['edge_index'])
        elif input_data.get('question'):
            directed_edges = build_directed_edges_from_question(input_data['question'])
    else:
        if input_data.get('edge_index') is not None:
            edge_set = build_edge_set_from_edge_index(input_data['edge_index'])
        elif input_data.get('graph_sequence'):
            edge_set = build_edge_set(input_data['graph_sequence'])
        elif input_data.get('question'):
            edge_set = build_edge_set_from_question(input_data['question'])
    predicted_answer = extract_first_sentence(predicted_answer)
    is_correct = compare_answers(predicted_answer, input_data['answer'], task_name, edge_set, directed_edges)

    return {
        "is_correct": bool(is_correct),
        "preprocess_sec": t_pre1 - t_pre0,
        "generation_sec": t_gen1 - t_gen0,
    }


def run_bfs_truncation_ablation(
    model_path: str = "../output/qwen3_8b_finetuned",
    base_model_name: str = "./Qwen3-8B-hf",
    tasks_dir: str = "../tasks",
    use_graph_bias: bool = True,
    fast_generation: bool = False,
    target_tasks: List[str] = None,
    density_threshold: float = 0.20,
    max_samples_per_task: int = 100,
    budget_grid: List[Dict[str, Any]] = None,
):
    """Ablation over graph path search budgets on dense graphs (accuracy vs efficiency)."""
    if target_tasks is None:
        target_tasks = ["shortest_path", "max_flow"]
    if budget_grid is None:
        budget_grid = [
            {"name": "tight", "max_paths": 10, "max_path_length": 8, "max_hops": 3, "max_sequences": 8, "search_time_budget_sec": 0.01},
            {"name": "medium", "max_paths": 25, "max_path_length": 14, "max_hops": 3, "max_sequences": 8, "search_time_budget_sec": 0.03},
            {"name": "loose", "max_paths": 50, "max_path_length": 20, "max_hops": 3, "max_sequences": 8, "search_time_budget_sec": 0.06},
            {"name": "very_loose", "max_paths": 100, "max_path_length": 40, "max_hops": 3, "max_sequences": 8, "search_time_budget_sec": 10.0},
        ]

    print("=== BFS Truncation Accuracy-Efficiency Ablation (Qwen3-8B) ===")
    print(f"target_tasks={target_tasks}, density_threshold={density_threshold}, max_samples_per_task={max_samples_per_task}")
    model, tokenizer, graph_encoder = load_finetuned_model(
        model_path=model_path,
        base_model_name=base_model_name,
        use_graph_bias=use_graph_bias,
        fast_generation=fast_generation,
    )
    if model is None or tokenizer is None or graph_encoder is None:
        print("❌ Model loading failed, cannot continue ablation")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "model_path": model_path,
        "base_model_name": base_model_name,
        "density_threshold": density_threshold,
        "max_samples_per_task": max_samples_per_task,
        "target_tasks": target_tasks,
        "budget_grid": budget_grid,
        "results": [],
        "timestamp": timestamp,
    }

    for budget in budget_grid:
        print(f"\n--- Budget: {budget.get('name', 'unnamed')} ---")
        graph_encoder.set_search_budgets(
            max_paths=budget.get("max_paths"),
            max_path_length=budget.get("max_path_length"),
            max_hops=budget.get("max_hops"),
            max_sequences=budget.get("max_sequences"),
            search_time_budget_sec=budget.get("search_time_budget_sec"),
        )
        graph_encoder.reset_search_stats()
        for task_name in target_tasks:
            test_json_path = os.path.join(tasks_dir, task_name, "test.json")
            if not os.path.exists(test_json_path):
                print(f"Skip task {task_name}: no test.json")
                continue
            with open(test_json_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            dense_data = [x for x in task_data if _compute_graph_density(x) >= density_threshold]
            if max_samples_per_task is not None and len(dense_data) > max_samples_per_task:
                dense_data = dense_data[:max_samples_per_task]
            if not dense_data:
                print(f"Task {task_name}: no dense samples >= {density_threshold}")
                continue

            correct = 0
            total_pre = 0.0
            total_gen = 0.0
            for item in dense_data:
                metrics = _evaluate_single_item(
                    model=model,
                    tokenizer=tokenizer,
                    graph_encoder=graph_encoder,
                    task_name=task_name,
                    item=item,
                    device=device,
                    use_graph_bias=use_graph_bias,
                )
                correct += int(metrics["is_correct"])
                total_pre += metrics["preprocess_sec"]
                total_gen += metrics["generation_sec"]

            stats = graph_encoder.get_search_stats()
            total_samples = len(dense_data)
            # Call-based trigger rates are reported for transparency.
            fap_calls = max(1, stats.get("find_all_paths_calls", 0))
            bfs_calls = max(1, stats.get("bfs_calls", 0))
            task_result = {
                "budget": budget,
                "task_name": task_name,
                "num_dense_samples": total_samples,
                "accuracy": correct / max(1, total_samples),
                "avg_preprocess_ms": (total_pre / max(1, total_samples)) * 1000.0,
                "avg_generation_ms": (total_gen / max(1, total_samples)) * 1000.0,
                "search_stats": stats,
                "trigger_rates": {
                    "find_all_paths_path_cap_rate": stats.get("find_all_paths_truncated_by_path_cap", 0) / fap_calls,
                    "find_all_paths_max_len_rate": stats.get("find_all_paths_truncated_by_max_len", 0) / fap_calls,
                    "find_all_paths_timeout_rate": stats.get("find_all_paths_truncated_by_timeout", 0) / fap_calls,
                    "bfs_path_cap_rate": stats.get("bfs_truncated_by_path_cap", 0) / bfs_calls,
                    "bfs_hop_cap_rate": stats.get("bfs_truncated_by_hop_cap", 0) / bfs_calls,
                    "bfs_timeout_rate": stats.get("bfs_truncated_by_timeout", 0) / bfs_calls,
                },
            }
            report["results"].append(task_result)
            print(
                f"Task={task_name}, n={total_samples}, acc={task_result['accuracy']:.4f}, "
                f"pre={task_result['avg_preprocess_ms']:.2f}ms, gen={task_result['avg_generation_ms']:.2f}ms"
            )

    out_path = os.path.join(tasks_dir, f"bfs_truncation_ablation_qwen3_8b_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Ablation report saved to: {out_path}")
def load_finetuned_model(model_path: str = "../output/qwen3_8b_finetuned",
                        base_model_name: str = "./Qwen3-8B-hf",
                        use_graph_bias: bool = True,
                        fast_generation: bool = False):
    """Load base model, LoRA adapter, graph encoder, and optional bias weights."""
    print(f"Loading base model: {base_model_name}")
    print(f"Loading fine-tuned model: {model_path}")
    hf_model_name = "Qwen/Qwen3-8B"
    if not os.path.exists(base_model_name):
        print(f"⚠️ Local model path does not exist: {base_model_name}")
        print(f"Will attempt to download model from Hugging Face: {hf_model_name}")
        actual_model_name = hf_model_name
    else:
        print(f"✓ Using local model: {base_model_name}")
        actual_model_name = base_model_name
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        print("Using bfloat16 precision")
    else:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using {torch_dtype} Accuracy")
    try:
        tokenizer = AutoTokenizer.from_pretrained(actual_model_name, use_fast=False, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            actual_model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading LoRA adapter...")
    lora_model = PeftModel.from_pretrained(base_model, model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() or (hasattr(lora_model, 'device') and isinstance(lora_model.device, torch.device) and lora_model.device.type == "cpu"):
        lora_model = lora_model.to(device)
    hidden_size = lora_model.config.hidden_size
    vocab_size = tokenizer.vocab_size
    num_attention_heads = lora_model.config.num_attention_heads
    print(f"Model configuration: hidden_size={hidden_size}, vocab_size={vocab_size}, num_heads={num_attention_heads}")
    graph_encoder = GraphSequenceEncoder(
        graph_dim=64,
        llm_dim=hidden_size,
        vocab_size=vocab_size,
        max_sequence_length=512,
        use_graph_bias=use_graph_bias,
        num_heads=num_attention_heads
    )
    if use_graph_bias:
        print("Loading graph bias weights...")
        graph_bias_path = os.path.join(model_path, 'graph_bias_weights.pt')
        if os.path.exists(graph_bias_path):
            graph_bias_state = torch.load(graph_bias_path, map_location=device)
            print(f"  From {graph_bias_path} loading graph bias weights")
        else:
            print(f"  ⚠️ Graph bias weights file does not exist: {graph_bias_path}，using default weights")
            graph_bias_state = None
        # Wrap LoRA model with graph bias injection for inference.
        model = GraphEnhancedLoRALLM(
            lora_model,
            graph_encoder,
            tokenizer=tokenizer,
            use_graph_bias=True,
            debug=False,
            fast_generation=fast_generation
        )
        if graph_bias_state is not None:
            if 'bias_weight' in graph_bias_state:
                if isinstance(graph_bias_state['bias_weight'], dict):
                    model.bias_weight.data = graph_bias_state['bias_weight'].get('data', graph_bias_state['bias_weight'])
                else:
                    model.bias_weight.data = graph_bias_state['bias_weight']
            if 'bias_projection' in graph_bias_state and hasattr(model, 'bias_projection'):
                model.bias_projection.load_state_dict(graph_bias_state['bias_projection'])
            print("  ✓ Graph bias weights loaded successfully")
    else:
        model = lora_model
    model.eval()
    return model, tokenizer, graph_encoder
def convert_test_item_to_input(item: Dict[str, Any], graph_encoder: GraphSequenceEncoder) -> Dict[str, Any]:
    """Convert a raw test JSON item into prompt fields and graph tensors."""
    edge_index_data = item['edge_index']
    num_nodes = item['num_nodes']
    if isinstance(edge_index_data, list):
        if len(edge_index_data) == 2 and isinstance(edge_index_data[0], list):
            edge_index = torch.tensor(edge_index_data, dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index_data, dtype=torch.long)
    else:
        edge_index = edge_index_data if isinstance(edge_index_data, torch.Tensor) else torch.tensor(edge_index_data)
    if edge_index.dim() == 2 and edge_index.shape[0] != 2:
        edge_index = edge_index.t()
    dummy_node_features = torch.randn(num_nodes, 64)
    question_text = item.get('full_question', item.get('question', ''))
    graph_sequence = graph_encoder(
        dummy_node_features,
        edge_index,
        encoder_type='node_sequence',
        question_text=question_text
    )
    lines = graph_sequence.split('\n')
    path_lines = []
    # Keep only Path lines and renumber sequentially for a clean prompt.
    for line in lines:
        if line.startswith('Path') and ':' in line:
            path_part = line.split(':', 1)[1].strip()
            path_lines.append(f"Path {len(path_lines) + 1}: {path_part}")
    if path_lines:
        is_directed = '→' in path_lines[0] if path_lines else True
        edge_symbol = '→' if is_directed else '-'
        graph_type = 'directed' if is_directed else 'undirected'
        graph_explanation = f"The graph is {graph_type}. Each path shows a sequence of connected nodes, where '{edge_symbol}' represents a {'directed' if is_directed else 'undirected'} connection between nodes.\n"
        cleaned_sequence = graph_explanation + '\n'.join(path_lines)
    else:
        cleaned_sequence = "No valid paths found in the graph."
    return {
        'graph_sequence': cleaned_sequence,
        'question': question_text,
        'answer': item.get('answer', ''),
        'graph_features': dummy_node_features,
        'edge_index': edge_index
    }
def generate_with_graph_bias(model, input_ids, attention_mask, graph_data, tokenizer, max_new_tokens=100,
                             temperature=0.7, do_sample=True, **kwargs):
    """Generate tokens with graph bias; falls back to a manual decode loop if needed."""
    if hasattr(model, 'generate') and callable(getattr(model, 'generate', None)):
        try:
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_data=graph_data,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
            )
        except Exception as e:
            print(f"Warning: Warning: Failed to use model's generate method, falling back to custom loop: {e}")
    device = input_ids.device
    model.eval()
    generated_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone() if attention_mask is not None else None
    eos_token_id = kwargs.get('eos_token_id', tokenizer.eos_token_id)
    pad_token_id = kwargs.get('pad_token_id', tokenizer.pad_token_id or tokenizer.eos_token_id)
    with torch.no_grad():
        # Manual token-by-token loop used when model.generate lacks graph_data support.
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=generated_ids,
                attention_mask=current_attention_mask,
                graph_data=graph_data,
            )
            logits = outputs.logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if current_attention_mask is not None:
                new_mask = torch.ones((current_attention_mask.shape[0], 1),
                                     dtype=current_attention_mask.dtype,
                                     device=device)
                current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)
            if next_token.item() == eos_token_id:
                break
    return generated_ids
def extract_answer_from_generation(full_response: str, input_text_len: int) -> str:
    """Strip the input prompt prefix and extract the model answer span."""
    if len(full_response) > input_text_len:
        answer_part = full_response[input_text_len:].strip()
    else:
        answer_part = full_response.strip()
    question_idx = answer_part.find("\nQuestion:")
    graph_idx = answer_part.find("\nGraph:")
    if question_idx != -1:
        answer_part = answer_part[:question_idx].strip()
    elif graph_idx != -1:
        answer_part = answer_part[:graph_idx].strip()
    answer_prefixes = ["Answer:", "answer:", "Answer", "answer"]
    for prefix in answer_prefixes:
        if answer_part.startswith(prefix):
            answer_part = answer_part[len(prefix):].strip()
            while any(answer_part.startswith(p) for p in answer_prefixes):
                for p in answer_prefixes:
                    if answer_part.startswith(p):
                        answer_part = answer_part[len(p):].strip()
            break
    stop_words = ["\n\nQuestion:", "\nQuestion:", "\n\nGraph:", "\nGraph:", "\n\nAnswer:", "\nAnswer:"]
    for stop_word in stop_words:
        if stop_word in answer_part:
            answer_part = answer_part[:answer_part.find(stop_word)].strip()
    return answer_part
def test_model_on_tasks(model_path: str = "../output/qwen3_8b_finetuned",
                       base_model_name: str = "./Qwen3-8B-hf",
                       tasks_dir: str = "../tasks",
                       use_graph_bias: bool = True,
                       fast_generation: bool = False,
                       max_samples_per_task: int = None,
                       show_samples: bool = False):
    """Run inference on all tasks under *tasks_dir* and report per-task accuracy."""
    print("=== === Testing Qwen3-8B Fine-tuned Model ===")
    print(f"Model path: {model_path}")
    print(f"Base model: {base_model_name}")
    print(f"Use graph bias: {use_graph_bias}")
    print(f"Fast generation mode: {fast_generation}")
    if fast_generation:
        print("  ⚠️   ⚠️ Note: In fast generation mode, graph bias is not used during generation (speed prioritized)")
    print(f"Test data directory: {tasks_dir}")
    print("Using test.json")
    print()
    model, tokenizer, graph_encoder = load_finetuned_model(
        model_path=model_path,
        base_model_name=base_model_name,
        use_graph_bias=use_graph_bias,
        fast_generation=fast_generation
    )
    if model is None or tokenizer is None or graph_encoder is None:
        print("❌ Model loading failed, cannot continue testing")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dirs = []
    for item in os.listdir(tasks_dir):
        task_path = os.path.join(tasks_dir, item)
        if os.path.isdir(task_path):
            test_json_path = os.path.join(task_path, "test.json")
            if os.path.exists(test_json_path):
                task_dirs.append(item)
                print(f"Found task: {item} (using test.json)")
    if not task_dirs:
        print(f"❌ In {tasks_dir} no test data files found")
        return
    print(f"\nFound {len(task_dirs)} tasks\n")
    all_results = {}
    for task_name in task_dirs:
        print(f"\n{'='*60}")
        print(f"Testing task: {task_name}")
        print(f"{'='*60}")
        test_json_path = os.path.join(tasks_dir, task_name, "test.json")
        try:
            with open(test_json_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            print(f"Loaded {len(test_data)} test samples (using test.json)")
        except Exception as e:
            print(f"❌ Loading {test_json_path} error: {e}")
            continue
        if max_samples_per_task and max_samples_per_task < len(test_data):
            test_data = test_data[:max_samples_per_task]
            print(f"Limited to first {max_samples_per_task} samples")
        correct_count = 0
        total_count = len(test_data)
        task_outputs = []
        for i, item in enumerate(test_data):
            try:
                input_data = convert_test_item_to_input(item, graph_encoder)
                # Extra visibility: if graph_sequence has no valid paths, log sample id for debugging.
                try:
                    gs = input_data.get('graph_sequence', '')
                    if isinstance(gs, str) and ('No valid paths found' in gs or 'Path ' not in gs):
                        print(f"  ⚠️ No node identifiers/paths detected for sample_id={i} (task={task_name})")
                except Exception:
                    pass
                input_text = f"Question: {input_data['question']}\nGraph: {input_data['graph_sequence']}\nAnswer:"
                inputs = tokenizer(
                    input_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(device)
                with torch.no_grad():
                    if use_graph_bias and isinstance(model, GraphEnhancedLoRALLM):
                        graph_data = {
                            'graph_features': input_data['graph_features'].to(device),
                            'edge_index': input_data['edge_index'].to(device)
                        }
                        try:
                            outputs = model.generate(
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs.get('attention_mask'),
                                graph_data=graph_data,
                                max_new_tokens=100,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                        except Exception as e:
                            print(f"  ⚠️ Warning: Failed to use model's generate method, falling back to custom loop: {e}")
                            outputs = generate_with_graph_bias(
                                model=model,
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs.get('attention_mask'),
                                graph_data=graph_data,
                                tokenizer=tokenizer,
                                max_new_tokens=100,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                    else:
                        if isinstance(model, GraphEnhancedLoRALLM):
                            outputs = model.lora_model.generate(
                                **inputs,
                                max_new_tokens=100,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                repetition_penalty=1.2
                            )
                        else:
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=100,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                repetition_penalty=1.2
                            )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_answer = extract_answer_from_generation(response, len(input_text))
                edge_set = None
                directed_edges = None
                if task_name == "topological_sort":
                    if input_data.get('edge_index') is not None:
                        directed_edges = build_directed_edges_from_edge_index(input_data['edge_index'])
                    elif input_data.get('question'):
                        directed_edges = build_directed_edges_from_question(input_data['question'])
                else:
                    if input_data.get('edge_index') is not None:
                        edge_set = build_edge_set_from_edge_index(input_data['edge_index'])
                    elif input_data.get('graph_sequence'):
                        edge_set = build_edge_set(input_data['graph_sequence'])
                    elif input_data.get('question'):
                        edge_set = build_edge_set_from_question(input_data['question'])
                predicted_answer = extract_first_sentence(predicted_answer)
                is_correct = compare_answers(predicted_answer, input_data['answer'], task_name, edge_set, directed_edges)
                if is_correct:
                    correct_count += 1
                output_item = {
                    'sample_id': i,
                    'question': input_data['question'],
                    'graph_sequence': input_data['graph_sequence'],
                    'ground_truth': input_data['answer'],
                    'predicted': predicted_answer,
                    'is_correct': is_correct,
                    'full_response': response
                }
                task_outputs.append(output_item)
                if show_samples or i < 3:
                    print(f"\n  Sample {i+1}/{total_count}:")
                    print(f"    Question: {input_data['question'][:100]}...")
                    print(f"    Predicted: {predicted_answer[:100]}...")
                    print(f"    Ground truth: {input_data['answer'][:100]}...")
                    print(f"    Result: {'✓ Correct' if is_correct else '✗ Incorrect'}")
            except Exception as e:
                print(f"  ❌ Sample {i+1} processing error: {e}")
                import traceback
                traceback.print_exc()
                task_outputs.append({
                    'sample_id': i,
                    'question': item.get('question', ''),
                    'error': str(e)
                })
                continue
        accuracy = correct_count / total_count if total_count > 0 else 0
        all_results[task_name] = {
            'correct': correct_count,
            'total': total_count,
            'accuracy': accuracy
        }
        print(f"\n  {task_name} Result: {correct_count}/{total_count} = {accuracy:.2%}")
        task_dir = os.path.join(tasks_dir, task_name)
        output_file = os.path.join(task_dir, f"test_results_qwen3_8b_{timestamp}.json")
        output_data = {
            'model_path': model_path,
            'base_model': base_model_name,
            'task_name': task_name,
            'test_time': timestamp,
            'use_graph_bias': use_graph_bias,
            'summary': {
                'correct': correct_count,
                'total': total_count,
                'accuracy': accuracy
            },
            'detailed_outputs': task_outputs
        }
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Test results saved to: {output_file}")
        except Exception as e:
            print(f"  ❌ Error saving results: {e}")
    print(f"\n{'='*60}")
    print("=== Overall Test Results ===")
    print(f"{'='*60}")
    print(f"{'Task Type':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 60)
    total_correct = 0
    total_samples = 0
    for task_name in sorted(all_results.keys()):
        result = all_results[task_name]
        print(f"{task_name:<25} {result['correct']:<10} {result['total']:<10} {result['accuracy']:<10.2%}")
        total_correct += result['correct']
        total_samples += result['total']
    if total_samples > 0:
        overall_accuracy = total_correct / total_samples
        print("-" * 60)
        print(f"{'Overall':<25} {total_correct:<10} {total_samples:<10} {overall_accuracy:<10.2%}")
    print(f"\n✅ All tests completed!")
def main():
    """
    Entry point: run standard task evaluation, or BFS ablation when
    ``RUN_BFS_ABLATION=1`` is set in the environment.
    """
    if os.environ.get("RUN_BFS_ABLATION", "0") == "1":
        run_bfs_truncation_ablation(
            model_path="../output/qwen3_8b_finetuned",
            base_model_name="./Qwen3-8B-hf",
            tasks_dir="../tasks",
            use_graph_bias=True,
            fast_generation=False,
            target_tasks=["shortest_path", "max_flow"],
            density_threshold=0.20,
            max_samples_per_task=100,
        )
    else:
        test_model_on_tasks(
            model_path="../output/qwen3_8b_finetuned",
            base_model_name="./Qwen3-8B-hf",
            tasks_dir="../tasks",
            use_graph_bias=True,
            fast_generation=False,
            max_samples_per_task=None,
            show_samples=True,
        )
if __name__ == "__main__":
    main()