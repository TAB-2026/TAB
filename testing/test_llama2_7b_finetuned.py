#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Llama-2-7B on TAB graph reasoning tasks.

Pipeline:
  1. Load base LLM + LoRA adapter + graph bias weights from ``output/``.
  2. Encode each ``test.json`` sample into a Question/Graph/Answer prompt.
  3. Generate an answer and compare against ground truth via ``answer_evaluator``.
  4. Save per-task results to ``tasks/<task>/test_results_llama2_7b_*.json``.

Run from the ``testing/`` directory: ``python test_llama2_7b_finetuned.py``
"""
import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'finetuning'))
from encoder.graph_sequence_encoder import GraphSequenceEncoder
from answer_evaluator import compare_answers, extract_first_sentence, build_edge_set_from_edge_index, build_directed_edges_from_edge_index, build_edge_set, build_edge_set_from_question, build_directed_edges_from_question
from datetime import datetime
from simple_llama2_7b_finetuning import GraphEnhancedLoRALLM
def load_finetuned_model(model_path: str = "../output/llama2_7b_finetuned",
                        base_model_name: str = "./Llama-2-7b-hf",
                        use_graph_bias: bool = True,
                        fast_generation: bool = False):
    """Load base model, LoRA adapter, graph encoder, and optional bias weights."""
    print(f"Loading base model: {base_model_name}")
    print(f"Loading fine-tuned model: {model_path}")
    hf_model_name = "meta-llama/Llama-2-7b-hf"
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
        print(f"Using {torch_dtype} precision")
    try:
        tokenizer = AutoTokenizer.from_pretrained(actual_model_name, use_fast=False)
        base_model = AutoModelForCausalLM.from_pretrained(
            actual_model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None
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
def test_model_on_tasks(model_path: str = "../output/llama2_7b_finetuned",
                       base_model_name: str = "./Llama-2-7b-hf",
                       tasks_dir: str = "../tasks",
                       use_graph_bias: bool = True,
                       fast_generation: bool = False,
                       max_samples_per_task: int = None,
                       show_samples: bool = False):
    """Run inference on all tasks under *tasks_dir* and report per-task accuracy."""
    print("=== === Testing Llama-2-7B Fine-tuned Model ===")
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
                                pad_token_id=tokenizer.eos_token_id,
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
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                    else:
                        if isinstance(model, GraphEnhancedLoRALLM):
                            outputs = model.lora_model.generate(
                                **inputs,
                                max_new_tokens=100,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                repetition_penalty=1.2
                            )
                        else:
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=100,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                repetition_penalty=1.2
                            )
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
        output_file = os.path.join(task_dir, f"test_results_llama2_7b_{timestamp}.json")
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
    """Entry point: evaluate the fine-tuned model on all graph tasks."""
    test_model_on_tasks(
        model_path="../output/llama2_7b_finetuned",
        base_model_name="./Llama-2-7b-hf",
        tasks_dir="../tasks",
        use_graph_bias=True,
        fast_generation=False,
        max_samples_per_task=None,
        show_samples=True,
    )
if __name__ == "__main__":
    main()