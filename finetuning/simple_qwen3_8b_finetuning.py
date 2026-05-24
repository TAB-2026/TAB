#!/usr/bin/env python3
"""
LoRA fine-tuning script for Qwen3-8B with TAB graph attentional bias.

Pipeline:
  1. Load graph QA samples from ``tasks/*/train.json``.
  2. Encode each graph via ``GraphSequenceEncoder`` into path text + bias tensors.
  3. Wrap the LoRA-adapted LLM in ``GraphEnhancedLoRALLM`` to inject structural bias.
  4. Train with HuggingFace ``Trainer`` (``train_enhanced_*``) or plain LoRA (``train_simple_*``).

Default output: ``../output/qwen3_8b_finetuned`` (enhanced) or ``../output/simple_qwen3_8b_finetuned``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import random
import json
import os
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoder.graph_sequence_encoder import GraphSequenceEncoder
from torch_geometric.data import Data

def generate_fixed_node_features(num_nodes: int, feature_dim: int = 64) -> torch.Tensor:
    """Return uniform node features (all ones) as a placeholder graph signal."""
    return torch.ones(num_nodes, feature_dim)


class GraphDataCollator:
    """
    Batch collator that separates graph tensors from text fields before LM collation.

    Pops ``graph_features`` and ``edge_index`` from each sample so the standard
    ``DataCollatorForLanguageModeling`` can pad ``input_ids`` / ``labels``, then
    re-attaches graph data for ``GraphBiasTrainer``.
    """

    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.text_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, features):
        graph_features_list = []
        edge_index_list = []
        text_features = []
        for feature in features:
            if "graph_features" in feature:
                graph_features_list.append(feature.pop("graph_features"))
            else:
                graph_features_list.append(None)
            if "edge_index" in feature:
                edge_index_list.append(feature.pop("edge_index"))
            else:
                edge_index_list.append(None)
            text_features.append(feature)
        batch = self.text_collator(text_features)
        if any(gf is not None for gf in graph_features_list):
            batch["graph_features"] = graph_features_list
        if any(ei is not None for ei in edge_index_list):
            batch["edge_index"] = edge_index_list
        return batch


class GraphBiasTrainer(Trainer):
    """
    HuggingFace Trainer that forwards graph tensors into the wrapped model.

    Extracts ``graph_features`` / ``edge_index`` from the batch, packs them as
    ``graph_data``, and logs bias application statistics every 100 steps.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        graph_features = inputs.pop("graph_features", None)
        edge_index = inputs.pop("edge_index", None)
        if graph_features is not None and edge_index is not None:
            if isinstance(graph_features, list):
                graph_features = graph_features[0] if len(graph_features) > 0 else None
            if isinstance(edge_index, list):
                edge_index = edge_index[0] if len(edge_index) > 0 else None
            if graph_features is not None and edge_index is not None:
                if not isinstance(graph_features, torch.Tensor):
                    graph_features = torch.tensor(graph_features) if graph_features is not None else None
                if not isinstance(edge_index, torch.Tensor):
                    edge_index = torch.tensor(edge_index) if edge_index is not None else None
                inputs["graph_data"] = {
                    "graph_features": graph_features,
                    "edge_index": edge_index,
                }
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
        if hasattr(model, '_bias_applied_count') and self._step_count % 100 == 0:
            total = model._bias_applied_count + model._bias_skipped_count + model._bias_failed_count
            if total > 0:
                print(f"\n[Graph Bias Statistics] Applied: {model._bias_applied_count}, Skipped: {model._bias_skipped_count}, Failed: {model._bias_failed_count} (Total: {total})")
        return result


class GraphEnhancedLoRALLM(nn.Module):
    """
    LoRA-wrapped causal LM with injectable graph structural attention bias.

    During forward / generate:
      1. ``GraphSequenceEncoder`` computes a node-level bias matrix.
      2. Token positions of node identifiers (N1, N2, …) are located in the prompt.
      3. Bias is added to attention logits via per-layer hooks (requires eager attention).

    Also trains a scalar ``bias_weight`` and a head projection ``bias_projection``.
    """

    def __init__(
        self,
        lora_model,
        graph_encoder: GraphSequenceEncoder,
        tokenizer=None,
        use_graph_bias: bool = True,
        debug: bool = False,
        fast_generation: bool = False,
    ):
        super().__init__()
        self.lora_model = lora_model
        self.graph_encoder = graph_encoder
        self.tokenizer = tokenizer
        self.use_graph_bias = use_graph_bias
        self.debug = debug
        self.fast_generation = fast_generation
        self._bias_applied_count = 0
        self._bias_skipped_count = 0
        self._bias_failed_count = 0
        self.bias_weight = nn.Parameter(torch.tensor(0.1))
        if use_graph_bias:
            base_model = lora_model.get_base_model() if hasattr(lora_model, "get_base_model") else lora_model
            num_heads = base_model.config.num_attention_heads
            self.bias_projection = nn.Linear(graph_encoder.num_heads, num_heads)
            # Flash/SDPA attention kernels cannot accept custom bias hooks; force eager mode.
            try:
                if hasattr(base_model, 'set_attn_implementation'):
                    base_model.set_attn_implementation('eager')
                elif hasattr(base_model, 'config'):
                    if hasattr(base_model.config, '_attn_implementation'):
                        base_model.config._attn_implementation = 'eager'
                    if hasattr(base_model.config, '_attn_implementation_internal'):
                        base_model.config._attn_implementation_internal = 'eager'
            except Exception as e:
                if self.debug:
                    print(f"Warning: Could not set attention implementation to 'eager': {e}")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Training forward pass: inject graph bias when ``graph_data`` is provided."""
        graph_data = kwargs.pop("graph_data", None)
        if self.use_graph_bias and graph_data is not None:
            attn_bias = self.graph_encoder.compute_graph_bias(
                graph_data["graph_features"],
                graph_data["edge_index"],
            )
            if attn_bias is not None and input_ids is not None:
                node_token_indices = self._create_node_token_mapping(input_ids)
                if node_token_indices:
                    try:
                        attn_bias_processed = self._prepare_attn_bias(
                            attn_bias, node_token_indices, input_ids.shape[1]
                        )
                        if attn_bias_processed is not None:
                            outputs = self._forward_with_graph_bias_hook(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                attn_bias=attn_bias_processed,
                                node_token_indices=node_token_indices,
                                **kwargs,
                            )
                            self._bias_applied_count += 1
                            if self.debug or self._bias_applied_count == 1:
                                print(f"✓ Graph bias applied (nodes: {len(node_token_indices)}, applied count: {self._bias_applied_count})")
                            if labels is not None and hasattr(outputs, 'loss') and outputs.loss is not None:
                                bias_reg_loss = self._compute_bias_regularization_loss(attn_bias)
                                outputs.loss = outputs.loss + bias_reg_loss
                            return outputs
                        else:
                            self._bias_failed_count += 1
                            if self.debug:
                                print("⚠️ Graph bias preparation failed")
                            outputs = self.lora_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                **kwargs,
                            )
                            return outputs
                    except Exception as e:
                        self._bias_failed_count += 1
                        if self.debug:
                            print(f"⚠️ Graph bias application failed: {e}")
                        import traceback
                        traceback.print_exc()
                        outputs = self.lora_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            **kwargs,
                        )
                        return outputs
                else:
                    self._bias_skipped_count += 1
                    if self.debug or self._bias_skipped_count == 1:
                        print(f"⚠️ Node tokens not found, skipping graph bias application (skipped count: {self._bias_skipped_count})")
                    outputs = self.lora_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        **kwargs,
                    )
                    return outputs
            else:
                self._bias_skipped_count += 1
                if self.debug:
                    if attn_bias is None:
                        print("⚠️ attn_bias is None, skipping graph bias application")
                    else:
                        print("⚠️ input_ids is None, skipping graph bias application")
                outputs = self.lora_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )
                return outputs
        else:
            outputs = self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        return outputs

    def _create_node_token_mapping(self, input_ids):
        """
        Locate token indices that correspond to graph node identifiers in the prompt.

        Scans the ``Graph: ... Answer:`` span for ``N1`` / ``[NODEID.X]`` tokens and
        maps them back to absolute positions in ``input_ids``.
        """
        tokenizer = self.tokenizer
        if tokenizer is None:
            tokenizer = getattr(self.lora_model, "tokenizer", None)
            if tokenizer is None and hasattr(self.lora_model, "get_base_model"):
                base_model = self.lora_model.get_base_model()
                tokenizer = getattr(base_model, "tokenizer", None)
        if tokenizer is None:
            return []
        node_tokens = []
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 2:
                input_ids_list = input_ids[0].cpu().tolist()
            else:
                input_ids_list = input_ids.cpu().tolist()
        else:
            input_ids_list = input_ids
        if isinstance(input_ids_list, list) and len(input_ids_list) > 0:
            if isinstance(input_ids_list[0], list):
                input_ids_list = input_ids_list[0]
            elif isinstance(input_ids_list[0], torch.Tensor):
                input_ids_list = input_ids_list[0].cpu().tolist()
        try:
            input_ids_list = [int(x) for x in input_ids_list]
        except (TypeError, ValueError):
            return []
        full_text = tokenizer.decode(input_ids_list, skip_special_tokens=True)
        import re
        graph_start_marker = "Graph:"
        answer_start_marker = "\nAnswer:"
        graph_start_idx = full_text.find(graph_start_marker)
        answer_start_idx = full_text.find(answer_start_marker)
        if graph_start_idx == -1:
            full_text_lower = full_text.lower()
            alternative_markers = ["graph:", "graph sequence:", "graph sequences:"]
            for marker in alternative_markers:
                marker_idx = full_text_lower.find(marker)
                if marker_idx != -1:
                    graph_start_idx = marker_idx
                    graph_start_idx += len(marker)
                    break
        if graph_start_idx == -1:
            n_format_matches = re.finditer(r'\bN(\d+)\b', full_text)
            for match in n_format_matches:
                node_text = match.group(0)
                positions = self._find_token_positions(input_ids, node_text, tokenizer)
                for pos in positions:
                    if pos not in node_tokens:
                        node_tokens.append(pos)
            node_tokens = sorted(list(set(node_tokens)))
            if not node_tokens:
                print("⚠️ Warning: No node identifiers found (N1/N2/N3 or [NODEID.X] format)")
                print(f"   Text Preview: {full_text[:200]}...")
            return node_tokens
        if answer_start_idx != -1 and answer_start_idx > graph_start_idx:
            graph_text = full_text[graph_start_idx:answer_start_idx].strip()
        else:
            graph_text = full_text[graph_start_idx:].strip()
        prefix_text = full_text[:graph_start_idx]
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        graph_start_token_idx = len(prefix_tokens)
        graph_tokens = tokenizer.encode(graph_text, add_special_tokens=False)
        graph_end_token_idx = graph_start_token_idx + len(graph_tokens)
        n_format_matches = list(re.finditer(r'\bN(\d+)\b', graph_text))
        for match in n_format_matches:
            node_text = match.group(0)
            char_start = match.start()
            char_end = match.end()
            try:
                prefix_chars = graph_text[:char_start]
                prefix_tokens_for_node = tokenizer.encode(prefix_chars, add_special_tokens=False)
                node_token_start = graph_start_token_idx + len(prefix_tokens_for_node)
                node_tokens_encoded = tokenizer.encode(node_text, add_special_tokens=False)
                if node_tokens_encoded:
                    context_start = max(0, node_token_start - 2)
                    context_end = min(len(input_ids_list), node_token_start + len(node_tokens_encoded) + 2)
                    context_tokens = input_ids_list[context_start:context_end]
                    context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
                    if node_text in context_text:
                        if node_token_start not in node_tokens:
                            node_tokens.append(node_token_start)
                        continue
            except:
                pass
            target_tokens = tokenizer.encode(node_text, add_special_tokens=False)
            found = False
            if target_tokens:
                for i in range(len(graph_tokens) - len(target_tokens) + 1):
                    if graph_tokens[i:i+len(target_tokens)] == target_tokens:
                        token_pos = graph_start_token_idx + i
                        if token_pos not in node_tokens:
                            node_tokens.append(token_pos)
                        found = True
                        break
                if not found:
                    for i in range(len(input_ids_list) - len(target_tokens) + 1):
                        if input_ids_list[i:i+len(target_tokens)] == target_tokens:
                            if i not in node_tokens:
                                node_tokens.append(i)
                            found = True
                            break
            if not found:
                try:
                    n_token = tokenizer.encode("N", add_special_tokens=False)
                    num_token = tokenizer.encode(match.group(1), add_special_tokens=False)
                    if n_token and num_token:
                        for i in range(len(graph_tokens) - len(n_token) - len(num_token) + 1):
                            if (graph_tokens[i:i+len(n_token)] == n_token and
                                graph_tokens[i+len(n_token):i+len(n_token)+len(num_token)] == num_token):
                                token_pos = graph_start_token_idx + i
                                if token_pos not in node_tokens:
                                    node_tokens.append(token_pos)
                                found = True
                                break
                        if not found:
                            for i in range(len(input_ids_list) - len(n_token) - len(num_token) + 1):
                                if (input_ids_list[i:i+len(n_token)] == n_token and
                                    input_ids_list[i+len(n_token):i+len(n_token)+len(num_token)] == num_token):
                                    if i not in node_tokens:
                                        node_tokens.append(i)
                                    found = True
                                    break
                except:
                    pass
        if not node_tokens:
            n_format_matches = re.finditer(r'\bN(\d+)\b', graph_text)
            for match in n_format_matches:
                node_text = match.group(0)
                target_tokens = tokenizer.encode(node_text, add_special_tokens=False)
                if target_tokens:
                    for i in range(len(input_ids_list) - len(target_tokens) + 1):
                        if input_ids_list[i:i+len(target_tokens)] == target_tokens:
                            if i not in node_tokens:
                                node_tokens.append(i)
                if not target_tokens or len(target_tokens) == 0:
                    try:
                        n_token = tokenizer.encode("N", add_special_tokens=False)
                        num_token = tokenizer.encode(match.group(1), add_special_tokens=False)
                        if n_token and num_token:
                            for i in range(len(input_ids_list) - len(n_token) - len(num_token) + 1):
                                if (input_ids_list[i:i+len(n_token)] == n_token and
                                    input_ids_list[i+len(n_token):i+len(n_token)+len(num_token)] == num_token):
                                    if i not in node_tokens:
                                        node_tokens.append(i)
                    except:
                        pass
        nodeid_matches = re.finditer(r"\[NODEID\.([A-Z]+)\]", graph_text)
        for match in nodeid_matches:
            node_text = f"[NODEID.{match.group(1)}]"
            target_tokens = tokenizer.encode(node_text, add_special_tokens=False)
            for i in range(len(graph_tokens) - len(target_tokens) + 1):
                if graph_tokens[i:i+len(target_tokens)] == target_tokens:
                    token_pos = graph_start_token_idx + i
                    if token_pos not in node_tokens:
                        node_tokens.append(token_pos)
        if not node_tokens:
            all_node_texts = []
            n_matches = list(re.finditer(r'\bN(\d+)\b', graph_text))
            for match in n_matches:
                all_node_texts.append(match.group(0))
            for node_text in all_node_texts[:20]:
                target_tokens = tokenizer.encode(node_text, add_special_tokens=False)
                if target_tokens:
                    for i in range(len(input_ids_list) - len(target_tokens) + 1):
                        if input_ids_list[i:i+len(target_tokens)] == target_tokens:
                            if i not in node_tokens:
                                node_tokens.append(i)
                            break
                if not target_tokens or len(target_tokens) == 0:
                    try:
                        n_token = tokenizer.encode("N", add_special_tokens=False)
                        num_match = re.match(r'N(\d+)', node_text)
                        if num_match:
                            num_token = tokenizer.encode(num_match.group(1), add_special_tokens=False)
                            if n_token and num_token:
                                for i in range(len(input_ids_list) - len(n_token) - len(num_token) + 1):
                                    if (input_ids_list[i:i+len(n_token)] == n_token and
                                        input_ids_list[i+len(n_token):i+len(n_token)+len(num_token)] == num_token):
                                        if i not in node_tokens:
                                            node_tokens.append(i)
                                        break
                    except:
                        pass
        node_tokens = sorted(list(set(node_tokens)))
        if not node_tokens:
            print("⚠️ Warning: No node identifiers found (N1/N2/N3 or [NODEID.X] format)")
            print(f"   Text Preview: {full_text[:200]}...")
            if graph_start_idx != -1:
                print(f"   Graph sequence part: {graph_text[:200]}...")
        return node_tokens

    def _find_token_positions(self, input_ids, target_text, tokenizer):
        """Return all token start positions where *target_text* appears as a subsequence."""
        positions = []
        target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 2:
                input_ids_list = input_ids[0].cpu().tolist()
            else:
                input_ids_list = input_ids.cpu().tolist()
        else:
            input_ids_list = input_ids
        if isinstance(input_ids_list, list) and len(input_ids_list) > 0:
            if isinstance(input_ids_list[0], list):
                input_ids_list = input_ids_list[0]
            elif isinstance(input_ids_list[0], torch.Tensor):
                input_ids_list = input_ids_list[0].cpu().tolist()
        try:
            input_ids_list = [int(x) for x in input_ids_list]
        except (TypeError, ValueError):
            return positions
        for i in range(len(input_ids_list) - len(target_tokens) + 1):
            if input_ids_list[i : i + len(target_tokens)] == target_tokens:
                positions.append(i)
        return positions
    def _apply_selective_graph_bias(self, outputs, attn_bias, node_token_indices):
        """Post-hoc bias addition on stored attention tensors (legacy path)."""
        attentions = outputs.attentions
        num_nodes = len(node_token_indices)
        if num_nodes == 0:
            return outputs
        if attn_bias.dim() == 5:
            attn_bias = attn_bias[:, 0, :, :, :]
        elif attn_bias.dim() != 4:
            if self.debug:
                print(f"Warning: attn_bias has unexpected shape {attn_bias.shape}, skipping graph bias")
            return outputs
        if attn_bias.shape[-1] != num_nodes:
            if attn_bias.shape[-1] > num_nodes:
                attn_bias = attn_bias[:, :, :num_nodes, :num_nodes]
            else:
                attn_bias = self._expand_bias_matrix(attn_bias, num_nodes)
        base_model = self.lora_model.get_base_model() if hasattr(self.lora_model, "get_base_model") else self.lora_model
        num_heads = base_model.config.num_attention_heads
        if attn_bias.shape[1] != num_heads:
            attn_bias = self.bias_projection(attn_bias.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        enhanced_attentions = []
        for layer_idx, attention in enumerate(attentions):
            bias_addition = torch.zeros_like(attention)
            for i, node_i in enumerate(node_token_indices):
                for j, node_j in enumerate(node_token_indices):
                    if i < attn_bias.shape[-1] and j < attn_bias.shape[-1]:
                        num_heads_to_apply = min(attention.shape[1], attn_bias.shape[1])
                        bias_addition[0, :num_heads_to_apply, node_i, node_j] = (
                            self.bias_weight * attn_bias[0, :num_heads_to_apply, i, j]
                        )
            enhanced_attention = attention + bias_addition
            enhanced_attentions.append(enhanced_attention)
        outputs.attentions = tuple(enhanced_attentions)
        return outputs
    def _prepare_attn_bias(self, attn_bias, node_token_indices, seq_len):
        """Project and slice the node-level bias matrix for the current sequence length."""
        num_nodes = len(node_token_indices)
        if num_nodes == 0:
            return None
        if attn_bias.dim() == 5:
            attn_bias = attn_bias[:, 0, :, :, :]
        elif attn_bias.dim() != 4:
            return None
        if attn_bias.shape[-1] != num_nodes:
            if attn_bias.shape[-1] > num_nodes:
                attn_bias = attn_bias[:, :, :num_nodes, :num_nodes]
            else:
                attn_bias = self._expand_bias_matrix(attn_bias, num_nodes)
        base_model = self.lora_model.get_base_model() if hasattr(self.lora_model, "get_base_model") else self.lora_model
        num_heads = base_model.config.num_attention_heads
        if attn_bias.shape[1] != num_heads:
            attn_bias = self.bias_projection(attn_bias.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        full_bias = torch.zeros(
            attn_bias.shape[0], attn_bias.shape[1], seq_len, seq_len,
            device=attn_bias.device, dtype=attn_bias.dtype
        )
        for i, node_i in enumerate(node_token_indices):
            for j, node_j in enumerate(node_token_indices):
                if i < attn_bias.shape[-1] and j < attn_bias.shape[-1]:
                    full_bias[:, :, node_i, node_j] = (
                        self.bias_weight * attn_bias[:, :, i, j]
                    )
        return full_bias
    def _forward_with_graph_bias_hook(self, input_ids, attention_mask=None, labels=None,
                                      attn_bias=None, node_token_indices=None, **kwargs):
        if attn_bias is None or node_token_indices is None or len(node_token_indices) == 0:
            return self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        base_model = self.lora_model.get_base_model() if hasattr(self.lora_model, "get_base_model") else self.lora_model
        layers = None
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
        elif hasattr(base_model, 'layers'):
            layers = base_model.layers
        if layers is None:
            if self.debug:
                print("⚠️ Cannot find model layers, falling back to no graph bias")
            return self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        if attn_bias.shape[0] != batch_size:
            attn_bias = attn_bias[:1].expand(batch_size, -1, -1, -1)
        if attn_bias.shape[2] != seq_len or attn_bias.shape[3] != seq_len:
            attn_bias = self._expand_attn_bias_to_seq_len(attn_bias, seq_len, node_token_indices)
        hooks = []
        def create_attention_hook(layer_idx, attn_bias_layer, node_token_indices):
            def attention_hook(module, input, output):
                return output
            return attention_hook
        original_attn_methods = []
        def create_attn_wrapper(layer_idx, attn_bias_layer, node_token_indices, seq_len):
            original_attn = None
            def patched_attn(query, key, value, attention_mask=None, head_mask=None, **kwargs):
                nonlocal original_attn
                attn_weights = torch.matmul(query, key.transpose(-2, -1))
                if hasattr(module, 'scale'):
                    attn_weights = attn_weights * module.scale
                else:
                    head_dim = query.size(-1)
                    attn_weights = attn_weights / (head_dim ** 0.5)
                if attn_bias_layer is not None:
                    if attn_bias_layer.shape == attn_weights.shape:
                        attn_bias_masked = torch.zeros_like(attn_weights)
                        for i, node_i in enumerate(node_token_indices):
                            for j, node_j in enumerate(node_token_indices):
                                if node_i < seq_len and node_j < seq_len:
                                    attn_bias_masked[:, :, node_i, node_j] = (
                                        self.bias_weight * attn_bias_layer[:, :, node_i, node_j]
                                    )
                        attn_weights = attn_weights + attn_bias_masked
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = torch.softmax(attn_weights, dim=-1)
                if head_mask is not None:
                    attn_weights = attn_weights * head_mask
                attn_output = torch.matmul(attn_weights, value)
                return attn_output, attn_weights
            return patched_attn, original_attn
        try:
            for layer_idx, layer in enumerate(layers):
                if layer_idx != 0:
                    continue
                if hasattr(layer, 'self_attn'):
                    attention_module = layer.self_attn
                elif hasattr(layer, 'attention'):
                    attention_module = layer.attention
                else:
                    continue
                if hasattr(attention_module, '_attn'):
                    original_attn = attention_module._attn
                    original_attn_methods.append((attention_module, original_attn))
                    def make_wrapper(attn_module, attn_bias_val, node_indices, seq_len_val, orig_attn_method):
                        def wrapped_attn(query, key, value, attention_mask=None, head_mask=None, **kwargs):
                            attn_weights = torch.matmul(query, key.transpose(-2, -1))
                            if hasattr(attn_module, 'scale'):
                                attn_weights = attn_weights * attn_module.scale
                            else:
                                head_dim = query.size(-1)
                                attn_weights = attn_weights / (head_dim ** 0.5)
                            if attn_bias_val is not None and node_indices:
                                attn_bias_masked = torch.zeros_like(attn_weights)
                                for i, node_i in enumerate(node_indices):
                                    for j, node_j in enumerate(node_indices):
                                        if (node_i < attn_weights.shape[2] and
                                            node_j < attn_weights.shape[3] and
                                            i < attn_bias_val.shape[2] and
                                            j < attn_bias_val.shape[3]):
                                            attn_bias_masked[:, :, node_i, node_j] = (
                                                self.bias_weight * attn_bias_val[:, :, node_i, node_j]
                                            )
                                attn_weights = attn_weights + attn_bias_masked
                            if attention_mask is not None:
                                attn_weights = attn_weights + attention_mask
                            attn_weights = torch.softmax(attn_weights, dim=-1)
                            if head_mask is not None:
                                attn_weights = attn_weights * head_mask
                            attn_output = torch.matmul(attn_weights, value)
                            return attn_output, attn_weights
                        return wrapped_attn
                    attention_module._attn = make_wrapper(
                        attention_module, attn_bias, node_token_indices, seq_len, original_attn
                    )
                    if self.debug:
                        print(f"✓ Graph bias applied to first layer (Layer 0)")
                    break
            outputs = self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        except Exception as e:
            if self.debug:
                print(f"Warning: Failed to apply graph bias via hook: {e}")
                import traceback
                traceback.print_exc()
            for module, original_attn in original_attn_methods:
                if hasattr(module, '_attn'):
                    module._attn = original_attn
            outputs = self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            for module, original_attn in original_attn_methods:
                if hasattr(module, '_attn'):
                    module._attn = original_attn
        return outputs
    def _expand_attn_bias_to_seq_len(self, attn_bias, target_seq_len, node_token_indices):
        """Pad or crop the bias matrix when tokenized sequence length differs from num_nodes."""
        if attn_bias.shape[2] == target_seq_len and attn_bias.shape[3] == target_seq_len:
            return attn_bias
        batch_size = attn_bias.shape[0]
        num_heads = attn_bias.shape[1]
        expanded_bias = torch.zeros(
            batch_size, num_heads, target_seq_len, target_seq_len,
            device=attn_bias.device, dtype=attn_bias.dtype
        )
        num_nodes = len(node_token_indices)
        if num_nodes > 0:
            if attn_bias.shape[2] != num_nodes or attn_bias.shape[3] != num_nodes:
                if attn_bias.shape[2] > num_nodes:
                    attn_bias = attn_bias[:, :, :num_nodes, :num_nodes]
                else:
                    attn_bias = self._expand_bias_matrix(attn_bias, num_nodes)
            for i, node_i in enumerate(node_token_indices):
                for j, node_j in enumerate(node_token_indices):
                    if node_i < target_seq_len and node_j < target_seq_len and i < num_nodes and j < num_nodes:
                        expanded_bias[:, :, node_i, node_j] = attn_bias[:, :, i, j]
        return expanded_bias
    def _compute_bias_regularization_loss(self, attn_bias):
        """L2-style regularizer on bias magnitudes to prevent runaway attention shifts."""
        if attn_bias is not None and attn_bias.numel() > 0:
            attn_bias_var = attn_bias.var()
            attn_bias_mean = attn_bias.mean()
            attn_bias_loss = 1e-5 * (attn_bias_var + attn_bias_mean.abs())
            l2_reg_loss = 1e-7 * (
                self.bias_weight ** 2 +
                (F.softplus(self.graph_encoder.structural_equivalence_weight) ** 2).sum() +
                (F.softplus(self.graph_encoder.distance_weight) ** 2).sum()
            )
            return attn_bias_loss + l2_reg_loss
        else:
            return 1e-6 * (
                self.bias_weight ** 2 +
                (F.softplus(self.graph_encoder.structural_equivalence_weight) ** 2).sum() +
                (F.softplus(self.graph_encoder.distance_weight) ** 2).sum()
            )
    def _expand_bias_matrix(self, attn_bias, target_size):
        """Zero-pad a square bias matrix up to target_size x target_size."""
        current_size = attn_bias.shape[-1]
        if current_size >= target_size:
            return attn_bias
        import torch.nn.functional as F
        expanded_bias = F.interpolate(
            attn_bias.view(attn_bias.shape[0], attn_bias.shape[1], -1),
            size=target_size * target_size,
            mode="linear",
            align_corners=False,
        )
        expanded_bias = expanded_bias.view(attn_bias.shape[0], attn_bias.shape[1], target_size, target_size)
        return expanded_bias
    def generate(self, input_ids, attention_mask=None, graph_data=None, **kwargs):
        """Autoregressive generation with optional per-step graph bias injection."""
        if self.fast_generation:
            with torch.no_grad():
                return self.lora_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
        if self.use_graph_bias and graph_data is not None:
            attn_bias = self.graph_encoder.compute_graph_bias(
                graph_data["graph_features"],
                graph_data["edge_index"],
            )
            if attn_bias is not None and input_ids is not None:
                node_token_indices = self._create_node_token_mapping(input_ids)
                if node_token_indices:
                    attn_bias_processed = self._prepare_attn_bias(
                        attn_bias, node_token_indices, input_ids.shape[1]
                    )
                    if attn_bias_processed is not None:
                        return self._generate_with_graph_bias(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            attn_bias=attn_bias_processed,
                            node_token_indices=node_token_indices,
                            graph_data=graph_data,
                            **kwargs,
                        )
        with torch.no_grad():
            return self.lora_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
    def _generate_with_graph_bias(self, input_ids, attention_mask=None, attn_bias=None,
                                  node_token_indices=None, graph_data=None, **kwargs):
        """Token-by-token generation loop with attention hooks active on each step."""
        if attn_bias is None or node_token_indices is None or len(node_token_indices) == 0:
            with torch.no_grad():
                return self.lora_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
        device = input_ids.device
        self.eval()
        batch_size = input_ids.shape[0]
        initial_seq_len = input_ids.shape[1]
        if attn_bias.shape[0] != batch_size:
            attn_bias = attn_bias[:1].expand(batch_size, -1, -1, -1)
        if attn_bias.shape[2] != initial_seq_len or attn_bias.shape[3] != initial_seq_len:
            attn_bias = self._expand_attn_bias_to_seq_len(attn_bias, initial_seq_len, node_token_indices)
        base_model = self.lora_model.get_base_model() if hasattr(self.lora_model, "get_base_model") else self.lora_model
        layers = None
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
        elif hasattr(base_model, 'layers'):
            layers = base_model.layers
        if layers is None:
            with torch.no_grad():
                return self.lora_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
        generated_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None
        eos_token_id = kwargs.get('eos_token_id', self.tokenizer.eos_token_id if self.tokenizer else None)
        pad_token_id = kwargs.get('pad_token_id', self.tokenizer.pad_token_id if self.tokenizer else None)
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        temperature = kwargs.get('temperature', 0.7)
        do_sample = kwargs.get('do_sample', True)
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id if self.tokenizer else 2
        original_attn_methods = []
        try:
            for layer_idx, layer in enumerate(layers):
                if layer_idx != 0:
                    continue
                if hasattr(layer, 'self_attn'):
                    attention_module = layer.self_attn
                elif hasattr(layer, 'attention'):
                    attention_module = layer.attention
                else:
                    continue
                if hasattr(attention_module, '_attn'):
                    original_attn = attention_module._attn
                    original_attn_methods.append((attention_module, original_attn))
                    def make_wrapper(attn_module, attn_bias_val, node_indices, orig_attn_method):
                        def wrapped_attn(query, key, value, attention_mask=None, head_mask=None, **kwargs):
                            attn_weights = torch.matmul(query, key.transpose(-2, -1))
                            if hasattr(attn_module, 'scale'):
                                attn_weights = attn_weights * attn_module.scale
                            else:
                                head_dim = query.size(-1)
                                attn_weights = attn_weights / (head_dim ** 0.5)
                            current_seq_len = attn_weights.shape[2]
                            if attn_bias_val is not None and node_indices:
                                attn_bias_masked = torch.zeros_like(attn_weights)
                                for i, node_i in enumerate(node_indices):
                                    for j, node_j in enumerate(node_indices):
                                        if (node_i < current_seq_len and
                                            node_j < current_seq_len and
                                            i < attn_bias_val.shape[2] and
                                            j < attn_bias_val.shape[3]):
                                            attn_bias_masked[:, :, node_i, node_j] = (
                                                self.bias_weight * attn_bias_val[:, :, node_i, node_j]
                                            )
                                attn_weights = attn_weights + attn_bias_masked
                            if attention_mask is not None:
                                attn_weights = attn_weights + attention_mask
                            attn_weights = torch.softmax(attn_weights, dim=-1)
                            if head_mask is not None:
                                attn_weights = attn_weights * head_mask
                            attn_output = torch.matmul(attn_weights, value)
                            return attn_output, attn_weights
                        return wrapped_attn
                    attention_module._attn = make_wrapper(
                        attention_module, attn_bias, node_token_indices, original_attn
                    )
                    break
            with torch.no_grad():
                for step in range(max_new_tokens):
                    outputs = self.lora_model(
                        input_ids=generated_ids,
                        attention_mask=current_attention_mask,
                        **kwargs,
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
                        new_mask = torch.ones(
                            (current_attention_mask.shape[0], 1),
                            dtype=current_attention_mask.dtype,
                            device=device
                        )
                        current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)
                    if next_token.item() == eos_token_id:
                        break
        finally:
            for module, original_attn in original_attn_methods:
                if hasattr(module, '_attn'):
                    module._attn = original_attn
        return generated_ids
    def get_input_embeddings(self):
        return self.lora_model.get_input_embeddings()
    def get_output_embeddings(self):
        return self.lora_model.get_output_embeddings()
    def resize_token_embeddings(self, new_num_tokens):
        return self.lora_model.resize_token_embeddings(new_num_tokens)
    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save LoRA adapters plus graph bias and encoder bias weight checkpoints."""
        os.makedirs(save_directory, exist_ok=True)
        self.lora_model.save_pretrained(save_directory, safe_serialization=False, *args, **kwargs)
        graph_bias_state = {
            "bias_weight": self.bias_weight.data.clone(),
        }
        if hasattr(self, "bias_projection") and self.bias_projection is not None:
            graph_bias_state["bias_projection"] = self.bias_projection.state_dict()
        graph_bias_path = os.path.join(save_directory, "graph_bias_weights.pt")
        torch.save(graph_bias_state, graph_bias_path)
        print(f"Graph bias weights saved to: {graph_bias_path}")
        if hasattr(self, "graph_encoder") and self.graph_encoder is not None:
            graph_encoder_bias_state = {}
            if hasattr(self.graph_encoder, "structural_equivalence_weight"):
                graph_encoder_bias_state["structural_equivalence_weight"] = self.graph_encoder.structural_equivalence_weight.data.clone()
            if hasattr(self.graph_encoder, "distance_weight"):
                graph_encoder_bias_state["distance_weight"] = self.graph_encoder.distance_weight.data.clone()
            if hasattr(self.graph_encoder, "common_neighbors_weight"):
                graph_encoder_bias_state["common_neighbors_weight"] = self.graph_encoder.common_neighbors_weight.data.clone()
            if hasattr(self.graph_encoder, "clustering_coefficient_weight"):
                graph_encoder_bias_state["clustering_coefficient_weight"] = self.graph_encoder.clustering_coefficient_weight.data.clone()
            if hasattr(self.graph_encoder, "triangle_count_weight"):
                graph_encoder_bias_state["triangle_count_weight"] = self.graph_encoder.triangle_count_weight.data.clone()
            if hasattr(self.graph_encoder, "bias_projection") and self.graph_encoder.bias_projection is not None:
                graph_encoder_bias_state["bias_projection"] = self.graph_encoder.bias_projection.state_dict()
            if graph_encoder_bias_state:
                graph_encoder_bias_path = os.path.join(save_directory, "graph_encoder_bias_weights.pt")
                torch.save(graph_encoder_bias_state, graph_encoder_bias_path)
                print(f"Graph encoder bias weights saved to: {graph_encoder_bias_path}")
    @classmethod
    def from_pretrained(cls, save_directory, lora_model, graph_encoder, tokenizer=None, use_graph_bias=True, debug=False):
        """Restore GraphEnhancedLoRALLM wrapper weights from a training output directory."""
        model = cls(lora_model, graph_encoder, tokenizer=tokenizer, use_graph_bias=use_graph_bias, debug=debug)
        graph_bias_path = os.path.join(save_directory, "graph_bias_weights.pt")
        if os.path.exists(graph_bias_path):
            print(f"Loading graph bias weights: {graph_bias_path}")
            graph_bias_state = torch.load(graph_bias_path, map_location="cpu")
            if "bias_weight" in graph_bias_state:
                model.bias_weight.data = graph_bias_state["bias_weight"].clone()
                model.bias_weight.requires_grad_(True)
                print("✓ bias_weight loaded")
            if "bias_projection" in graph_bias_state and hasattr(model, "bias_projection") and model.bias_projection is not None:
                model.bias_projection.load_state_dict(graph_bias_state["bias_projection"])
                print("✓ bias_projection loaded")
        graph_encoder_bias_path = os.path.join(save_directory, "graph_encoder_bias_weights.pt")
        if os.path.exists(graph_encoder_bias_path):
            print(f"Loading graph encoder bias weights: {graph_encoder_bias_path}")
            graph_encoder_bias_state = torch.load(graph_encoder_bias_path, map_location="cpu")
            if "structural_equivalence_weight" in graph_encoder_bias_state:
                model.graph_encoder.structural_equivalence_weight.data = graph_encoder_bias_state["structural_equivalence_weight"].clone()
                model.graph_encoder.structural_equivalence_weight.requires_grad_(True)
                print("✓ structural_equivalence_weight loaded")
            if "distance_weight" in graph_encoder_bias_state:
                model.graph_encoder.distance_weight.data = graph_encoder_bias_state["distance_weight"].clone()
                model.graph_encoder.distance_weight.requires_grad_(True)
                print("✓ distance_weight loaded")
            if "common_neighbors_weight" in graph_encoder_bias_state:
                model.graph_encoder.common_neighbors_weight.data = graph_encoder_bias_state["common_neighbors_weight"].clone()
                model.graph_encoder.common_neighbors_weight.requires_grad_(True)
                print("✓ common_neighbors_weight loaded")
            if "clustering_coefficient_weight" in graph_encoder_bias_state:
                model.graph_encoder.clustering_coefficient_weight.data = graph_encoder_bias_state["clustering_coefficient_weight"].clone()
                model.graph_encoder.clustering_coefficient_weight.requires_grad_(True)
                print("✓ clustering_coefficient_weight loaded")
            if "triangle_count_weight" in graph_encoder_bias_state:
                model.graph_encoder.triangle_count_weight.data = graph_encoder_bias_state["triangle_count_weight"].clone()
                model.graph_encoder.triangle_count_weight.requires_grad_(True)
                print("✓ triangle_count_weight loaded")
            if "bias_projection" in graph_encoder_bias_state and hasattr(model.graph_encoder, "bias_projection") and model.graph_encoder.bias_projection is not None:
                model.graph_encoder.bias_projection.load_state_dict(graph_encoder_bias_state["bias_projection"])
                print("✓ graph_encoder.bias_projection loaded")
        return model
    def __getattr__(self, name):
        """Delegate unknown attributes to the inner LoRA model."""
        if name in ["base_model", "model"]:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. Use 'lora_model' instead."
            )
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.lora_model, name)
class SimpleGraphDataset(Dataset):
    """
    PyTorch dataset for causal-LM fine-tuning on graph QA prompts.

    Each sample is formatted as Question + Graph + Answer.
    Labels mask the question/graph prefix so loss is computed only on answer tokens.
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 512, include_graph_data: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_graph_data = include_graph_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        """Tokenize one sample; optionally attach raw graph tensors for bias injection."""
        item = self.data[idx]
        question_text = item.get("full_question", item.get("question", ""))
        question_part = f"Question: {question_text}\nGraph: {item['graph_sequence']}\nAnswer: "
        answer_part = item["answer"]
        full_text = question_part + answer_part
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = encoding["input_ids"].clone().squeeze()
        question_tokens = self.tokenizer.encode(
            question_part,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        question_length = len(question_tokens)
        labels[:question_length] = -100
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels,
        }
        if self.include_graph_data and "graph_features" in item and "edge_index" in item:
            result["graph_features"] = item["graph_features"]
            result["edge_index"] = item["edge_index"]
        return result
def load_tasks_dataset(tasks_dir: str = "../tasks", max_samples: int = None, use_n_format: bool = True) -> List[Dict[str, Any]]:
    """Load and concatenate train.json from every subdirectory under tasks_dir."""
    print(f"Loading training data from all tasks in {tasks_dir}...")
    print("  Using only train.json (N1, N2, N3 format)")
    all_data = []
    task_dirs = []
    if not os.path.exists(tasks_dir):
        print(f"❌ Tasks directory does not exist: {tasks_dir}")
        return []
    for item in os.listdir(tasks_dir):
        task_path = os.path.join(tasks_dir, item)
        if os.path.isdir(task_path):
            train_json_path = os.path.join(task_path, "train.json")
            if os.path.exists(train_json_path):
                task_dirs.append(item)
                print(f"  Found task: {item} (using train.json)")
    if not task_dirs:
        print(f"❌ No train.json files found in {tasks_dir}")
        return []
    for task_name in task_dirs:
        train_json_path = os.path.join(tasks_dir, task_name, "train.json")
        try:
            with open(train_json_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
                print(f"  Loaded {task_name} (train.json): {len(task_data)} samples")
                all_data.extend(task_data)
        except Exception as e:
            print(f"  ⚠️ Error loading {task_name}: {e}")
            continue
    print(f"Total loaded {len(all_data)} samples (from {len(task_dirs)} tasks)")
    if max_samples and max_samples < len(all_data):
        random.shuffle(all_data)
        all_data = all_data[:max_samples]
        print(f"Truncated to first {max_samples} samples")
    return all_data
def convert_single_sample_with_timeout_and_bias(
    item: Dict[str, Any],
    graph_encoder: GraphSequenceEncoder,
) -> Dict[str, Any]:
    """Convert one raw JSON graph sample into text + graph tensors for training."""
    try:
        edge_index_data = item["edge_index"]
        num_nodes = item["num_nodes"]
        if isinstance(edge_index_data, list):
            if len(edge_index_data) == 2 and isinstance(edge_index_data[0], list):
                edge_index = torch.tensor(edge_index_data, dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_index_data, dtype=torch.long)
        else:
            edge_index = edge_index_data if isinstance(edge_index_data, torch.Tensor) else torch.tensor(edge_index_data)
        if edge_index.dim() == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t()
        graph = Data(edge_index=edge_index, num_nodes=num_nodes)
        dummy_node_features = generate_fixed_node_features(num_nodes, 64)
        question_text = item["question"]
        if graph_encoder.use_graph_bias:
            graph_sequence, attn_bias = graph_encoder.forward_with_bias(
                dummy_node_features,
                edge_index,
                encoder_type="node_sequence",
                question_text=question_text,
                return_bias=True,
            )
        else:
            graph_sequence = graph_encoder(
                dummy_node_features,
                edge_index,
                encoder_type="node_sequence",
                question_text=question_text,
            )
            attn_bias = None
        lines = graph_sequence.split("\n")
        path_lines = []
        for line in lines:
            if line.startswith("Path") and ":" in line:
                path_part = line.split(":", 1)[1].strip()
                path_lines.append(f"Path {len(path_lines) + 1}: {path_part}")
        if path_lines:
            is_directed = "→" in path_lines[0] if path_lines else True
            edge_symbol = "→" if is_directed else "-"
            graph_type = "directed" if is_directed else "undirected"
            graph_explanation = (
                f"The graph is {graph_type}. Each path shows a sequence of connected nodes, "
                f"where '{edge_symbol}' represents a {'directed' if is_directed else 'undirected'} connection between nodes.\n"
            )
            cleaned_sequence = graph_explanation + "\n".join(path_lines)
        else:
            cleaned_sequence = "No valid paths found in the graph."
        question_text = item.get("full_question", item.get("question", ""))
        result = {
            "graph_sequence": cleaned_sequence,
            "question": question_text,
            "answer": item["answer"],
        }
        result["graph_features"] = dummy_node_features
        result["edge_index"] = edge_index
        return result
    except Exception as e:
        raise e
def convert_persistent_to_training_data_with_bias(
    persistent_data: List[Dict[str, Any]],
    graph_encoder: GraphSequenceEncoder,
) -> List[Dict[str, Any]]:
    """Batch-convert raw task JSON records into encoder-processed training samples."""
    print("Converting persistent dataset to training format (with graph structure bias)...")
    training_data = []
    successful = 0
    failed = 0
    for i, item in enumerate(persistent_data):
        if i % 50 == 0:
            print(f"Processing progress: {i}/{len(persistent_data)} (success: {successful}, failed: {failed})")
        try:
            result = convert_single_sample_with_timeout_and_bias(item, graph_encoder)
            if result is not None:
                training_data.append(result)
                successful += 1
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            failed += 1
            continue
    print(f"Conversion completed: {successful} successful, {failed} failed")
    return training_data
def generate_simple_training_data(num_samples: int = 50, model_config=None):
    """Placeholder for synthetic data generation (not used in the tasks pipeline)."""
    print("⚠️ Current version has removed dependency on graph_qa_dataset.py, generate_simple_training_data returns empty list.")
    return []
def train_enhanced_qwen3_8b(
    use_tasks_dataset: bool = True,
    tasks_dir: str = "../tasks",
    data_dir: str = "./graph_qa_data",
    max_samples: int = 1000,
    use_graph_bias: bool = True,
    model_name: str = "./Qwen3-8B-hf",
):
    """Qwen3-8B with graph bias and trainable bias projection (TAB enhanced mode)."""
    print("=== Enhanced Qwen/Qwen3-8B Fine-tuning (LoRA fine-tuning with graph structure bias) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model_name = "Qwen/Qwen3-8B"
    if os.path.exists(model_name) and os.path.isdir(model_name):
        print(f"✓ Using local model: {model_name}")
        actual_model_name = model_name
    else:
        print(f"⚠️ Local model path does not exist: {model_name}")
        print(f"Will attempt to download model from Hugging Face: {hf_model_name}")
        actual_model_name = hf_model_name
    try:
        if os.path.exists(actual_model_name) and os.path.isdir(actual_model_name):
            tokenizer = AutoTokenizer.from_pretrained(actual_model_name, use_fast=False, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(actual_model_name, use_fast=False, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        print("Using bfloat16 precision")
    else:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using {torch_dtype} precision")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            actual_model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    if not torch.cuda.is_available():
        model = model.to(device)
    elif hasattr(model, "device"):
        if isinstance(model.device, torch.device) and model.device.type == "cpu":
            model = model.to(device)
    elif not hasattr(model, "hf_device_map"):
        model = model.to(device)
    hidden_size = model.config.hidden_size
    vocab_size = tokenizer.vocab_size
    num_attention_heads = model.config.num_attention_heads
    print(f"Model configuration: hidden_size={hidden_size}, vocab_size={vocab_size}, num_heads={num_attention_heads}")
    print("\nConfiguring LoRA parameters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    print(f"\nNumber of trainable parameters: {len(trainable_params)}")
    if trainable_params:
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {total_params:,}")
        for i, p in enumerate(trainable_params[:5]):
            print(f"  Parameter {i}: shape={p.shape}, requires_grad={p.requires_grad}")
    graph_encoder = GraphSequenceEncoder(
        graph_dim=64,
        llm_dim=hidden_size,
        vocab_size=vocab_size,
        max_sequence_length=512,
        use_graph_bias=use_graph_bias,
        num_heads=num_attention_heads,
    )
    if use_graph_bias:
        model = GraphEnhancedLoRALLM(lora_model, graph_encoder, tokenizer=tokenizer, use_graph_bias=True, debug=False)
    else:
        model = lora_model
    if not use_tasks_dataset:
        print("❌ Current version only supports training with tasks dataset (train.json)")
        return None, None
    print("\n1. Loading training data from tasks folder (using only train.json, N1, N2, N3 format)...")
    tasks_data = load_tasks_dataset(tasks_dir, max_samples=max_samples, use_n_format=True)
    if not tasks_data:
        print("❌ Failed to load any tasks data")
        return None, None
    training_data = convert_persistent_to_training_data_with_bias(tasks_data, graph_encoder)
    print("\n2. Creating dataset...")
    if not training_data:
        print("❌ No training data successfully converted, cannot create dataset")
        return None, None
    include_graph_data = use_graph_bias
    dataset = SimpleGraphDataset(training_data, tokenizer, max_length=512, include_graph_data=include_graph_data)
    print(f"Successfully created dataset with {len(dataset)} samples")
    output_dir = "../output/qwen3_8b_finetuned"
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        warmup_steps=30,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="epoch",
        logging_steps=50,
        save_steps=500,
        eval_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,
        learning_rate=2e-4,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        report_to=[],
    )
    if use_graph_bias:
        data_collator = GraphDataCollator(
            tokenizer=tokenizer,
            mlm=False,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    if use_graph_bias:
        trainer = GraphBiasTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    print("\n3. Starting training...")
    trainer.train()
    if use_graph_bias and isinstance(model, GraphEnhancedLoRALLM):
        total = model._bias_applied_count + model._bias_skipped_count + model._bias_failed_count
        if total > 0:
            print(f"\n[Graph Bias Statistics] Applied: {model._bias_applied_count}, Skipped: {model._bias_skipped_count}, Failed: {model._bias_failed_count} (Total: {total})")
            if model._bias_applied_count > 0:
                print(f"✓ Graph bias successfully applied {model._bias_applied_count} times")
            if model._bias_skipped_count > 0:
                print(f"⚠️ Graph bias skipped {model._bias_skipped_count} times (node tokens not found)")
            if model._bias_failed_count > 0:
                print(f"❌ Graph bias failed {model._bias_failed_count} times")
    if use_graph_bias and isinstance(model, GraphEnhancedLoRALLM):
        model.save_pretrained(output_dir)
    else:
        trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining completed! Model saved to {output_dir}")
    return model, tokenizer
def train_simple_qwen3_8b(
    use_tasks_dataset: bool = True,
    tasks_dir: str = "../tasks",
    data_dir: str = "./graph_qa_data",
    max_samples: int = 1000,
    model_name: str = "./Qwen3-8B-hf",
):
    """Qwen3-8B with LoRA only (no graph bias wrapper)."""
    print("=== Simple Qwen/Qwen3-8B Fine-tuning (LoRA fine-tuning) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model_name = "Qwen/Qwen3-8B"
    if os.path.exists(model_name) and os.path.isdir(model_name):
        print(f"✓ Using local model: {model_name}")
        actual_model_name = model_name
    else:
        print(f"⚠️ Local model path does not exist: {model_name}")
        print(f"Will attempt to download model from Hugging Face: {hf_model_name}")
        actual_model_name = hf_model_name
    try:
        if os.path.exists(actual_model_name) and os.path.isdir(actual_model_name):
            tokenizer = AutoTokenizer.from_pretrained(actual_model_name, use_fast=False, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(actual_model_name, use_fast=False, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        print("Using bfloat16 precision")
    else:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Using {torch_dtype} precision")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            actual_model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    if not torch.cuda.is_available():
        model = model.to(device)
    elif hasattr(model, "device"):
        if isinstance(model.device, torch.device) and model.device.type == "cpu":
            model = model.to(device)
    elif not hasattr(model, "hf_device_map"):
        model = model.to(device)
    hidden_size = model.config.hidden_size
    vocab_size = tokenizer.vocab_size
    print(f"Model configuration: hidden_size={hidden_size}, vocab_size={vocab_size}")
    print("\nConfiguring LoRA parameters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    graph_encoder = GraphSequenceEncoder(
        graph_dim=64,
        llm_dim=hidden_size,
        vocab_size=vocab_size,
        max_sequence_length=512,
    )
    if not use_tasks_dataset:
        print("❌ Current version only supports training with tasks dataset (train.json)")
        return None, None
    print("\n1. Loading training data from tasks folder (using only train.json, N1, N2, N3 format)...")
    tasks_data = load_tasks_dataset(tasks_dir, max_samples=max_samples, use_n_format=True)
    if not tasks_data:
        print("❌ Failed to load any tasks data")
        return None, None
    training_data = convert_persistent_to_training_data_with_bias(tasks_data, graph_encoder)
    print("\n2. Creating dataset...")
    dataset = SimpleGraphDataset(training_data, tokenizer, max_length=512)
    output_dir = "../output/simple_qwen3_8b_finetuned"
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=30,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="epoch",
        logging_steps=50,
        save_steps=500,
        eval_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,
        learning_rate=5e-4,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        gradient_accumulation_steps=4,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    print("\n3. Starting training...")
    trainer.train()
    if isinstance(model, GraphEnhancedLoRALLM):
        total = model._bias_applied_count + model._bias_skipped_count + model._bias_failed_count
        if total > 0:
            print(f"\n[Graph Bias Statistics] Applied: {model._bias_applied_count}, Skipped: {model._bias_skipped_count}, Failed: {model._bias_failed_count} (Total: {total})")
            if model._bias_applied_count > 0:
                print(f"✓ Graph bias successfully applied {model._bias_applied_count} times")
            if model._bias_skipped_count > 0:
                print(f"⚠️ Graph bias skipped {model._bias_skipped_count} times (node tokens not found)")
            if model._bias_failed_count > 0:
                print(f"❌ Graph bias failed {model._bias_failed_count} times")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining completed! Model saved to {output_dir}")
    return model, tokenizer
def main():
    """Entry point: run enhanced Qwen/Qwen3-8B fine-tuning with graph bias enabled."""
    model, tokenizer = train_enhanced_qwen3_8b(
        use_tasks_dataset=True,
        tasks_dir="../tasks",
        data_dir="./graph_qa_data",
        max_samples=900,
        use_graph_bias=True,
        model_name="./Qwen3-8B-hf",
    )
    if model is not None and tokenizer is not None:
        print("✅ Qwen/Qwen3-8B enhanced training completed successfully")
    else:
        print("❌ Qwen/Qwen3-8B enhanced training failed")
if __name__ == "__main__":
    main()