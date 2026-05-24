"""
Graph sequence encoder for TAB (Task-adaptive Mixture of Experts with Attentional Bias).

This module converts graph-structured inputs into natural-language path sequences that
LLMs can consume, and optionally computes multi-head attention bias tensors that encode
structural graph properties (distance, common neighbors, clustering, etc.).

The encoder is used during both fine-tuning and inference to bridge raw graph topology
and token-level LLM attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import time

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import to_dense_batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not available")


class GraphSequenceEncoder(nn.Module):
    """
    Encodes graphs as linear path sequences and computes task-adaptive attention bias.

    Two complementary outputs are produced:
      1. Text sequences  – multiple graph paths rendered as ``N1 - N2 - N3`` strings.
      2. Attention bias  – per-head additive bias matrices derived from graph structure.

    Path generation is question-aware: nodes referenced in the question text are used
    as anchors for targeted path search (BFS / all-paths enumeration).
    """

    def __init__(self, graph_dim: int, llm_dim: int, vocab_size: int,
                 max_sequence_length: int = 512,
                 use_graph_bias: bool = False,
                 num_heads: int = 8,
                 max_paths: int = 50,
                 max_path_length: int = 20,
                 max_hops: int = 3,
                 max_sequences: int = 8,
                 search_time_budget_sec: float = 0.0):
        """
        Args:
            graph_dim: Dimensionality of per-node feature vectors.
            llm_dim: Hidden size of the target LLM (reserved for future projections).
            vocab_size: Token vocabulary size of the target LLM.
            max_sequence_length: Maximum token length for encoded sequences.
            use_graph_bias: Whether to compute structural attention bias matrices.
            num_heads: Number of attention heads (must match the LLM config).
            max_paths: Cap on paths returned by all-paths enumeration.
            max_path_length: Maximum hop count for a single enumerated path.
            max_hops: BFS depth limit when expanding from a target node.
            max_sequences: Maximum number of path lines in the output text.
            search_time_budget_sec: Wall-clock timeout for path search (0 = unlimited).
        """
        super().__init__()
        self.graph_dim = graph_dim
        self.llm_dim = llm_dim
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.use_graph_bias = use_graph_bias
        self.num_heads = num_heads
        # Search budget knobs used by path/BFS-based sequence generation.
        self.max_paths = max(1, int(max_paths))
        self.max_path_length = max(2, int(max_path_length))
        self.max_hops = max(1, int(max_hops))
        self.max_sequences = max(1, int(max_sequences))
        self.search_time_budget_sec = max(0.0, float(search_time_budget_sec))
        self.reset_search_stats()
        if self.use_graph_bias:
            self._init_graph_bias_calculator()
            print("✓ The graph structure bias calculator has been enabled.")
        else:
            self.graphormer = None
            self.bias_projection = None
            self.distance_bias = None
            self.centrality_bias = None
        self._init_weights()

    def reset_search_stats(self):
        """Reset counters that track path-search truncation events."""
        self.search_stats = {
            "samples_encoded": 0,
            "find_all_paths_calls": 0,
            "find_all_paths_truncated_by_path_cap": 0,
            "find_all_paths_truncated_by_max_len": 0,
            "find_all_paths_truncated_by_timeout": 0,
            "bfs_calls": 0,
            "bfs_truncated_by_path_cap": 0,
            "bfs_truncated_by_hop_cap": 0,
            "bfs_truncated_by_timeout": 0,
        }

    def get_search_stats(self) -> Dict[str, Any]:
        """Return a copy of the current path-search statistics."""
        return dict(self.search_stats)

    def set_search_budgets(
        self,
        max_paths: Optional[int] = None,
        max_path_length: Optional[int] = None,
        max_hops: Optional[int] = None,
        max_sequences: Optional[int] = None,
        search_time_budget_sec: Optional[float] = None,
    ):
        """Update path-search limits at runtime (e.g. for ablation studies)."""
        if max_paths is not None:
            self.max_paths = max(1, int(max_paths))
        if max_path_length is not None:
            self.max_path_length = max(2, int(max_path_length))
        if max_hops is not None:
            self.max_hops = max(1, int(max_hops))
        if max_sequences is not None:
            self.max_sequences = max(1, int(max_sequences))
        if search_time_budget_sec is not None:
            self.search_time_budget_sec = max(0.0, float(search_time_budget_sec))
    def _init_weights(self):
        """Xavier-uniform init for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    def _init_graph_bias_calculator(self):
        """
        Initialize learnable per-head weights for each structural bias component.

        Five bias channels are combined: structural equivalence, shortest-path
        proximity, common neighbors, clustering coefficient similarity, and triangle count.
        Each channel has its own head-wise weight initialized to a small positive value.
        """
        if not PYG_AVAILABLE:
            print("Warning: PyTorch Geometric not available, graph bias calculator disabled")
            self.graphormer = None
            self.bias_projection = None
            self.distance_bias = None
            self.centrality_bias = None
            return
        self.graphormer = None
        self.bias_projection = None
        self.structural_equivalence_weight = nn.Parameter(torch.ones(self.num_heads) * 0.02)
        self.distance_weight = nn.Parameter(torch.ones(self.num_heads) * 0.14)
        self.common_neighbors_weight = nn.Parameter(torch.ones(self.num_heads) * 0.1)
        self.clustering_coefficient_weight = nn.Parameter(torch.ones(self.num_heads) * 0.09)
        self.triangle_count_weight = nn.Parameter(torch.ones(self.num_heads) * 0.08)
        self.distance_bias = nn.Parameter(torch.randn(1))
        self.centrality_bias = nn.Parameter(torch.randn(1))
    def compute_graph_bias(self, graph_features: torch.Tensor,
                          edge_index: torch.Tensor,
                          batch: torch.Tensor = None) -> torch.Tensor:
        """
        Public entry point for attention bias computation.

        Returns:
            Tensor of shape ``[batch, num_heads, num_nodes, num_nodes]``, or ``None``
            when graph bias is disabled or PyG is unavailable.
        """
        if not self.use_graph_bias:
            return None
        if not PYG_AVAILABLE:
            print("Warning: PyTorch Geometric not available, cannot compute graph bias")
            return None
        try:
            if batch is None:
                batch = torch.zeros(graph_features.size(0), dtype=torch.long, device=graph_features.device)
            data = Data(x=graph_features, edge_index=edge_index, batch=batch)
            attn_bias = self._compute_attention_bias(data)
            return attn_bias
        except Exception as e:
            print(f"Warning: Failed to compute graph bias: {e}")
            import traceback
            traceback.print_exc()
            return None
    def _compute_attention_bias(self, data):
        """
        Assemble the full multi-head attention bias from structural sub-components.

        Steps:
          1. Densify node features via ``to_dense_batch``.
          2. Compute five pairwise bias matrices from ``edge_index``.
          3. Scale each matrix by its learnable per-head coefficient (softplus).
          4. Mask out padding positions using ``real_nodes``.
        """
        if not PYG_AVAILABLE:
            return None
        try:
            if hasattr(self, 'graphormer') and self.graphormer is not None:
                struct_data = self.graphormer(data)
                node_features = struct_data.x
                batch_info = struct_data.batch
            else:
                node_features = data.x
                batch_info = data.batch
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(-1)
            elif node_features.dim() > 2:
                node_features = node_features.view(-1, node_features.shape[-1])
            num_nodes = node_features.size(0)
            if batch_info is None or batch_info.numel() == 0:
                batch_info = torch.zeros(num_nodes, dtype=torch.long, device=node_features.device)
            elif batch_info.numel() != num_nodes:
                if batch_info.numel() < num_nodes:
                    last_batch_id = batch_info[-1].item() if batch_info.numel() > 0 else 0
                    padding = torch.full((num_nodes - batch_info.numel(),), last_batch_id,
                                        dtype=torch.long, device=node_features.device)
                    batch_info = torch.cat([batch_info, padding])
                else:
                    batch_info = batch_info[:num_nodes]
            if batch_info.dim() > 1:
                batch_info = batch_info.squeeze()
            if batch_info.dim() == 0:
                batch_info = batch_info.unsqueeze(0)
            if batch_info.size(0) != num_nodes:
                batch_info = torch.zeros(num_nodes, dtype=torch.long, device=node_features.device)
            x, real_nodes = to_dense_batch(node_features, batch_info, max_num_nodes=num_nodes)
            if x.dim() == 2:
                x = x.unsqueeze(0)
                real_nodes = real_nodes.unsqueeze(0) if real_nodes.dim() == 1 else real_nodes
            elif x.dim() > 3:
                x = x.view(-1, x.shape[-2], x.shape[-1])
        except Exception as e:
            print(f"Warning: Failed to compute attention bias: {e}")
            import traceback
            traceback.print_exc()
            return None
        if x.dim() != 3:
            print(f"Warning: Expected x to be 3D, but got shape {x.shape}")
            return None
        batch_size, seq_len, embed_dim = x.shape
        structural_equivalence_coeffs = F.softplus(self.structural_equivalence_weight).to(x.device)
        distance_coeffs = F.softplus(self.distance_weight).to(x.device)
        common_neighbors_coeffs = F.softplus(self.common_neighbors_weight).to(x.device)
        clustering_coefficient_coeffs = F.softplus(self.clustering_coefficient_weight).to(x.device)
        triangle_count_coeffs = F.softplus(self.triangle_count_weight).to(x.device)
        attn_bias = torch.zeros(batch_size, self.num_heads, seq_len, seq_len,
                               device=x.device, dtype=x.dtype)
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            # --- Structural bias components (each [batch, 1, seq, seq]) ---
            structural_equivalence_bias = self._compute_structural_equivalence_bias(
                data.edge_index, seq_len, real_nodes
            )
            attn_bias += structural_equivalence_bias * structural_equivalence_coeffs.view(1, -1, 1, 1)
            proximity = self._compute_distance_bias(data.edge_index, seq_len, real_nodes)
            attn_bias += proximity * distance_coeffs.view(1, -1, 1, 1)
            common_neighbors_bias = self._compute_common_neighbors_bias(
                data.edge_index, seq_len, real_nodes
            )
            attn_bias += common_neighbors_bias * common_neighbors_coeffs.view(1, -1, 1, 1)
            clustering_coefficient_bias = self._compute_clustering_coefficient_bias(
                data.edge_index, seq_len, real_nodes
            )
            attn_bias += clustering_coefficient_bias * clustering_coefficient_coeffs.view(1, -1, 1, 1)
            triangle_count_bias = self._compute_triangle_count_bias(
                data.edge_index, seq_len, real_nodes
            )
            attn_bias += triangle_count_bias * triangle_count_coeffs.view(1, -1, 1, 1)
        # Zero out bias entries corresponding to padded (non-real) node slots.
        mask = real_nodes.unsqueeze(1).unsqueeze(2)
        attn_bias = attn_bias * mask.unsqueeze(1)
        return attn_bias

    def _compute_distance_bias(self, edge_index, seq_len, real_nodes):
        """
        Shortest-path proximity bias: closer nodes receive higher bias values.

        Disconnected pairs are assigned a small constant (0.01); connected pairs use
        ``1 / (1 + alpha * distance)`` with alpha=0.55.
        """
        batch_size = real_nodes.shape[0]
        distance_matrix = torch.zeros(batch_size, seq_len, seq_len, device=edge_index.device)
        if edge_index.dim() == 1:
            return torch.zeros(batch_size, 1, seq_len, seq_len, device=edge_index.device)
        elif edge_index.dim() == 2:
            if edge_index.size(0) != 2:
                if edge_index.size(1) == 2:
                    edge_index = edge_index.t()
                else:
                    return torch.zeros(batch_size, 1, seq_len, seq_len, device=edge_index.device)
        for b in range(batch_size):
            if edge_index.size(1) > 0:
                dist_matrix = self._compute_shortest_paths(edge_index, seq_len)
                distance_matrix[b] = dist_matrix
        finite_mask = distance_matrix >= 0
        if finite_mask.any():
            max_finite_dist = distance_matrix[finite_mask].max()
            if not isinstance(max_finite_dist, torch.Tensor):
                max_finite_dist = torch.tensor(float(max_finite_dist), device=distance_matrix.device, dtype=distance_matrix.dtype)
            if max_finite_dist.item() == 0:
                max_finite_dist = torch.tensor(10.0, device=distance_matrix.device, dtype=distance_matrix.dtype)
        else:
            max_finite_dist = torch.tensor(10.0, device=distance_matrix.device, dtype=distance_matrix.dtype)
        disconnected_mask = (distance_matrix == -1)
        distance_matrix_processed = torch.where(
            disconnected_mask,
            max_finite_dist + 1,
            distance_matrix
        )
        # Convert hop distance to a proximity score in (0, 1].
        alpha = 0.55
        proximity = 1.0 / (1.0 + alpha * distance_matrix_processed)
        proximity[disconnected_mask] = 0.01
        return proximity.unsqueeze(1)

    def _compute_shortest_paths(self, edge_index, num_nodes):
        """All-pairs shortest path lengths via Floyd-Warshall (-1 = unreachable)."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return torch.full((num_nodes, num_nodes), -1, device=edge_index.device, dtype=torch.float)
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        num_edges = edge_index.size(1)
        for i in range(num_edges):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src < num_nodes and dst < num_nodes:
                adj_matrix[src, dst] = 1
                adj_matrix[dst, src] = 1
        dist_matrix = adj_matrix.clone()
        dist_matrix[dist_matrix == 0] = float('inf')
        for i in range(num_nodes):
            dist_matrix[i, i] = 0
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
        dist_matrix[dist_matrix == float('inf')] = -1
        return dist_matrix

    def _compute_common_neighbors_bias(self, edge_index, seq_len, real_nodes):
        """Jaccard-style normalized common-neighbor count for each node pair."""
        batch_size = real_nodes.shape[0]
        common_neighbors_matrix = torch.zeros(batch_size, seq_len, seq_len, device=edge_index.device)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return torch.zeros(batch_size, 1, seq_len, seq_len, device=edge_index.device)
        num_nodes = seq_len
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src < num_nodes and dst < num_nodes:
                adj_list[src].append(dst)
                adj_list[dst].append(src)
        for i in range(num_nodes):
            neighbors_i = set(adj_list[i])
            for j in range(num_nodes):
                neighbors_j = set(adj_list[j])
                common = len(neighbors_i & neighbors_j)
                max_common = min(len(neighbors_i), len(neighbors_j))
                if max_common > 0:
                    common_neighbors_matrix[0, i, j] = common / max_common
                else:
                    common_neighbors_matrix[0, i, j] = 0.0
        return common_neighbors_matrix.unsqueeze(1)

    def _compute_clustering_coefficient_bias(self, edge_index, seq_len, real_nodes):
        """
        Pairwise clustering-coefficient similarity.

        Nodes with similar local clustering coefficients receive higher bias,
        encouraging attention between structurally analogous positions.
        """
        batch_size = real_nodes.shape[0]
        clustering_similarity_matrix = torch.zeros(batch_size, seq_len, seq_len, device=edge_index.device)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return torch.zeros(batch_size, 1, seq_len, seq_len, device=edge_index.device)
        num_nodes = seq_len
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src < num_nodes and dst < num_nodes:
                adj_list[src].append(dst)
                adj_list[dst].append(src)
        clustering_coefficients = torch.zeros(num_nodes, device=edge_index.device)
        for i in range(num_nodes):
            neighbors_i = set(adj_list[i])
            k_i = len(neighbors_i)
            if k_i <= 1:
                clustering_coefficients[i] = 0.0
            else:
                E_i = 0
                neighbors_list = list(neighbors_i)
                for idx1 in range(len(neighbors_list)):
                    for idx2 in range(idx1 + 1, len(neighbors_list)):
                        n1, n2 = neighbors_list[idx1], neighbors_list[idx2]
                        if n2 in adj_list[n1] or n1 in adj_list[n2]:
                            E_i += 1
                clustering_coefficients[i] = (2.0 * E_i) / (k_i * (k_i - 1))
        max_cc = clustering_coefficients.max()
        if max_cc > 0:
            for i in range(num_nodes):
                for j in range(num_nodes):
                    cc_diff = abs(clustering_coefficients[i] - clustering_coefficients[j])
                    clustering_similarity_matrix[0, i, j] = 1.0 - (cc_diff / max_cc)
        else:
            clustering_similarity_matrix[0, :, :] = 1.0
        return clustering_similarity_matrix.unsqueeze(1)

    def _compute_triangle_count_bias(self, edge_index, seq_len, real_nodes):
        """
        Normalized triangle count between adjacent node pairs.

        For directly connected (i, j), counts shared neighbors (triangles) and
        normalizes by the maximum possible triangles involving that edge.
        """
        batch_size = real_nodes.shape[0]
        triangle_count_matrix = torch.zeros(batch_size, seq_len, seq_len, device=edge_index.device)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return torch.zeros(batch_size, 1, seq_len, seq_len, device=edge_index.device)
        num_nodes = seq_len
        adj_list = [[] for _ in range(num_nodes)]
        adj_set = [set() for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src < num_nodes and dst < num_nodes:
                adj_list[src].append(dst)
                adj_list[dst].append(src)
                adj_set[src].add(dst)
                adj_set[dst].add(src)
        for i in range(num_nodes):
            neighbors_i = set(adj_list[i])
            for j in range(num_nodes):
                if i == j:
                    triangle_count_matrix[0, i, j] = 0.0
                    continue
                neighbors_j = set(adj_list[j])
                if j in neighbors_i:
                    common_neighbors = neighbors_i & neighbors_j
                    triangle_count = len(common_neighbors)
                    max_possible = max(1, min(len(neighbors_i) - 1, len(neighbors_j) - 1))
                    if max_possible > 0:
                        triangle_count_matrix[0, i, j] = triangle_count / max_possible
                    else:
                        triangle_count_matrix[0, i, j] = 0.0
                else:
                    triangle_count_matrix[0, i, j] = 0.0
        return triangle_count_matrix.unsqueeze(1)

    def _compute_structural_equivalence_bias(self, edge_index, seq_len, real_nodes):
        """
        Jaccard index of neighbor sets – measures structural role equivalence.

        Two nodes with identical neighborhoods score 1.0; disjoint neighborhoods score 0.0.
        """
        batch_size = real_nodes.shape[0]
        structural_equivalence_matrix = torch.zeros(batch_size, seq_len, seq_len, device=edge_index.device)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return torch.zeros(batch_size, 1, seq_len, seq_len, device=edge_index.device)
        num_nodes = seq_len
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src < num_nodes and dst < num_nodes:
                adj_list[src].append(dst)
                adj_list[dst].append(src)
        for i in range(num_nodes):
            neighbors_i = set(adj_list[i])
            for j in range(num_nodes):
                neighbors_j = set(adj_list[j])
                intersection = len(neighbors_i & neighbors_j)
                union = len(neighbors_i | neighbors_j)
                if union > 0:
                    structural_equivalence_matrix[0, i, j] = intersection / union
                else:
                    structural_equivalence_matrix[0, i, j] = 1.0 if i == j else 0.0
        return structural_equivalence_matrix.unsqueeze(1)

    def _compute_centrality_bias(self, x, real_nodes):
        """Feature-norm centrality bias (currently unused in the main pipeline)."""
        batch_size, seq_len, embed_dim = x.shape
        centrality = torch.norm(x, dim=-1)
        centrality_bias = centrality.unsqueeze(1) + centrality.unsqueeze(2)
        return centrality_bias.unsqueeze(1)

    def forward_with_bias(self, graph_features: torch.Tensor,
                         edge_index: torch.Tensor,
                         encoder_type: str = 'hybrid',
                         question_text: str = None,
                         return_bias: bool = False) -> tuple:
        """
        Encode graph to text and optionally return the attention bias tensor.

        This is the primary API used by training and testing pipelines.
        """
        attn_bias = None
        if self.use_graph_bias:
            attn_bias = self.compute_graph_bias(graph_features, edge_index)
        graph_sequence_text = self.forward(graph_features, edge_index, encoder_type, question_text)
        if return_bias:
            return graph_sequence_text, attn_bias
        else:
            return graph_sequence_text

    def _extract_target_nodes_from_question(self, question_text: str) -> List[int]:
        """
        Parse node identifiers referenced in the question portion of the prompt.

        Supports two formats:
          - ``N1, N2, ...``  (1-indexed, converted to 0-indexed internally)
          - ``[NODEID.A]``   (letter-sequence encoding for large vocabularies)

        Only the interrogative sentence is scanned to avoid spurious matches in
        instruction text.
        """
        import re
        target_nodes = []
        question_patterns = [
            r'(What\s+[^?]*\?)',
            r'(Are\s+[^?]*\?)',
            r'(How\s+[^?]*\?)',
            r'(Is\s+[^?]*\?)',
            r'(Do\s+[^?]*\?)',
            r'(Does\s+[^?]*\?)',
            r'(Can\s+[^?]*\?)',
            r'(Will\s+[^?]*\?)',
            r'(Should\s+[^?]*\?)',
            r'(Would\s+[^?]*\?)',
            r'(Could\s+[^?]*\?)'
        ]
        question_part = ""
        all_matches = []
        for pattern in question_patterns:
            matches = re.finditer(pattern, question_text, re.IGNORECASE)
            for match in matches:
                all_matches.append(match)
        if all_matches:
            question_part = all_matches[-1].group(1)
        else:
            sentences = re.split(r'[.!?]\s+', question_text)
            for sentence in reversed(sentences):
                if '?' in sentence:
                    question_part = sentence.strip()
                    break
            if not question_part:
                question_part = question_text
        question_part_cleaned = re.sub(r'\([^)]*\)', '', question_part)
        question_part_cleaned = re.sub(r'\[[^\]]*node[^\]]*\]', '', question_part_cleaned, flags=re.IGNORECASE)
        n_format_pattern = r'\bN(\d+)\b'
        n_matches = re.findall(n_format_pattern, question_part_cleaned)
        for match in n_matches:
            node_id = int(match) - 1
            if node_id >= 0:
                target_nodes.append(node_id)
        if not target_nodes:
            nodeid_pattern = r'\[NODEID\.([A-Z]+)\]'
            matches = re.findall(nodeid_pattern, question_part_cleaned)
            for match in matches:
                node_id = self._letter_sequence_to_node_id(match)
                if node_id is not None:
                    target_nodes.append(node_id)
        return target_nodes

    def _letter_sequence_to_node_id(self, letter_seq: str) -> Optional[int]:
        """Map a letter sequence (A, B, ..., AA, AB, ...) to a 0-indexed node id."""
        letter_sequence = []
        for i in range(26):
            letter_sequence.append(chr(ord('A') + i))
        for i in range(26):
            for j in range(26):
                letter_sequence.append(chr(ord('A') + i) + chr(ord('A') + j))
        for i in range(26):
            for j in range(26):
                for k in range(26):
                    letter_sequence.append(chr(ord('A') + i) + chr(ord('A') + j) + chr(ord('A') + k))
        try:
            node_id = letter_sequence.index(letter_seq)
            return node_id
        except ValueError:
            return None

    def _find_shortest_paths(self, adj_list: List[List[int]], start: int, end: int) -> List[List[int]]:
        """BFS enumeration of all shortest paths between *start* and *end*."""
        if start == end:
            return [[start]]
        queue = [(start, [start])]
        visited = set()
        shortest_paths = []
        shortest_length = float('inf')
        while queue:
            current, path = queue.pop(0)
            if len(path) > shortest_length:
                break
            if current == end:
                if len(path) < shortest_length:
                    shortest_length = len(path)
                    shortest_paths = [path[:]]
                elif len(path) == shortest_length:
                    shortest_paths.append(path[:])
                continue
            if current in visited:
                continue
            visited.add(current)
            for neighbor in adj_list[current]:
                if neighbor not in path:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        return shortest_paths

    def _encode_linear_sequences(self, graph_features: torch.Tensor,
                                edge_index: torch.Tensor, question_text: str = None,
                                selected_indices: torch.Tensor = None) -> str:
        """
        Core path-sequence encoder: build human-readable path lines from the graph.

        Strategy depends on how many target nodes were extracted from the question:
          - >= 2 targets: enumerate paths between target pairs, then BFS fallback.
          - 1 target:     BFS expansion from that node, then coverage fill.
          - 0 targets:    coverage-based paths from high-degree nodes.

        Selected paths are deduplicated and diversified via overlap scoring.
        """
        self.search_stats["samples_encoded"] += 1
        num_nodes = graph_features.size(0)
        is_directed = self._is_directed_graph(edge_index)
        edge_symbol = "→" if is_directed else "-"
        adj_list = [[] for _ in range(num_nodes)]
        if is_directed:
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0][i].item(), edge_index[1][i].item()
                if src < num_nodes and dst < num_nodes:
                    adj_list[src].append(dst)
        else:
            processed_edges = set()
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0][i].item(), edge_index[1][i].item()
                if src < num_nodes and dst < num_nodes:
                    edge_key = tuple(sorted([src, dst]))
                    if edge_key not in processed_edges:
                        adj_list[src].append(dst)
                        adj_list[dst].append(src)
                        processed_edges.add(edge_key)
        target_nodes = []
        if question_text:
            target_nodes = self._extract_target_nodes_from_question(question_text)
        sequences = []
        used_paths = set()
        used_seq_strings = set()
        if len(target_nodes) >= 2:
            # --- Case 1: multiple question targets – pair-wise path enumeration ---
            max_sequences = min(self.max_sequences, num_nodes)
            sequences_generated = 0
            pair_candidates: List[Tuple[int, int]] = []
            # Adjacent target pairs first, then all non-adjacent pairs.
            for i in range(len(target_nodes) - 1):
                pair_candidates.append((target_nodes[i], target_nodes[i + 1]))
            for i in range(len(target_nodes)):
                for j in range(i + 2, len(target_nodes)):
                    pair_candidates.append((target_nodes[i], target_nodes[j]))
            for (t0, t1) in pair_candidates:
                if sequences_generated >= max_sequences:
                    break
                all_paths = self._find_all_paths(
                    adj_list,
                    t0,
                    t1,
                    is_directed,
                    max_paths=self.max_paths,
                    max_path_length=self.max_path_length,
                    search_time_budget_sec=self.search_time_budget_sec,
                )
                if not all_paths:
                    continue
                valid_paths = [p for p in all_paths if len(p) > 1 and self._is_valid_path(adj_list, p, is_directed)]
                if not valid_paths:
                    continue
                valid_paths.sort(key=len)
                selected_paths = []
                remaining_paths = valid_paths.copy()
                # Greedily pick paths with minimal overlap to already-selected ones.
                while remaining_paths and len(selected_paths) < (max_sequences - sequences_generated):
                    if not selected_paths:
                        selected_paths.append(remaining_paths[0])
                        remaining_paths.remove(remaining_paths[0])
                    else:
                        current_length = len(remaining_paths[0])
                        same_length_paths = [p for p in remaining_paths if len(p) == current_length]
                        overlap_scores = []
                        for p in same_length_paths:
                            overlap = self._calculate_path_overlap_with_selected(p, selected_paths)
                            overlap_scores.append((overlap, p))
                        overlap_scores.sort(key=lambda x: (x[0], len(x[1])))
                        best_path = overlap_scores[0][1]
                        selected_paths.append(best_path)
                        remaining_paths.remove(best_path)
                for p in selected_paths:
                    if sequences_generated >= max_sequences:
                        break
                    path_str = self._path_to_string(p, edge_symbol, None)
                    if path_str in used_seq_strings:
                        continue
                    sequences.append(path_str)
                    used_seq_strings.add(path_str)
                    used_paths.add(tuple(p))
                    sequences_generated += 1
            # BFS fallback: expand from each target if quota not yet filled.
            if sequences_generated < max_sequences:
                for target in target_nodes:
                    if sequences_generated >= max_sequences:
                        break
                    bfs_paths = self._bfs_paths_within_hops(
                        adj_list=adj_list,
                        start=target,
                        max_hops=self.max_hops,
                        max_paths=max_sequences - sequences_generated,
                        is_directed=is_directed,
                        search_time_budget_sec=self.search_time_budget_sec,
                    )
                    for p in bfs_paths:
                        if sequences_generated >= max_sequences:
                            break
                        if p and len(p) > 1 and self._is_valid_path(adj_list, p, is_directed):
                            path_str = self._path_to_string(p, edge_symbol, None)
                            if path_str in used_seq_strings:
                                continue
                            sequences.append(path_str)
                            used_seq_strings.add(path_str)
                            used_paths.add(tuple(p))
                            sequences_generated += 1
        elif len(target_nodes) == 1:
            # --- Case 2: single target – BFS neighborhood + coverage fill ---
            max_sequences = min(self.max_sequences, num_nodes)
            sequences_generated = 0
            target = target_nodes[0]
            bfs_paths = self._bfs_paths_within_hops(
                adj_list=adj_list,
                start=target,
                max_hops=self.max_hops,
                max_paths=max_sequences,
                is_directed=is_directed,
                search_time_budget_sec=self.search_time_budget_sec,
            )
            for p in bfs_paths:
                if sequences_generated >= max_sequences:
                    break
                if p and len(p) > 1 and self._is_valid_path(adj_list, p, is_directed):
                    path_str = self._path_to_string(p, edge_symbol, None)
                    if path_str in used_seq_strings:
                        continue
                    sequences.append(path_str)
                    used_seq_strings.add(path_str)
                    used_paths.add(tuple(p))
                    sequences_generated += 1
            if sequences_generated < max_sequences:
                fill_seqs = self._generate_sequences_without_targets(
                    adj_list, num_nodes, is_directed, edge_symbol, max_sequences=min(8, num_nodes)
                )
                for s in fill_seqs:
                    if sequences_generated >= max_sequences:
                        break
                    if s in used_seq_strings:
                        continue
                    sequences.append(s)
                    used_seq_strings.add(s)
                    sequences_generated += 1
        else:
            # --- Case 3: no targets – coverage paths from high-degree nodes ---
            sequences = self._generate_sequences_without_targets(
                adj_list, num_nodes, is_directed, edge_symbol, max_sequences=min(8, num_nodes)
            )
            used_seq_strings.update(sequences)
        if sequences:
            graph_sequence_text = f"Graph sequences ({'directed' if is_directed else 'undirected'}):\n"
            graph_sequence_text += f"Edge symbol '{edge_symbol}' means {'directed connection' if is_directed else 'undirected connection'}.\n"
            for i, seq in enumerate(sequences):
                graph_sequence_text += f"Path {i+1}: {seq}\n"
        else:
            graph_sequence_text = "No valid paths found in the graph."
        return graph_sequence_text

    def _bfs_paths_within_hops(
        self,
        adj_list: List[List[int]],
        start: int,
        max_hops: int = 3,
        max_paths: int = 5,
        is_directed: bool = False,
        search_time_budget_sec: float = 0.0,
    ) -> List[List[int]]:
        """
        Breadth-first enumeration of simple paths within *max_hops* from *start*.

        Returns up to *max_paths* distinct paths; respects the wall-clock budget when set.
        """
        self.search_stats["bfs_calls"] += 1
        if start is None or start < 0 or start >= len(adj_list):
            return []
        if max_hops <= 0 or max_paths <= 0:
            return []
        from collections import deque
        queue = deque()
        queue.append((start, [start], 0))
        visited_depth: Dict[int, int] = {start: 0}
        results: List[List[int]] = []
        seen_paths = set()
        start_t = time.perf_counter()
        hit_hop_cap = False
        while queue and len(results) < max_paths:
            if search_time_budget_sec > 0 and (time.perf_counter() - start_t) >= search_time_budget_sec:
                self.search_stats["bfs_truncated_by_timeout"] += 1
                break
            node, path, depth = queue.popleft()
            if 0 < depth <= max_hops:
                path_t = tuple(path)
                if path_t not in seen_paths:
                    results.append(path)
                    seen_paths.add(path_t)
                    if len(results) >= max_paths:
                        break
            if depth >= max_hops:
                hit_hop_cap = True
                continue
            if node >= len(adj_list):
                continue
            for nbr in adj_list[node]:
                if nbr in path:
                    continue
                next_depth = depth + 1
                if nbr in visited_depth and visited_depth[nbr] <= next_depth:
                    continue
                visited_depth[nbr] = next_depth
                queue.append((nbr, path + [nbr], next_depth))
        if len(results) >= max_paths:
            self.search_stats["bfs_truncated_by_path_cap"] += 1
        if hit_hop_cap:
            self.search_stats["bfs_truncated_by_hop_cap"] += 1
        return results

    def _is_directed_graph(self, edge_index: torch.Tensor) -> bool:
        """
        Heuristic directed-graph detection.

        If any edge lacks a reverse counterpart, the graph is treated as directed.
        """
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0][i].item(), edge_index[1][i].item()
            has_reverse = ((edge_index[0] == dst) & (edge_index[1] == src)).any()
            if not has_reverse:
                return True
        return False

    def _find_all_paths(
        self,
        adj_list: List[List[int]],
        start: int,
        end: int,
        is_directed: bool,
        max_paths: int = 10,
        max_path_length: int = 20,
        search_time_budget_sec: float = 0.0,
    ) -> List[List[int]]:
        """
        Enumerate simple paths from *start* to *end* via BFS, sorted by length.

        Search is bounded by *max_paths*, *max_path_length*, and optionally a
        wall-clock timeout.  Returns at most *max_paths* shortest paths.
        """
        self.search_stats["find_all_paths_calls"] += 1
        if start == end:
            return [[start]]
        if start >= len(adj_list) or end >= len(adj_list):
            return []
        all_paths = []
        queue = [(start, [start])]
        visited_paths = set()
        start_t = time.perf_counter()
        hit_max_len = False
        while queue:
            if search_time_budget_sec > 0 and (time.perf_counter() - start_t) >= search_time_budget_sec:
                self.search_stats["find_all_paths_truncated_by_timeout"] += 1
                break
            current, path = queue.pop(0)
            if current == end:
                path_tuple = tuple(path)
                if path_tuple not in visited_paths:
                    all_paths.append(path[:])
                    visited_paths.add(path_tuple)
                    if len(all_paths) >= max_paths * 5:
                        self.search_stats["find_all_paths_truncated_by_path_cap"] += 1
                        break
                continue
            if len(path) >= max_path_length:
                hit_max_len = True
                continue
            if len(all_paths) >= max_paths * 5:
                self.search_stats["find_all_paths_truncated_by_path_cap"] += 1
                break
            if current < len(adj_list):
                for neighbor in adj_list[current]:
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
        if hit_max_len:
            self.search_stats["find_all_paths_truncated_by_max_len"] += 1
        all_paths.sort(key=len)
        return all_paths[:max_paths]

    def _generate_sequences_without_targets(self, adj_list: List[List[int]], num_nodes: int,
                                           is_directed: bool, edge_symbol: str,
                                           max_sequences: int = 8) -> List[str]:
        """
        Build coverage paths when no question targets are available.

        Starts from high-degree nodes and greedily extends to cover unvisited nodes/edges.
        """
        sequences = []
        used_paths = set()
        visited_nodes = set()
        visited_edges = set()
        node_degrees = [len(neighbors) for neighbors in adj_list]
        sorted_nodes = sorted(range(num_nodes), key=lambda x: node_degrees[x], reverse=True)
        used_starts = set()
        for start_node in sorted_nodes:
            if len(sequences) >= max_sequences:
                break
            if start_node in used_starts:
                continue
            path = self._generate_coverage_path(adj_list, start_node, visited_nodes, visited_edges,
                                               is_directed, max_length=12)
            if path and len(path) > 1:
                if self._is_valid_path(adj_list, path, is_directed):
                    path_str = self._path_to_string(path, edge_symbol, None)
                    sequences.append(path_str)
                    used_paths.add(tuple(path))
                    used_starts.add(start_node)
                    visited_nodes.update(path)
                    for i in range(len(path) - 1):
                        edge = tuple(sorted([path[i], path[i+1]]))
                        visited_edges.add(edge)
        if len(sequences) < max_sequences:
            unvisited_nodes = set(range(num_nodes)) - visited_nodes
            if unvisited_nodes:
                unvisited_sorted = sorted(unvisited_nodes, key=lambda x: node_degrees[x], reverse=True)
                for start_node in unvisited_sorted[:max_sequences - len(sequences)]:
                    if len(sequences) >= max_sequences:
                        break
                    path = self._generate_coverage_path(adj_list, start_node, visited_nodes, visited_edges,
                                                       is_directed, max_length=10)
                    if path and len(path) > 1:
                        if self._is_valid_path(adj_list, path, is_directed):
                            path_str = self._path_to_string(path, edge_symbol, None)
                            sequences.append(path_str)
                            used_paths.add(tuple(path))
                            visited_nodes.update(path)
                            for i in range(len(path) - 1):
                                edge = tuple(sorted([path[i], path[i+1]]))
                                visited_edges.add(edge)
        return sequences

    def _generate_coverage_path(self, adj_list: List[List[int]], start_node: int,
                               visited_nodes: set, visited_edges: set,
                               is_directed: bool, max_length: int = 12) -> List[int]:
        """
        Greedily extend a walk from *start_node*, preferring unvisited nodes/edges.

        Returns the longest valid path found within *max_length* hops.
        """
        if start_node >= len(adj_list):
            return []
        queue = [(start_node, [start_node])]
        visited = {start_node}
        best_path = [start_node]
        while queue and len(queue[0][1]) < max_length:
            node, path = queue.pop(0)
            if node >= len(adj_list):
                continue
            neighbors = adj_list[node]
            if not neighbors:
                continue
            unvisited_neighbors = [n for n in neighbors if n not in visited_nodes and n not in visited]
            visited_neighbors = [n for n in neighbors if n in visited_nodes and n not in visited]
            candidates = unvisited_neighbors + visited_neighbors
            prioritized_candidates = []
            other_candidates = []
            for neighbor in candidates:
                if neighbor in visited:
                    continue
                edge = tuple(sorted([node, neighbor]))
                if edge not in visited_edges:
                    prioritized_candidates.append(neighbor)
                else:
                    other_candidates.append(neighbor)
            for neighbor in prioritized_candidates[:2]:
                if neighbor not in visited and len(path) < max_length:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
                    visited.add(neighbor)
                    if len(new_path) > len(best_path):
                        best_path = new_path
            if not prioritized_candidates:
                for neighbor in other_candidates[:1]:
                    if neighbor not in visited and len(path) < max_length:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
                        visited.add(neighbor)
                        if len(new_path) > len(best_path):
                            best_path = new_path
                        break
        return best_path if len(best_path) > 1 else []

    def _select_start_candidates(self, adj_list: List[List[int]], target_nodes: List[int],
                               used_paths: set) -> List[int]:
        """Pick diverse start nodes at moderate distance from question targets."""
        num_nodes = len(adj_list)
        candidates = []
        distances = {}
        for node in range(num_nodes):
            min_dist = float('inf')
            for target in target_nodes:
                dist = self._calculate_distance(adj_list, node, target)
                min_dist = min(min_dist, dist)
            distances[node] = min_dist
        distance_ranges = [(2, 3), (1, 4), (1, 5), (0, 6)]
        for min_dist, max_dist in distance_ranges:
            for node in range(num_nodes):
                if min_dist <= distances[node] <= max_dist:
                    in_used_paths = any(node in path for path in used_paths)
                    if not in_used_paths and node not in candidates:
                        candidates.append(node)
            if len(candidates) >= 3:
                break
        if len(candidates) < 3:
            for node in range(num_nodes):
                if node not in candidates and not any(node in path for path in used_paths):
                    candidates.append(node)
        return candidates[:5]

    def _calculate_distance(self, adj_list: List[List[int]], start: int, end: int) -> int:
        """Unweighted BFS hop distance between two nodes (inf if unreachable)."""
        if start == end:
            return 0
        if start >= len(adj_list) or end >= len(adj_list):
            return float('inf')
        queue = [(start, 0)]
        visited = {start}
        while queue:
            node, dist = queue.pop(0)
            if node < len(adj_list):
                for neighbor in adj_list[node]:
                    if neighbor == end:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return float('inf')

    def _generate_path_from_node(self, adj_list: List[List[int]], start_node: int,
                               used_paths: set, max_length: int = 10, target_nodes: List[int] = None) -> List[int]:
        """BFS path generation from a single start, avoiding subpaths of already-used paths."""
        if target_nodes is None:
            target_nodes = []
        visited = set()
        queue = [(start_node, [start_node])]
        valid_paths = []
        while queue and len(queue[0][1]) < max_length:
            node, path = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            current_path = tuple(path)
            is_subpath = any(self._is_subpath(current_path, used_path) for used_path in used_paths)
            if is_subpath:
                continue
            if len(path) > 1:
                valid_paths.append(path)
            if node < len(adj_list):
                neighbors = adj_list[node]
                if target_nodes:
                    target_neighbors = [n for n in neighbors if n in target_nodes and n not in visited and len(path) < max_length]
                    other_neighbors = [n for n in neighbors if n not in target_nodes and n not in visited and len(path) < max_length]
                    for neighbor in target_neighbors:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
                    for neighbor in other_neighbors:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
                        break
                else:
                    for neighbor in neighbors:
                        if neighbor not in visited and len(path) < max_length:
                            new_path = path + [neighbor]
                            queue.append((neighbor, new_path))
                            break
        if valid_paths:
            return max(valid_paths, key=len)
        return []

    def _is_subpath(self, path: tuple, used_path: tuple) -> bool:
        """Return True if *path* appears as a contiguous subsequence of *used_path*."""
        if len(path) > len(used_path):
            return False
        for i in range(len(used_path) - len(path) + 1):
            if used_path[i:i+len(path)] == path:
                return True
        return False

    def _path_to_string(self, path: List[int], edge_symbol: str, selected_indices: torch.Tensor = None) -> str:
        """
        Render a node-id path as a token string, e.g. ``N1-N3-N5`` or ``N1→N3→N5``.

        Node ids are 1-indexed in the output (``N{node_id + 1}``).
        """
        if not path:
            return ""
        node_strings = []
        for node_id in path:
            if selected_indices is not None:
                original_node_id = selected_indices[node_id].item()
                node_strings.append(f"N{original_node_id + 1}")
            else:
                node_strings.append(f"N{node_id + 1}")
        return edge_symbol.join(node_strings)

    def _is_valid_path(self, adj_list: List[List[int]], path: List[int], is_directed: bool) -> bool:
        """Check that every consecutive pair in *path* is connected by an edge."""
        if len(path) < 2:
            return True
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            has_edge = next_node in adj_list[current]
            if not has_edge and not is_directed:
                has_edge = current in adj_list[next_node]
            if not has_edge:
                return False
        return True

    def _calculate_path_overlap(self, path1: List[int], path2: List[int]) -> float:
        """
        Weighted overlap between two paths (0 = disjoint, 1 = identical).

        Combines middle-node Jaccard overlap (60%) and edge-set Jaccard overlap (40%).
        """
        if not path1 or not path2:
            return 0.0
        if len(path1) <= 2:
            middle_nodes1 = set()
        else:
            middle_nodes1 = set(path1[1:-1])
        if len(path2) <= 2:
            middle_nodes2 = set()
        else:
            middle_nodes2 = set(path2[1:-1])
        if middle_nodes1 or middle_nodes2:
            node_overlap = len(middle_nodes1 & middle_nodes2) / max(len(middle_nodes1 | middle_nodes2), 1)
        else:
            node_overlap = 0.0
        edges1 = set()
        for i in range(len(path1) - 1):
            edges1.add((path1[i], path1[i + 1]))
        edges2 = set()
        for i in range(len(path2) - 1):
            edges2.add((path2[i], path2[i + 1]))
        edge_overlap = len(edges1 & edges2) / max(len(edges1 | edges2), 1) if (edges1 | edges2) else 0.0
        overlap = 0.6 * node_overlap + 0.4 * edge_overlap
        return overlap

    def _calculate_path_overlap_with_selected(self, path: List[int], selected_paths: List[List[int]]) -> float:
        """Mean overlap of *path* against all already-selected paths."""
        if not selected_paths:
            return 0.0
        total_overlap = 0.0
        for selected_path in selected_paths:
            total_overlap += self._calculate_path_overlap(path, selected_path)
        return total_overlap / len(selected_paths)

    def _encode_node_sequence(self, graph_features: torch.Tensor,
                             edge_index: torch.Tensor, question_text: str = None,
                             selected_indices: torch.Tensor = None) -> str:
        """Alias for ``_encode_linear_sequences`` (primary encoding mode)."""
        return self._encode_linear_sequences(graph_features, edge_index, question_text, selected_indices)

    def _encode_structure_description(self, graph_features: torch.Tensor,
                                    edge_index: torch.Tensor) -> str:
        """Return a one-line directed/undirected graph type label."""
        is_directed = self._is_directed_graph(edge_index)
        structure_text = f"Graph type: {'Directed' if is_directed else 'Undirected'}\n"
        return structure_text

    def forward(self, graph_features: torch.Tensor, edge_index: torch.Tensor,
                encoder_type: str = 'hybrid', question_text: str = None) -> str:
        """
        Main forward pass: produce a text representation of the graph.

        Args:
            graph_features: Node feature matrix ``[num_nodes, graph_dim]``.
            edge_index: COO edge list ``[2, num_edges]``.
            encoder_type: One of ``'node_sequence'``, ``'structure_description'``, or ``'hybrid'``.
            question_text: Optional question string for target-node extraction.

        Returns:
            A multi-line string of graph path sequences ready for LLM input.
        """
        target_nodes = []
        if question_text:
            target_nodes = self._extract_target_nodes_from_question(question_text)
        if encoder_type == 'node_sequence':
            return self._encode_node_sequence(graph_features, edge_index, question_text, None)
        elif encoder_type == 'structure_description':
            return self._encode_structure_description(graph_features, edge_index)
        elif encoder_type == 'hybrid':
            node_seq_text = self._encode_node_sequence(graph_features, edge_index, question_text, None)
            struct_desc_text = self._encode_structure_description(graph_features, edge_index)
            combined_text = f"{struct_desc_text}\n\n{node_seq_text}"
            return combined_text
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")