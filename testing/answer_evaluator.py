"""
Task-aware answer evaluation for TAB graph reasoning benchmarks.

Provides text normalization, node-id extraction, graph structure parsing, and
per-task comparison logic used by the ``testing/test_*_finetuned.py`` scripts.

Supported tasks: bipartite_check, common_neighbors, connectivity, cycle_detection,
graph_diameter, hamiltonian_path, max_flow, shortest_path, topological_sort.
"""

import re
import torch


def extract_first_sentence(text):
    """
    Return the first sentence from *text*, respecting brackets and quoted strings.

    Stops at the first ``.!?`` that is not inside ``[]`` or quotes, so list
    answers like ``[N1 -> N2 -> N3]`` are not truncated prematurely.
    """
    if not text:
        return text
    text = text.strip()
    if not text:
        return text
    first_line = text.split('\n')[0]
    bracket_depth = 0
    in_single_quote = False
    in_double_quote = False
    i = 0
    while i < len(first_line):
        char = first_line[i]
        if char == '\\' and i + 1 < len(first_line):
            i += 2
            continue
        if char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth = max(0, bracket_depth - 1)
        elif bracket_depth == 0:
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
        if char in '.!?' and bracket_depth == 0 and not in_single_quote and not in_double_quote:
            return first_line[:i+1].strip()
        i += 1
    return first_line.strip()


def normalize_text(text):
    """Lowercase, strip, and collapse whitespace for fuzzy string comparison."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_node_ids(text):
    """
    Parse node identifiers from free-form answer text.

    Supports:
      - ``N1, N2, ...`` (1-indexed, returned as 0-indexed strings)
      - ``[NODEID.A]`` letter encoding
      - Numeric paths ``1 -> 2`` and comma-separated lists ``[1, 2, 3]``

    Returns sorted unique node ids as strings, or ``[]`` if none found.
    """
    nodes = []
    n_format_matches = re.findall(r'\bN(\d+)\b', text, re.IGNORECASE)
    for match in n_format_matches:
        node_id = int(match) - 1
        if node_id >= 0:
            nodes.append(str(node_id))
    nodeid_matches = re.findall(r'\[NODEID\.([^\]]+)\]', text, re.IGNORECASE)
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
    for letter_seq in nodeid_matches:
        try:
            node_id = letter_sequence.index(letter_seq)
            nodes.append(str(node_id))
        except ValueError:
            pass
    path_patterns = [
        r'\b(\d+)\s*[→-]\s*(\d+)',
        r'\[(\d+)\s*[→-]\s*(\d+)\]',
    ]
    for pattern in path_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                nodes.extend(match)
            else:
                nodes.append(match)
    list_patterns = [
        r'\[(\d+(?:\s*,\s*\d+)+)\]',
        r'\((\d+(?:\s*,\s*\d+)+)\)',
    ]
    for pattern in list_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            numbers = re.findall(r'\d+', match)
            nodes.extend(numbers)
    if nodes:
        return sorted(set(str(n) for n in nodes))
    return []


def extract_key_info(text, task_type):
    """
    Extract the task-specific semantic content from an answer string.

    Returns a normalized representation used for equality checking, e.g.:
      - ``"yes"`` / ``"no"`` for boolean tasks
      - integer for numeric tasks (diameter, max flow, path length)
      - sorted node list for sequence tasks
      - ``"has_cycle"`` when topological sort is impossible
    """
    if not text:
        return None
    text = text.strip()
    # --- Boolean yes/no tasks ---
    if task_type == "bipartite_check":
        text_lower = text.lower()
        has_yes = bool(re.search(r'\byes\b', text_lower))
        has_no = bool(re.search(r'\bno\b', text_lower))
        if has_yes and not has_no:
            return "yes"
        elif has_no:
            return "no"
        return None
    elif task_type == "common_neighbors":
        # Extract neighbor list after "are:" or from bracket-enclosed node lists.
        if 'no common neighbors' in text.lower() or 'no common' in text.lower():
            return "no_common"
        if 'are:' in text.lower():
            match = re.search(r'are:\s*', text, re.IGNORECASE)
            if match:
                answer_part = text[match.end():]
                bracket_count = 0
                list_start = -1
                for i, char in enumerate(answer_part):
                    if char == '[':
                        if bracket_count == 0:
                            list_start = i
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0 and list_start != -1:
                            list_content = answer_part[list_start:i+1]
                            nodes = extract_node_ids(list_content)
                            if nodes:
                                return sorted(set(nodes))
                            break
                end_match = re.search(r'[.\n]', answer_part)
                if end_match:
                    answer_part = answer_part[:end_match.start()]
                nodes = extract_node_ids(answer_part)
                if nodes:
                    return sorted(set(nodes))
        nodes = extract_node_ids(text)
        if nodes:
            return sorted(set(nodes))
        return None
    elif task_type == "connectivity":
        text_lower = text.lower()
        has_yes = bool(re.search(r'\byes\b', text_lower))
        has_no = bool(re.search(r'\bno\b', text_lower))
        if has_yes and not has_no:
            return "yes"
        elif has_no:
            return "no"
        return None
    elif task_type == "cycle_detection":
        text_lower = text.lower()
        has_yes = bool(re.search(r'\byes\b', text_lower))
        has_no = bool(re.search(r'\bno\b', text_lower))
        if has_yes and not has_no:
            return "yes"
        elif has_no:
            return "no"
        return None
    elif task_type == "graph_diameter":
        # Use the last integer in the answer as the diameter value.
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[-1])
        return None
    elif task_type == "hamiltonian_path":
        if bool(re.search(r'\bno\b', text.lower())) or 'does not have' in text.lower():
            return "no"
        nodes = extract_node_ids(text)
        if nodes:
            return sorted(set(nodes))
        elif 'yes' in text.lower():
            return "yes"
        return None
    elif task_type == "max_flow":
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[-1])
        return None
    elif task_type == "shortest_path":
        # Prefer explicit "length N"; otherwise infer from path node count or last number.
        length_match = re.search(r'length\s+(\d+)', text, re.IGNORECASE)
        if length_match:
            return int(length_match.group(1))
        nodes = extract_node_ids(text)
        if nodes and ('->' in text or '→' in text or '-' in text):
            return len(nodes) - 1
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[-1])
        return None
    elif task_type == "topological_sort":
        # Detect cycle impossibility; otherwise preserve node order along path arrows.
        if 'cycle' in text.lower() or 'cannot be topologically sorted' in text.lower():
            return "has_cycle"
        nodes = extract_node_ids(text)
        if nodes:
            if '→' in text or '->' in text or '-' in text:
                path_nodes = []
                n_path_pattern = r'\bN(\d+)\b'
                n_matches = re.finditer(n_path_pattern, text, re.IGNORECASE)
                for match in n_matches:
                    node_id = int(match.group(1)) - 1
                    if node_id >= 0:
                        path_nodes.append(str(node_id))
                if not path_nodes:
                    path_pattern = r'(?:\[NODEID\.([^\]]+)\]|\b(\d+)\b)'
                    matches = re.finditer(path_pattern, text)
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
                    for match in matches:
                        if match.group(1):
                            try:
                                node_id = letter_sequence.index(match.group(1))
                                path_nodes.append(str(node_id))
                            except ValueError:
                                pass
                        elif match.group(2):
                            path_nodes.append(str(match.group(2)))
                if path_nodes:
                    return path_nodes
            return nodes
        return None
    return None


def build_edge_set_from_edge_index(edge_index):
    """Build an undirected edge set ``{(u, v), ...}`` from a COO ``edge_index`` tensor or list."""
    edges = set()
    if edge_index is None:
        return edges
    if isinstance(edge_index, torch.Tensor):
        if edge_index.dim() == 2 and edge_index.shape[0] == 2:
            source_nodes = edge_index[0].cpu().numpy()
            target_nodes = edge_index[1].cpu().numpy()
        else:
            return edges
    elif isinstance(edge_index, (list, tuple)) and len(edge_index) >= 2:
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
    else:
        return edges
    for i in range(len(source_nodes)):
        node1 = str(source_nodes[i])
        node2 = str(target_nodes[i])
        # Store undirected edges in canonical sorted order.
        edge = tuple(sorted([node1, node2]))
        edges.add(edge)
    return edges


def build_edge_set_from_question(question):
    """Parse undirected edges from ``(i, j)`` pairs embedded in the question text."""
    edges = set()
    if not question:
        return edges
    edge_pattern = r'\((\d+)\s*,\s*(\d+)\)'
    matches = re.findall(edge_pattern, question)
    for match in matches:
        node1 = str(match[0])
        node2 = str(match[1])
        edge = tuple(sorted([node1, node2]))
        edges.add(edge)
    return edges


def build_directed_edges_from_edge_index(edge_index):
    """Build a directed edge list ``[(u, v), ...]`` from a COO ``edge_index``."""
    edges = []
    if edge_index is None:
        return edges
    if isinstance(edge_index, torch.Tensor):
        if edge_index.dim() == 2 and edge_index.shape[0] == 2:
            source_nodes = edge_index[0].cpu().numpy()
            target_nodes = edge_index[1].cpu().numpy()
        else:
            return edges
    elif isinstance(edge_index, (list, tuple)) and len(edge_index) >= 2:
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
    else:
        return edges
    for i in range(len(source_nodes)):
        u = str(source_nodes[i])
        v = str(target_nodes[i])
        edges.append((u, v))
    return edges


def build_directed_edges_from_question(question):
    """
    Parse directed edges from the question text.

    Tries ``(u -> v)`` / ``(u → v)`` first, then falls back to ``(u, v)`` pairs.
    """
    edges = []
    if not question:
        return edges
    directed_pattern = r'\((\d+)\s*(?:->|→)\s*(\d+)\)'
    matches = re.findall(directed_pattern, question)
    for match in matches:
        u = str(match[0])
        v = str(match[1])
        edges.append((u, v))
    if not edges:
        edge_pattern = r'\((\d+)\s*,\s*(\d+)\)'
        matches = re.findall(edge_pattern, question)
        for match in matches:
            u = str(match[0])
            v = str(match[1])
            edges.append((u, v))
    return edges


def validate_topological_sort(sequence_nodes, directed_edges):
    """
    Check whether *sequence_nodes* is a valid topological ordering.

    Requires all nodes to appear exactly once and every directed edge (u, v)
    to satisfy position(u) < position(v).
    """
    if not sequence_nodes:
        return False
    if len(sequence_nodes) != len(set(sequence_nodes)):
        return False
    if not directed_edges:
        return True
    # Map each node to its position in the proposed ordering.
    pos = {}
    for i, node in enumerate(sequence_nodes):
        pos[str(node)] = i
    for u, v in directed_edges:
        u_str = str(u)
        v_str = str(v)
        if u_str not in pos or v_str not in pos:
            continue
        if pos[u_str] >= pos[v_str]:
            return False
    return True


def build_edge_set(graph_sequence):
    """
    Reconstruct undirected edges by walking consecutive nodes in each Path line.

    Used when ``edge_index`` is unavailable but the encoded graph text is present.
    """
    edges = set()
    if not graph_sequence:
        return edges
    paths = graph_sequence.split('\n')
    for path in paths:
        if 'Path' in path and ':' in path:
            path_part = path.split(':', 1)[1].strip()
            nodes = extract_node_ids(path_part)
            if not nodes:
                path_nodes = []
                n_pattern = r'\bN(\d+)\b'
                n_matches = re.finditer(n_pattern, path_part, re.IGNORECASE)
                for match in n_matches:
                    node_id = int(match.group(1)) - 1
                    if node_id >= 0:
                        path_nodes.append(str(node_id))
                if not path_nodes:
                    pattern = r'(?:\[NODEID\.([^\]]+)\]|\b(\d+)\b)'
                    matches = re.finditer(pattern, path_part)
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
                    for match in matches:
                        if match.group(1):
                            try:
                                node_id = letter_sequence.index(match.group(1))
                                path_nodes.append(str(node_id))
                            except ValueError:
                                pass
                        elif match.group(2):
                            path_nodes.append(str(match.group(2)))
                nodes = path_nodes
            for i in range(len(nodes) - 1):
                node1, node2 = nodes[i], nodes[i+1]
                edge = tuple(sorted([str(node1), str(node2)]))
                edges.add(edge)
    return edges


def validate_hamiltonian_path(predicted_nodes, ground_truth_nodes, edge_set):
    """
    Verify that *predicted_nodes* is a valid Hamiltonian path on *edge_set*.

    Checks node-set equality, no duplicates, and consecutive pairs forming edges.
    """
    pred_set = set(predicted_nodes)
    gt_set = set(ground_truth_nodes)
    if pred_set != gt_set:
        return False
    if len(predicted_nodes) != len(pred_set):
        return False
    if len(ground_truth_nodes) != len(gt_set):
        return False
    if not edge_set:
        return set(predicted_nodes) == set(ground_truth_nodes) and len(predicted_nodes) == len(set(predicted_nodes))
    for i in range(len(predicted_nodes) - 1):
        node1, node2 = str(predicted_nodes[i]), str(predicted_nodes[i + 1])
        edge = tuple(sorted([node1, node2]))
        if edge not in edge_set:
            return False
    return True


def compare_hamiltonian_path_with_multiple_answers(predicted, ground_truth, edge_set):
    """
    Compare Hamiltonian-path answers, supporting multiple acceptable ground truths.

    Handles explicit ``no`` answers and validates node sequences against *edge_set*
    when both predictions and references contain path nodes.
    """
    if isinstance(ground_truth, list):
        for gt_item in ground_truth:
            if compare_hamiltonian_path_with_multiple_answers(predicted, gt_item, edge_set):
                return True
        return False
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    if pred_norm == gt_norm:
        return True
    pred_is_no = bool(re.search(r'\bno\b', predicted.lower())) or 'does not have' in predicted.lower()
    gt_is_no = bool(re.search(r'\bno\b', ground_truth.lower())) or 'does not have' in ground_truth.lower()
    if pred_is_no and gt_is_no:
        return True
    elif pred_is_no or gt_is_no:
        return False
    pred_nodes = extract_node_ids(predicted)
    gt_nodes = extract_node_ids(ground_truth)
    if not pred_nodes:
        n_pattern = r'\bN(\d+)\b'
        n_matches = re.finditer(n_pattern, predicted, re.IGNORECASE)
        for match in n_matches:
            node_id = int(match.group(1)) - 1
            if node_id >= 0:
                pred_nodes.append(str(node_id))
        if not pred_nodes:
            pattern = r'(?:\[NODEID\.([^\]]+)\]|\b(\d+)\b)'
            matches = re.finditer(pattern, predicted)
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
            for match in matches:
                if match.group(1):
                    try:
                        node_id = letter_sequence.index(match.group(1))
                        pred_nodes.append(str(node_id))
                    except ValueError:
                        pass
                elif match.group(2):
                    pred_nodes.append(str(match.group(2)))
    if not gt_nodes:
        n_pattern = r'\bN(\d+)\b'
        n_matches = re.finditer(n_pattern, ground_truth, re.IGNORECASE)
        for match in n_matches:
            node_id = int(match.group(1)) - 1
            if node_id >= 0:
                gt_nodes.append(str(node_id))
        if not gt_nodes:
            pattern = r'(?:\[NODEID\.([^\]]+)\]|\b(\d+)\b)'
            matches = re.finditer(pattern, ground_truth)
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
            for match in matches:
                if match.group(1):
                    try:
                        node_id = letter_sequence.index(match.group(1))
                        gt_nodes.append(str(node_id))
                    except ValueError:
                        pass
                elif match.group(2):
                    gt_nodes.append(str(match.group(2)))
    if not pred_nodes or not gt_nodes:
        return pred_norm == gt_norm
    return validate_hamiltonian_path(pred_nodes, gt_nodes, edge_set)


def compare_topological_sort_with_multiple_answers(predicted, ground_truth, directed_edges):
    """
    Compare topological-sort answers, supporting multiple acceptable orderings.

    Accepts cycle impossibility answers, exact text matches, or valid orderings
    that satisfy all directed edge constraints.
    """
    if isinstance(ground_truth, list):
        for gt_item in ground_truth:
            if compare_topological_sort_with_multiple_answers(predicted, gt_item, directed_edges):
                return True
        return False
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    if pred_norm == gt_norm:
        return True
    pred_has_cycle = 'cycle' in predicted.lower() or 'cannot be topologically sorted' in predicted.lower()
    gt_has_cycle = 'cycle' in ground_truth.lower() or 'cannot be topologically sorted' in ground_truth.lower()
    if pred_has_cycle and gt_has_cycle:
        return True
    elif pred_has_cycle or gt_has_cycle:
        return False
    pred_nodes = extract_key_info(predicted, "topological_sort")
    gt_nodes = extract_key_info(ground_truth, "topological_sort")
    if pred_nodes is None or gt_nodes is None:
        return pred_norm == gt_norm
    if not isinstance(pred_nodes, list) or not isinstance(gt_nodes, list):
        return pred_nodes == gt_nodes
    if set(str(n) for n in pred_nodes) != set(str(n) for n in gt_nodes):
        return False
    if directed_edges:
        pred_valid = validate_topological_sort([str(n) for n in pred_nodes], directed_edges)
        if pred_valid:
            return True
        else:
            return False
    return pred_nodes == gt_nodes


def compare_answers(predicted, ground_truth, task_type, edge_set=None, directed_edges=None):
    """
    Main comparison entry point: dispatch to task-specific logic or key-info equality.

    Args:
        predicted: Model-generated answer text.
        ground_truth: Reference answer text (may be a list for multi-answer tasks).
        task_type: One of the TAB graph task names.
        edge_set: Undirected edges for path-validation tasks.
        directed_edges: Directed edges for topological sort validation.
    """
    if task_type == "hamiltonian_path":
        return compare_hamiltonian_path_with_multiple_answers(predicted, ground_truth, edge_set)
    if task_type == "topological_sort":
        return compare_topological_sort_with_multiple_answers(predicted, ground_truth, directed_edges)
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    if pred_norm == gt_norm:
        return True
    pred_info = extract_key_info(predicted, task_type)
    gt_info = extract_key_info(ground_truth, task_type)
    if pred_info is None or gt_info is None:
        return pred_norm == gt_norm
    return pred_info == gt_info


def evaluate_answer(predicted, ground_truth, task_type=None, edge_index=None, graph_sequence=None, question=None):
    """
    High-level evaluation helper: build graph context, truncate to first sentence, then compare.

    Automatically derives ``edge_set`` or ``directed_edges`` from the best available
    source (edge_index, graph_sequence, or question text).
    """
    if task_type is None:
        pred_norm = normalize_text(predicted)
        gt_norm = normalize_text(ground_truth)
        return pred_norm == gt_norm
    edge_set = None
    directed_edges = None
    # Pick graph representation based on task type (directed vs undirected).
    if task_type == "topological_sort":
        if edge_index is not None:
            directed_edges = build_directed_edges_from_edge_index(edge_index)
        elif question:
            directed_edges = build_directed_edges_from_question(question)
    else:
        if edge_index is not None:
            edge_set = build_edge_set_from_edge_index(edge_index)
        elif graph_sequence:
            edge_set = build_edge_set(graph_sequence)
        elif question:
            edge_set = build_edge_set_from_question(question)
    predicted = extract_first_sentence(predicted)
    return compare_answers(predicted, ground_truth, task_type, edge_set, directed_edges)