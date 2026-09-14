from typing import List


def execute_optimized_sparse_moe(hidden_states, dynamic_probs, topk_indices, experts_list, prob_epsilon: float = 1e-8):
    """
    Sparse token dispatcher for MoE.

    Args:
        hidden_states: Tensor [num_tokens, hidden_dim]
        dynamic_probs: Tensor [num_tokens, max_k]
        topk_indices: Tensor [num_tokens, max_k]
        experts_list: sequence of expert modules
        prob_epsilon: assignments <= epsilon are treated as zero

    Returns:
        Tensor [num_tokens, hidden_dim] recombined output.
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("execute_optimized_sparse_moe requires torch") from exc

    num_tokens, _ = hidden_states.shape
    max_k = topk_indices.shape[1]
    num_experts = len(experts_list)

    final_output = torch.zeros_like(hidden_states)

    flat_expert_indices = topk_indices.reshape(-1)
    flat_probs = dynamic_probs.reshape(-1)
    token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(max_k)

    valid_mask = flat_probs > prob_epsilon
    if not valid_mask.any():
        return final_output

    active_expert_indices = flat_expert_indices[valid_mask]
    active_probs = flat_probs[valid_mask]
    active_token_ids = token_ids[valid_mask]

    sort_order = torch.argsort(active_expert_indices)
    sorted_experts = active_expert_indices[sort_order]
    sorted_token_ids = active_token_ids[sort_order]
    sorted_probs = active_probs[sort_order]

    unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)

    offset = 0
    for expert_id_tensor, count_tensor in zip(unique_experts, counts):
        count = int(count_tensor.item())
        if count <= 0:
            continue

        expert_idx = int(expert_id_tensor.item())
        if expert_idx < 0 or expert_idx >= num_experts:
            offset += count
            continue

        bucket_slice = slice(offset, offset + count)
        tokens_for_expert = sorted_token_ids[bucket_slice]
        weights_for_expert = sorted_probs[bucket_slice].unsqueeze(-1)

        expert_inputs = hidden_states[tokens_for_expert]
        expert_outputs = experts_list[expert_idx](expert_inputs)
        weighted_outputs = expert_outputs * weights_for_expert

        final_output.index_add_(0, tokens_for_expert, weighted_outputs)
        offset += count

    return final_output
