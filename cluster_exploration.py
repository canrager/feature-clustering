# %%
import json
import numpy as np
from collections import Counter


# %%
# Print results
# Load dataset and helper functions
# Dataset
# dataset_canonical = "./dataset_test_tokenized_200k/"
# dataset = datasets.load_from_disk(dataset_canonical)

# Dataset handling
# starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

# def loss_idx_to_dataset_idx(idx):
#     """given an idx in range(0, 10658635), return
#     a sample index in range(0, 20000) and pred-in-sample
#     index in range(0, 1023). Note token-in-sample idx is
#     exactly pred-in-sample + 1"""
#     sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
#     pred_in_sample_index = idx - starting_indexes[sample_index]
#     return int(sample_index), int(pred_in_sample_index)

filename = './contexts_pythia-70m-deduped_loss-thresh0.005_skip50_ntok10000_nonzero_pos-reduction-final_mlp.json'
context_y = json.loads(open(filename).read())
y_global_idx = np.array(list(context_y.keys()), dtype=int)

def get_context(idx):
    """given idx in range(0, 10658635), return dataset sample
    and predicted token index within sample, in range(1, 1024)."""
    idx = str(idx)
    return context_y[idx]['context'], context_y[idx]['y']

# def print_context(idx):
#     """
#     given idx in range(0, 10658635), print prompt preceding the corresponding
#     prediction, and highlight the predicted token.
#     """
#     sample, token_idx = get_context(idx)
#     prompt = sample["split_by_token"][:token_idx]
#     prompt = "".join(prompt)
#     token = sample["split_by_token"][token_idx]
#     print(prompt + "\033[41m" + token + "\033[0m")
#     return prompt, token

def convert_global_idxs_to_token_str(idxs):
    """given a list of global indexes, return a list of corresponding token strings"""
    return [context_y[str(idx)]['y'] for idx in idxs]

def find_global_idxs_for_tokens_in_cluster(clustering_results, cluster_idx, n_total_clusters, abs_scores=False):
    if abs_scores:
        abs_int = 1
    else:
        abs_int = 0
    num_y = len(y_global_idx)
    ones = np.ones(num_y)
    mask = clustering_results[str(n_total_clusters)][abs_int] == cluster_idx * ones
    idxs = y_global_idx[mask]
    return idxs

def return_token_occurrences_in_cluster(clustering_results, cluster_idx, n_total_clusters, abs_scores=False):
    """given a cluster index, return a list of tuples of (token, count) for all unique tokens"""
    idxs = find_global_idxs_for_tokens_in_cluster(clustering_results, cluster_idx, n_total_clusters, abs_scores)
    token_strs = convert_global_idxs_to_token_str(idxs)
    counts = Counter(token_strs)
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return counts

