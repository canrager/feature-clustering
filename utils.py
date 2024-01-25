# %%
import streamlit as st
import os
import json
import numpy as np
from collections import Counter

class ClusterCfg:
  def __init__(self):
    self.score_type = None
    self.pos_reduction = None
    self.abs_scores = None
    self.dim_reduction = None

filename = './data/contexts_pythia-70m-deduped_loss-thresh0.005_skip50_ntok10000_nonzero_pos-reduction-final_mlp.json'
context_y = json.loads(open(filename).read())
y_global_idx = np.array(list(context_y.keys()), dtype=int)

# Load cluster results
# @st.cache_data
def load_cluster_results(_ccfg):
  clusters_dir = './data'
  clusters_filename = os.path.join(clusters_dir, f'clusters_{_ccfg.score_type}_{_ccfg.pos_reduction}_{_ccfg.dim_reduction}.json')
  clustering_results = json.loads(open(clusters_filename).read())
  cluster_totals = [ int(s) for s in list(clustering_results.keys()) ]
  return clustering_results, cluster_totals

def get_context(idx):
    """given idx in range(0, 10658635), return dataset sample
    and predicted token index within sample, in range(1, 1024)."""
    idx = str(idx)
    return context_y[idx]['context'], context_y[idx]['y']

def convert_global_idxs_to_token_str(idxs):
    """given a list of global indexes, return "token\t(document_id: X, global_token_id: Y)" strings"""
    y = [context_y[str(idx)]['y'] for idx in idxs]
    doc_idxs = [context_y[str(idx)]['document_idx'] for idx in idxs]
    token_strs = [f'{y[i]}   (doc {doc_idxs[i]})' for i in range(len(idxs))]
    # token_strs = [context_y[str(idx)]['y'] + f'\t(in document {context_y[str(idx)]["document_idx"]} with global_token_idx {idx})' for idx in str(idxs)]
    return token_strs


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
    token_strs = [context_y[str(idx)]['y'] for idx in idxs]
    counts = Counter(token_strs)
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return counts