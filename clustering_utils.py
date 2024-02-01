# %%
import os
import json
import numpy as np
from collections import Counter, defaultdict
from markupsafe import escape
import streamlit as st

class ClusterCfg:
  def __init__(self, score_type=None, pos_reduction=None, abs_scores=None):
    self.score_type = score_type
    self.pos_reduction = pos_reduction
    self.abs_scores = abs_scores
    self.dim_reduction = 'nosvd' # always no SVD

filename = './data/contexts_pythia-70m-deduped_tloss0.03_ntok10000_skip512_npos10_mlp.json'
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

def get_contexts(idxs):
    """given a list of idxs in range(0, 10658635), return list of dataset samples
    and predicted token indexes within samples, in range(1, 1024)."""
    contexts = [context_y[str(idx)]['context'] for idx in idxs]
    ys = [context_y[str(idx)]['y'] for idx in idxs]
    return contexts, ys

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

def return_token_occurrences_in_cluster(clustering_results, cluster_idx, n_total_clusters, abs_scores=False, token="y"):
    """given a cluster index, return a list of tuples of (token, count) for all unique tokens"""
    idxs = find_global_idxs_for_tokens_in_cluster(clustering_results, cluster_idx, n_total_clusters, abs_scores)
    if token == "y":
        token_strs = [context_y[str(idx)]['y'] for idx in idxs]
    elif type(token) == int:
        token_strs = [context_y[str(idx)]['context'][token] for idx in idxs]
    else:
        raise ValueError(f"token must be 'y' or an integer, not {token}")
    counts = Counter(token_strs)
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    cnt_dict = defaultdict(list)
    for y, count in counts:
        cnt_dict[count].append(y)
    return cnt_dict

def render_context_y(text, y=None, render_newlines=False):
    text = "".join(text)
    text = escape(text) # display html tags as text
    if y:
        context_formatted = f"<pre style='white-space:pre-wrap;'>{text}<span style='background-color: rgba(0, 255, 0, 0.5)'>{y}</span></pre>"
    else:
        context_formatted = f"<pre style='white-space:pre-wrap;'>{text}</pre>"
    if render_newlines:
        context_formatted = context_formatted.replace("\n", "<br>") # display newlines in html
    st.write(context_formatted, unsafe_allow_html=True)