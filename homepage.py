import streamlit as st
from collections import defaultdict
from utils import *


# Set up the title of the page
st.title('Feature pattern explorer')

# Method description
"""
> **Unsupervised method for clustering sentences with diverse contexts into groups where similar features are involved for next token prediction.**

## Method
We call the input sequence of tokens *context* and the token to be predicted *y*.
1. Choose contexts from The Pile where the model achieves low loss on predicting y. 
    (Loss metric: cross-entropy. We choose the minimal loss threshold such that we get 10k contexts. 
    Predicted tokens y are spaced at least 100 tokens from each other to cover a diverse range of contexts.)

2. For each context, we compute a vector of feature scores. Scores correspond to either 1) Feature activation or 2) Linear effect on loss: activation * gradient (of loss w.r.t. feature activation)

 
3. (Optional) Perform dimensionality reduction on the feature scores using Truncated SVD. We map feature scores to the basis of right singular vectors of the feature score matrix X: [contexts, features].

4. Perform spectral clustering on the feature score matrix X: [contexts, features]. The number of clusters is a hyperparameter.
"""


# Dropdown menus for data preparation
st.header('Data preparation')
_ccfg = ClusterCfg()

option_importance_metric = st.selectbox('Feature score metric', ('Activation', 'Activation * gradient'), index=1)
if option_importance_metric == 'Activation':
    _ccfg.score_type = 'act'
elif option_importance_metric == 'Activation * gradient':
    _ccfg.score_type = 'act-grad'

option_positon = st.selectbox('Reduction of feature pattern across positions', ('Final position only', 'Sum over positions (NOT IMPLEMENTED)', 'No reduction (NOT IMPLEMENTED)'))
if option_positon == 'Final position only':
    _ccfg.pos_reduction = 'final'
elif option_positon == 'Sum over positions (NOT IMPLEMENTED)':
    _ccfg.pos_reduction = 'final'
elif option_positon == 'No reduction (NOT IMPLEMENTED)':
    _ccfg.pos_reduction = 'final'
    
option_absolutes = st.selectbox('Use absolute scores', ('Yes', 'No'), index=1)
if option_absolutes == 'Yes':
    _ccfg.abs_scores = True
elif option_absolutes == 'No':
    _ccfg.abs_scores = False

option_dim_reduction = st.selectbox('Dimensionality reduction on feature pattern', ('None', 'SVD'), index=1)
if option_dim_reduction == 'None':
    _ccfg.dim_reduction = "nosvd"
elif option_dim_reduction == 'SVD':
    _ccfg.dim_reduction = "svd"

clustering_results, cluster_totals = load_cluster_results(_ccfg)


# Dropdown menus for inspecting clusters
st.header('Inspect clusters')

option_n_clusters = st.selectbox('Total number of clusters in algorithm', cluster_totals, index=5)
if option_n_clusters:
    clusters_available = range(1, option_n_clusters + 1)
    option_cluster_idx = st.selectbox('Inspect cluster index', clusters_available, index=9)


# Accumulated view of true next tokens y in cluster
st.subheader('True next tokens y in cluster')

# dictionary with counts as keys and list of tokens as values
counts = return_token_occurrences_in_cluster(clustering_results, n_total_clusters=option_n_clusters, cluster_idx=option_cluster_idx, abs_scores=_ccfg.abs_scores)
counts = sorted(counts, key=lambda x: x[1], reverse=True)
cnt_dict = defaultdict(list)
for y, count in counts:
    cnt_dict[count].append(y)

# Show the counts in descending order
for count in sorted(cnt_dict.keys(), reverse=True):
    st.markdown(f'**{count} occurence(s) in cluster:**\n')
    l = len(cnt_dict[count])
    # print sets of ten tokens per line
    for i in range(0, l, 10):
        st.write(", ".join(cnt_dict[count][i:i+10]))


# For a single token selected in a dropdown, show the preceeding context
st.subheader('Context for true next token')

# st.write('Select a true token y from the dropdown menu to see the corresponding context. The true token y is printed in red.')
global_idxs = find_global_idxs_for_tokens_in_cluster(clustering_results, cluster_idx=option_cluster_idx, n_total_clusters=option_n_clusters, abs_scores=_ccfg.abs_scores)
# create a list of (global_idx, token) tuples
global_idxs_tokens = [(idx, convert_global_idxs_to_token_str([idx])[0]) for idx in global_idxs]
option_token = st.selectbox('Global token index, true token prediction y (printed in red)', global_idxs_tokens)
context, y = get_context(option_token[0])
st.text(context)
st.write(f'<p<span style="color:red;">{y}</span></p>', unsafe_allow_html=True)
