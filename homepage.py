import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict

import sys
sys.path.append("/Users/canrager/code/streamlit_first")
from cluster_exploration import *

class ClusterCfg:
  def __init__(self):
    self.score_type = None
    self.pos_reduction = None
    self.abs_scores = None
    self.dim_reduction = None
    
ccfg = ClusterCfg()




# Set up the title of the page
st.title('Feature pattern explorer')

st.markdown("""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; border: 1px solid #aaa;">
        <p>Unsupervised method for clustering sentences with diverse contexts into groups where similar features are involved for next token prediction.</p>
    </div>
""", unsafe_allow_html=True)

st.text('')

"""
### Method
We call the input sequence of tokens *context* and the token to be predicted *y*.
1. Choose contexts where the model achieves low loss on predicting y. 
    (Loss metric: cross-entropy. We choose the minimal loss threshold such that we get 10k contexts. 
    Predicted tokens y are spaced at least 100 tokens from each other to cover a diverse range of contexts.)

2. For each context, we compute a vector of feature scores. Scores correspond to either 1) Feature activation or 2) Linear effect on loss: activation * gradient (of loss w.r.t. feature activation)

 
3. (Optional) Perform dimensionality reduction on the feature scores using Truncated SVD. We map feature scores to the basis of right singular vectors of the feature score matrix X: [contexts, features].

4. Perform spectral clustering on the feature score matrix X: [contexts, features]. The number of clusters is a hyperparameter.



"""

# Creating three dropdown menus
st.header('Data preparation')

option_importance_metric = st.selectbox('Feature score metric', ('Activation on context' , 'Activation * gradient'))
if option_importance_metric == 'Activation on context':
    ccfg.score_type = 'act'
elif option_importance_metric == 'Activation * gradient':
    ccfg.score_type = 'act-grad'

option_positon = st.selectbox('Reduction of feature pattern across positions', ('Final position only', 'Sum over positions (NOT IMPLEMENTED)', 'No reduction (NOT IMPLEMENTED)'))
if option_positon == 'Final position only':
    ccfg.pos_reduction = 'final'
elif option_positon == 'Sum over positions (NOT IMPLEMENTED)':
    ccfg.pos_reduction = 'final'
elif option_positon == 'No reduction (NOT IMPLEMENTED)':
    ccfg.pos_reduction = 'final'
    
option_absolutes = st.selectbox('Use absolute scores', ('Yes', 'No'))
if option_absolutes == 'Yes':
    ccfg.abs_scores = True
elif option_absolutes == 'No':
    ccfg.abs_scores = False

# load act-n-grad results
@st.cache_data
def load_act_n_grad_results():
  if option_positon == 'Final position only':
    filename = "/Users/canrager/code/streamlit_first/act-n-grad_pythia-70m-deduped_loss-thresh0.005_skip50_ntok10000_nonzero_mlp.json"
  act_per_context = json.loads(open(filename).read())
  y_global_idx = np.array(list(act_per_context.keys()), dtype=int)
  num_y = len(act_per_context)
  return act_per_context, y_global_idx, num_y

act_per_context, y_global_idx, num_y = load_act_n_grad_results()



st.header('Clustering')
option_dim_reduction = st.selectbox('Dimensionality reduction on feature pattern', ('None', 'SVD'))
if option_dim_reduction == 'None':
    ccfg.dim_reduction = "nosvd"
elif option_dim_reduction == 'SVD':
    ccfg.dim_reduction = "svd"

# Load cluster results
# @st.cache_data
def load_cluster_results(_ccfg):
  clusters_dir = '/Users/canrager/code/streamlit_first/clusters'
  clusters_filename = os.path.join(clusters_dir, f'clusters_{_ccfg.score_type}_{_ccfg.pos_reduction}_{_ccfg.dim_reduction}.json')
  clustering_results = json.loads(open(clusters_filename).read())
  cluster_totals = [ int(s) for s in list(clustering_results.keys()) ]
  return clustering_results, cluster_totals

clustering_results, cluster_totals = load_cluster_results(ccfg)

# Inspection of clusters
option_n_clusters = st.selectbox('Total number of clusters in algorithm', cluster_totals)
if option_n_clusters:
    clusters_available = range(1, option_n_clusters + 1)
    option_cluster_idx = st.selectbox('Inspect cluster index', clusters_available)


# Display the selected options
# st.write('You selected:', option_importance_metric, option_positon, option_dim_reduction)



# Show predicted y
st.header('y token counts in cluster')

counts = return_token_occurrences_in_cluster(clustering_results, y_global_idx, n_total_clusters=option_n_clusters, cluster_idx=option_cluster_idx, abs_scores=ccfg.abs_scores)
counts = sorted(counts, key=lambda x: x[1], reverse=True)

# dictionary with counts as keys and list of tokens as values
cnt_dict = defaultdict(list)
for token, count in counts:
    cnt_dict[count].append(token)

# Show the counts in descending order
for count in sorted(cnt_dict.keys(), reverse=True):
    st.markdown(f'**{count} occurence(s) in cluster:**\n')
    l = len(cnt_dict[count])
    # print sets of ten tokens per line
    for i in range(0, l, 10):
        st.write(", ".join(cnt_dict[count][i:i+10]))

# For a single token selected in a dropdown, show the context using the print_context function
st.header('Context for selected token')
global_idxs = find_global_idxs_for_tokens_in_cluster(clustering_results, y_global_idx, cluster_idx=option_cluster_idx, n_total_clusters=option_n_clusters, abs_scores=ccfg.abs_scores)
# create a list of (global_idx, token) tuples
global_idxs_tokens = [(idx, convert_global_idxs_to_token_str([idx])[0]) for idx in global_idxs]
option_token = st.selectbox('Global token index, token y', global_idxs_tokens)
prompt, token = print_context(option_token[0])
st.write(f'<p style="color:white;">{prompt}<span style="color:red;">{token}</span></p>', unsafe_allow_html=True)
