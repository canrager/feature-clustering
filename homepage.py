"""
This script loads the clusters and corresponding contexts from the data folder and
displays them in a Streamlit app. 
"""

from collections import defaultdict
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_shortcuts import add_keyboard_shortcuts

def tokens_to_html(tokens, max_len=150):
    """Given a list of tokens (strings), returns html for displaying the tokenized text.
    """
    newline_tokens = ['\n', '\r', '\r\n', '\v', '\f']
    html = ""
    txt = ""
    if len(tokens) > max_len:
        html += '<span>...</span>'
    tokens = tokens[-max_len:]
    for i, token in enumerate(tokens):
        background_color = "white" if i != len(tokens) - 1 else "#FF9999"
        txt += token
        if all([c in newline_tokens for c in token]):
            # replace all instances with ⏎
            token_rep = len(token) * "⏎"
            brs = "<br>" * len(token)
            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; white-space: pre-wrap;">{token_rep}</span>{brs}'
        else:
            # replace any $ with \$ to avoid markdown interpretation
            token = token.replace("$", "\$")
            # replace any < with &lt; to avoid html interpretation
            # token = token.replace("<", "&lt;")
            # replace any > with &gt; to avoid html interpretation
            # token = token.replace(">", "&gt;")
            # replace any & with &amp; to avoid html interpretation
            token = token.replace("&", "&amp;")
            # replace any _ with \_ to avoid markdown interpretation
            token = token.replace("_", "\_")
            # also escape * to avoid markdown interpretation
            token = token.replace("*", "\*")
            # there's also an issue with the backtick, so escape it
            token = token.replace("`", "\`")

            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; white-space: pre-wrap;">{token}</span>'
    if "</" in txt:
        return "CONTEXT NOT LOADED FOR SECURITY REASONS SINCE IT CONTAINS HTML CODE (could contain javascript)."
    else:
        return html


# Create sidebar for selecting clusters file and cluster
st.sidebar.header('Cluster choice')

# Selectbox for the clusters file
cluster_files = os.listdir("clusters")
cluster_file = st.sidebar.selectbox('Select cluster file', cluster_files)
with open(f"clusters/{cluster_file}") as f:
    clusters = json.load(f)


# Selectbox for choosing n_clusters
n_clusters_options = sorted(list(clusters.keys()), key=int)
n_clusters = st.sidebar.selectbox('n_clusters used in clustering algorithm', n_clusters_options, index=len(n_clusters_options) - 1)
clusters = clusters[n_clusters] # note that n_clusters is a string
if "ERIC" in cluster_file:
    clusters = clusters[0] # ignore clusters[1], which is based on absolute values

# From the clusters list, create a dictionary mapping cluster index to token indices
cluster_to_tokens = defaultdict(list)
for i, cluster in enumerate(clusters):
    cluster_to_tokens[cluster].append(i)

# sort clusters by size (dictionary of rank -> old cluster index)
new_index_old_index = {i: cluster for i, cluster in enumerate(sorted(cluster_to_tokens, key=lambda k: len(cluster_to_tokens[k]), reverse=True))}

def get_idx(cluster_file, n_clusters):
    if cluster_file not in st.session_state:
        st.session_state[cluster_file] = dict()
    if n_clusters not in st.session_state[cluster_file]:
        st.session_state[cluster_file][n_clusters] = int(n_clusters) // 2
    return st.session_state[cluster_file][n_clusters]

def increment_idx(cluster_file, n_clusters):
    st.session_state[cluster_file][n_clusters] += 1
    return st.session_state[cluster_file][n_clusters]

def decrement_idx(cluster_file, n_clusters):
    st.session_state[cluster_file][n_clusters] -= 1
    return st.session_state[cluster_file][n_clusters]

def set_idx(cluster_file, n_clusters, idx):
    st.session_state[cluster_file][n_clusters] = idx
    return st.session_state[cluster_file][n_clusters]

# choose a cluster index
cluster_idx = st.sidebar.selectbox('Select cluster index', range(int(n_clusters)), index=get_idx(cluster_file, n_clusters))
set_idx(cluster_file, n_clusters, cluster_idx)

def left_callback():
    if cluster_idx > 0:
        decrement_idx(cluster_file, n_clusters)

def right_callback():
    if cluster_idx < int(n_clusters) - 1:
        increment_idx(cluster_file, n_clusters)

# these don't take any action. fix this:
if st.sidebar.button('Previous cluster', on_click=left_callback):
    pass
if st.sidebar.button('Next cluster', on_click=right_callback):
    pass

# add keyboard shortcuts
add_keyboard_shortcuts({
    "ArrowLeft": "Previous cluster",
    "ArrowRight": "Next cluster"
})

# add text to the sidebar
st.sidebar.write(f"You can use the left and right arrow keys to move quickly between clusters.")

# load up the contexts and the clusters
with open("cluster_context_map.json") as f:
    context_filename = json.load(f)[cluster_file]
with open(f"contexts/{context_filename}") as f:
    samples = json.load(f)

idx_to_token_idx = list(samples.keys())

# write as large bolded heading the cluster index
st.write(f"## Cluster {cluster_idx}")

# get a histogram of the top 'y' tokens in the cluster
counts = defaultdict(int)
for i in cluster_to_tokens[new_index_old_index[cluster_idx]]:
    sample = samples[idx_to_token_idx[i]]
    y = sample['answer']
    counts[y] += 1

# plot the histogram for the top 10 tokens with matplotlib
top_10 = sorted(counts, key=counts.get, reverse=True)[:10]
top_10_counts = [counts[y] for y in top_10]
# convert the top 10 tokens to literals (i.e. newlines and tabs are escaped)
top_10 = [repr(y)[1:-1] for y in top_10]
plt.figure(figsize=(6, 2))
plt.bar(top_10, top_10_counts)
plt.xlabel('Token', fontsize=8)
plt.ylabel('Count', fontsize=8)
plt.title('Top 10 tokens in cluster (answer tokens)', fontsize=8)
# rotate the tick labels
plt.xticks(rotation=45, fontsize=9)
plt.yticks(fontsize=8)
st.pyplot(plt)

for i in cluster_to_tokens[new_index_old_index[cluster_idx]]:
    sample = samples[idx_to_token_idx[i]]
    context = sample['context']
    y = sample['answer']
    tokens = context + [y]
    html = tokens_to_html(tokens)
    st.write("-----------------------------------------------------------")
    st.write(html, unsafe_allow_html=True)

