import streamlit as st
import numpy as np
from clustering_utils import *
import plotly.express as px

# CSS to apply a light gray background to each token
token_style = "background-color: #f0f2f6; padding: 2px; margin: 2px; border-radius: 4px;"



# Set up the title of the page
st.title('Feature pattern explorer')

# Method description
"""
> **Unsupervised method for clustering texts by dictionary features involved for next token prediction.**

## Method
We call the input sequence of tokens *context* and the token to be predicted *y*.
1. Choose contexts from The Pile where the model achieves low loss on predicting y. 
    (Loss metric: cross-entropy. We choose the minimal loss threshold such that we get 10k contexts. 
    Predicted tokens y are spaced at least 100 tokens from each other to cover a diverse range of contexts.)

2. For each context, we compute a vector of dictionary feature scores. Scores correspond to either 1) Feature activation or 2) Linear effect on loss: activation * gradient (of loss w.r.t. feature activation)

3. Perform spectral clustering on the feature score matrix X: [contexts, features]. 
    The number of clusters is a hyperparameter. 
    (Update: We examine the SVD in a separate tab, see index on the left. Previously, we performed dimensionality (Truncated SVD).
    We mapped feature scores to the basis of right singular vectors of the feature score matrix X: [contexts, features]. 
    Representing X in sparse format significantly sped up the clustering, so dimensionality reduction is not necessary for clustering.)
"""







# Dropdown menus for data preparation
st.header('Choose cluster')
_ccfg = ClusterCfg()

option_importance_metric = st.selectbox('Feature score metric', ('Activation', 'Activation * gradient'), index=1)
if option_importance_metric == 'Activation':
    _ccfg.score_type = 'act'
elif option_importance_metric == 'Activation * gradient':
    _ccfg.score_type = 'act-grad'

option_positon = st.selectbox('Reduction of feature pattern across positions', ('Sum over positions', 'Final position only', 'No reduction (NOT IMPLEMENTED)'), index=1)
if option_positon == 'Final position only':
    _ccfg.pos_reduction = 'final'
elif option_positon == 'Sum over positions':
    _ccfg.pos_reduction = 'sum'
elif option_positon == 'No reduction (NOT IMPLEMENTED, default to sum)':
    _ccfg.pos_reduction = 'sum'
    
option_absolutes = st.selectbox('Use absolute scores', ('Yes', 'No'), index=1)
if option_absolutes == 'Yes':
    _ccfg.abs_scores = True
elif option_absolutes == 'No':
    _ccfg.abs_scores = False

# option_dim_reduction = st.selectbox('Dimensionality reduction on feature pattern', ('None', 'SVD'), index=0)
# elif option_dim_reduction == 'SVD':
#     _ccfg.dim_reduction = "svd"
# if option_dim_reduction == 'None':
_ccfg.dim_reduction = "nosvd"

clustering_results, cluster_totals = load_cluster_results(_ccfg)


# Dropdown menus for inspecting clusters
option_n_clusters = st.selectbox('Total number of clusters in algorithm', cluster_totals, index=9)
if option_n_clusters:
    clusters_available = range(1, option_n_clusters + 1)
    option_cluster_idx = st.selectbox('Inspect cluster index', clusters_available, index=9)









st.header('Inspect cluster')

# Find number of points in cluster
global_idxs_in_cluster = find_global_idxs_for_tokens_in_cluster(clustering_results, cluster_idx=option_cluster_idx, n_total_clusters=option_n_clusters, abs_scores=_ccfg.abs_scores)
n_points = len(global_idxs_in_cluster)
if n_points == 1 :
    st.write(f'This cluster contains **{n_points} datapoint** in total.')
else:
    st.write(f'This cluster contains **{n_points} datapoints** in total.')

# Accumulated view of tokens at selected position in cluster
st.subheader('Count token occurences')

option_token = st.selectbox('Select token position', ['final token in context', 'true next token y', 'I want to inspect one of the final 10 tokens in the context'], index=0)
if option_token == 'final token in context':
    option_token = -1
elif option_token == 'true next token y':
    option_token = 'y'
elif option_token == 'I want to inspect one of the final 10 tokens in the context':
    option_token = st.number_input('Enter token position (from -10 to -1, where -1 is the final token in the context)', min_value=-10, max_value=-1, value=-1)

# dictionary with counts as keys and list of tokens as values
cnt_dict = return_token_occurrences_in_cluster(
    clustering_results, 
    n_total_clusters=option_n_clusters, 
    cluster_idx=option_cluster_idx, 
    abs_scores=_ccfg.abs_scores,
    token=option_token)

# Plot the distribution of token occurrences if max count is larger than 4
if len(cnt_dict) >= 3:
    # Preparing data
    occurrences = list(cnt_dict.keys())
    token_counts = [len(cnt_dict[occ]) for occ in occurrences]
    tokens = ['<br>'.join(cnt_dict[occ]) for occ in occurrences]

    # Create a bar chart
    fig = px.bar(x=occurrences, y=token_counts, #[f'Count: {t}' for t in token_counts],
                title='Distribution of token occurrences at selected position in cluster',
                labels={'x': 'Number of occurrences', 'y': 'Number of tokens'})

    # Update layout for hover label font size
    fig.update_layout(xaxis_title='Number of occurrences', 
                    yaxis_title='Number of unique tokens', 
                    xaxis_type='category',
                    yaxis_type='log')

    # Show the plot in Streamlit
    st.plotly_chart(fig)
else:
    st.write('*We will show the distribution of token occurrences if at least one token occurs >3 times in the selected cluster.*')


# Show the counts in descending order
for count in sorted(cnt_dict.keys(), reverse=True):
    if count == 1:
        st.markdown(f'**{count} occurence in cluster:**\n')
    else:
        st.markdown(f'**{count} occurences in cluster:**\n')

    # Wrap each token in a span with the style applied
    tokens_with_style = [f"<span style='{token_style}'>{token}</span>" for token in cnt_dict[count]]
    tokens_html = " ".join(tokens_with_style)
    st.markdown(tokens_html, unsafe_allow_html=True)






# For a single token or interval selected in a dropdown, show the preceeding context
st.subheader('Browse contexts')

st.write('*Select a true token y from the dropdown menu to its the preceeding context. '+
         'We display up to 100 tokens before the true next token y (highlighted in green), dependent on the context length. '+
         'Each context is sampled from single document in the dataset. Check whether two contexts overlap by comparing the document number in the dropdown.*')

def display_one_context():
    global_idxs_tokens_options = convert_global_idxs_to_token_str(global_idxs_in_cluster)
    option_token = st.selectbox('Select token', global_idxs_tokens_options, index=0)
    selected_idx = global_idxs_tokens_options.index(option_token)
    context, y = get_context(global_idxs_in_cluster[selected_idx])
    render_context_y(context, y)

if n_points == 1:
    display_one_context()
    
else:
    # Selectbox for the amount of points to display
    n_points_options = np.array([1, 5, 10, 20, 50])
    n_points_options = n_points_options[n_points_options <= n_points]
    if n_points not in n_points_options: # Provide option to display all points
        n_points_options = np.append(n_points_options, n_points)
        n_points_options = np.sort(n_points_options)

    n_points_to_display = st.selectbox('Select number of contexts to display', n_points_options, index=1)
    if n_points_to_display == 1:
        display_one_context()
    else:
        # Select the group index of points to display
        first_group_idxs = np.arange(0, n_points, n_points_to_display)
        group_intervals = [
            np.arange(idx, min(n_points, idx + n_points_to_display))\
            for idx in first_group_idxs
            ]

        group_intervals_str = [f'{group[0]} - {group[-1]}' for group in group_intervals]
        group_intervals_map = {group_intervals_str[i]: group_intervals[i] for i in range(len(group_intervals_str))}
        option_group_str = st.selectbox('Select interval', group_intervals_str, index=0)
        selected_group_interval = group_intervals_map[option_group_str]

        # global_idxs_tokens_options = convert_global_idxs_to_token_str(global_idxs)
        # option_token = st.selectbox('Select token', global_idxs_tokens_options, index=0)
        # selected_idx = global_idxs_tokens_options.index(option_token)
        contexts, ys = get_contexts(global_idxs_in_cluster[selected_group_interval])
        # Format context to be rendered as plain text, not markdown
        for i, context, y in zip(selected_group_interval, contexts, ys):
            st.write(f'#### Context {i}:\n')
            render_context_y(context, y)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Test Playground
    
import numpy as np
n_points = 13
n_points_to_display = 5

# Select the group index of points to display
first_group_idxs = np.arange(0, n_points, n_points_to_display)
group_intervals = [
    np.arange(idx, min(n_points, idx + n_points_to_display))\
    for idx in first_group_idxs
    ]

# %%
