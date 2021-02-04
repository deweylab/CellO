import pygraphviz
#import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
from pygraphviz import AGraph
import matplotlib as mpl
from IPython.display import Image

from . import cello
from .graph_lib import graph
from . import ontology_utils as ou

def probabilities_on_graph(cell, results_df, rsrc_loc, root_label=None, p_thresh=0.0):
    label_graph = cello._retrieve_label_graph(rsrc_loc) # This should move to a utility function
    is_term_ids = 'CL:' in results_df.columns[0]

    span_labels = set([
        label
        for label, prob in zip(results_df.columns, results_df.loc[cell])
        if prob > p_thresh
    ])
    if root_label:
        span_labels &= label_graph._downstream_nodes(
            root_label, 
            label_graph.source_to_targets
        )
    label_graph = graph.subgraph_spanning_nodes(
        label_graph,
        span_labels
    )
    

    label_to_prob = {
        label: prob
        for label, prob in zip(results_df.columns, results_df.loc[cell])
        if label in label_graph.source_to_targets
    }
    if is_term_ids:
        label_to_name = {
            label: '{}\n{:.2f}'.format(
                ou.get_term_name(label), 
                prob
            )
            for label, prob in label_to_prob.items()
        }
    else:
        label_to_name = {
            label: '{}\n{:.2f}'.format(
                label,
                prob
            )
            for label, prob in label_to_prob.items()
        }

    g = _render_graph(
        label_graph.source_to_targets,
        label_to_name,
        "Probabilities for {}".format(cell),
        label_to_prob
    )
    return g

def _render_graph(
        source_to_targets,
        node_to_label,
        metric_name,
        node_to_value
    ):

    g = AGraph(directed=True)

    # Gather all nodes in graph
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)

    # Build graphviz graph
    for node in all_nodes:
        if node not in node_to_value:
            continue

        cmap = mpl.cm.get_cmap('viridis')
        rgba = cmap(node_to_value[node])
        color_value = mpl.colors.rgb2hex(rgba)

        if node_to_value[node] > 0.5:
            font_color = 'black'
        else:
            font_color = 'white'

        g.add_node(
            node_to_label[node],
            label=node_to_label[node],
            fontname='arial',
            style='filled', 
            fontcolor=font_color,
            fillcolor=color_value
        )

    for source, targets in source_to_targets.items():
        for target in targets:
            if source in node_to_value and target in node_to_value:
                g.add_edge(node_to_label[source], node_to_label[target])
    return g


