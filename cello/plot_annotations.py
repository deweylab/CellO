"""
Create plots for viewing CellO's output.

Author: Matthew Bernstein mbernstein@morgridge.org 
"""

import pygraphviz
from pygraphviz import AGraph
import matplotlib as mpl

from . import cello
from .graph_lib import graph
from . import ontology_utils as ou


def probabilities_on_graph(
        cell_or_clust,
        results_df, 
        rsrc_loc,
        clust=True,
        root_label=None, 
        p_thresh=0.0
    ):
    """
    cell_or_clust
        The name of a cell or cluster for which to plot the probabilities of it 
        being each cell type in the Cell Ontology.
    results_df
        A DataFrame storing CellO's output probabilities in which rows correspond 
        to cells and columns to cell types.
    rsrc_loc
        The location of the CellO resources directory.
    clust: default True
        If True, `cell_or_clust` is the ID of a cluster.
        If False, `cell_or_clust` is the ID of a cell.
    root_label: default None
        Cell type name or ID. Only plot the subgraph of the Cell Ontology rooted 
        at this cell type.
    p_thresh: default 0.0
        A probabilitiy value. Only plot the subgraph of the Cell Ontology spanning
        cell types for which the output probability exceeds the given probability.
    """

    # TODO this should move to a utility function
    label_graph = cello._retrieve_label_graph(rsrc_loc) 

    # Determine if columns are ontology term ID's or term names
    is_term_ids = 'CL:' in results_df.columns[0]

    # TODO
    cell = cell_or_clust

    # Create subgraph spanning the terms
    span_labels = set([
        label
        for label, prob in zip(results_df.columns, results_df.loc[cell])
        if prob > p_thresh
    ])
    if root_label:
        if not is_term_ids:
            root_id = ou.get_term_id(root_label)
        else:
            root_id = root_label
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


