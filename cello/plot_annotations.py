import pygraphviz
import matplotlib as mpl

from . import cello
from . import ontology_utils as ou

def plot_annotations(cell, results_df, rsrc_loc):
    label_graph = cello._retrieve_label_graph(rsrc_loc) # This should move to a utility function

    is_term_ids = 'CL:' in results_df.columns[0]

    label_to_prob = {
        label: prob
        for label, prob in zip(results_df.index, results_df.loc[cell])
    }

    if is_term_ids:
        label_to_name = {
            label: ou.get_term_name(label)
            for label in label_to_prob
        }

    _render_graph(
        label_graph.source_to_targets,
        label_to_name,
        "Probabilities for {}".format(cell),
        label_to_prob
    )

def _diff_dot(
        source_to_targets,
        node_to_label,
        metric_name,
        node_to_value
    ):
    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)

    print(node_to_value)
    for node in all_nodes:
        if node not in node_to_value:
            continue

        cmap = mpl.cm.get_cmap('viridis')
        rgba = cmap(node_to_value[node])
        color_value = mpl.colors.rgb2hex(rgba)

        if node_to_color_intensity[node] > 0.5:
            font_color = 'black'
        else:
            font_color = 'white'

        g += '"%s\n%s = %f" [style=filled,  fillcolor="%s", fontcolor=%s, fontname = "arial", label="%s\n%s = %f"]\n' % (
            node_to_label[node],
            metric_name,
            node_to_value[node],
            color_value,
            font_color,
            node_to_label[node],
            metric_name,
            node_to_value[node]
        )
    for source, targets in source_to_targets.items():
        for target in targets:
            if source in node_to_value and target in node_to_value:
                g += '"%s\n%s = %f" -> "%s\n%s = %f"\n' % (
                    node_to_label[source],
                    metric_name,
                    node_to_value[source],
                    node_to_label[target],
                    metric_name,
                    node_to_value[target]
                )
    g += "}"
    return g


def _render_graph(
        source_to_targets,
        node_to_label,
        metric_name,
        node_to_value
    ):

    g = Digraph(comment='The Round Table')

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

        if node_to_color_intensity[node] > 0.5:
            font_color = 'black'
        else:
            font_color = 'white'

        g.node(
            node_to_label[node],
            label=node_to_label[node],
            attrs={
                'fontname': 'arial',
                'style': 'filled', 
                'fontcolor': font_color,
                'fillcolor': '"{}"'.format(color_value)
            }
        )

    for source, targets in source_to_targets.items():
        for target in targets:
            if source in node_to_value and target in node_to_value:
                g.edge(node_to_label[source], node_to_label[target])

    g.view()
