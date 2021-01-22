import json
from os.path import join
import h5py
import sys
from collections import defaultdict

from . import the_ontology
from .graph_lib.graph import DirectedAcyclicGraph


def load(features, rsrc_loc):
    labels_f = join(
        rsrc_loc, 
        'resources',
        'training_set',
        'labels.json'
    )
    studys_f = join(
        rsrc_loc, 
        'resources',
        'training_set',
        'experiment_to_study.json'
    )
    tags_f = join(
        rsrc_loc, 
        'resources',
        'training_set',
        'experiment_to_tags.json'
    )
    expr_matrix_f = join(
        rsrc_loc, 
        'resources',
        'training_set',
        '{}.h5'.format(features)
    )

    # Load the ontology
    og = the_ontology.the_ontology()

    # Load labels and labels-graph
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)
        source_to_targets= labels_data['label_graph']
        exp_to_labels = labels_data['labels']
    label_graph = DirectedAcyclicGraph(source_to_targets)

    # Map each ontology term label to its human-readable
    # name
    label_to_name = {
        label: og.id_to_term[label].name
        for label in source_to_targets.keys()
    }

    # Map each experiment to its most-specific labels
    exp_to_ms_labels = {
        exp: label_graph.most_specific_nodes(labels)
        for exp, labels in exp_to_labels.items()
    }
    
    # Load study metadata
    with open(studys_f, 'r') as f:
        exp_to_study = json.load(f)
    study_to_exps = defaultdict(lambda: set())
    for exp, study in exp_to_study.items():
        study_to_exps[study].add(exp)
    study_to_exps = dict(study_to_exps)

    # Load technical tags
    with open(tags_f, 'r') as f:
        exp_to_tags = json.load(f)

    # Load the data matrix
    print('Loading expression data from {}...'.format(expr_matrix_f))
    with h5py.File(expr_matrix_f, 'r') as f:
        the_exps = [
            str(x)[2:-1]
            for x in f['experiment'][:]
        ]
        gene_ids = [
            str(x)[2:-1]
            for x in f['gene_id'][:]
        ]
        data_matrix = f['expression'][:]
    print('Loaded matrix of shape {}'.format(data_matrix.shape))
    print('done.')

    # Map each experiment to its index
    exp_to_index = {
        exp: ind
        for ind, exp in enumerate(the_exps)
    }
    return (
        og,
        label_graph,
        label_to_name,
        the_exps,
        exp_to_index,
        exp_to_labels,
        exp_to_tags,
        exp_to_study,
        study_to_exps,
        exp_to_ms_labels,
        data_matrix,
        gene_ids
    )


def load_sparse_dataset():
    labels_f = join(data_dir, 'labels.json')
    studys_f = join(data_dir, 'experiment_to_study.json')
    tags_f = join(data_dir, 'experiment_to_tags.json')
    expr_matrix_f = join(data_dir, '{}.h5'.format(features))

    # Load the ontology
    og = the_ontology.the_ontology()

    # Load labels and labels-graph
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)
        source_to_targets= labels_data['label_graph']
        exp_to_labels = labels_data['labels']
    label_graph = DirectedAcyclicGraph(source_to_targets)

    # Map each ontology term label to its human-readable
    # name
    label_to_name = {
        label: og.id_to_term[label].name
        for label in source_to_targets.keys()
    }

    # Map each experiment to its most-specific labels
    exp_to_ms_labels = {
        exp: label_graph.most_specific_nodes(labels)
        for exp, labels in exp_to_labels.items()
    }

    # Load study metadata
    with open(studys_f, 'r') as f:
        exp_to_study = json.load(f)
    study_to_exps = defaultdict(lambda: set())
    for exp, study in exp_to_study.items():
        study_to_exps[study].add(exp)
    study_to_exps = dict(study_to_exps)

    # Load technical tags
    with open(tags_f, 'r') as f:
        exp_to_tags = json.load(f)

    # Load the data matrix
    print('Loading expression data from {}...'.format(expr_matrix_f))
    with h5py.File(expr_matrix_f, 'r') as f:
        the_exps = [
            str(x)
            for x in f['experiment'][:]
        ]
        gene_ids = [
            str(x)
            for x in f['gene_id'][:]
        ]
        data_matrix = f['expression'][:]
    print('Loaded matrix of shape {}.'.format(data_matrix.shape))
    print('done.')

    # Map each experiment to its index
    exp_to_index = {
        exp: ind
        for ind, exp in enumerate(the_exps)
    }
    return (
        og,
        label_graph,
        label_to_name,
        the_exps,
        exp_to_index,
        exp_to_labels,
        exp_to_tags,
        exp_to_study,
        study_to_exps,
        exp_to_ms_labels,
        data_matrix,
        gene_ids
    )

if __name__ == '__main__':
    main()
