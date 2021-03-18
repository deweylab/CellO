"""\
Classify human cell against the Cell Ontology using CellO.

Here we implement a function for running CellO following the conventions
in Scanpy's external API (https://scanpy.readthedocs.io/en/stable/external/).

Author: Matthew Bernstein
Email: mbernstein@morgridge.org
"""

from anndata import AnnData
import dill
from collections import defaultdict
import pandas as pd
import io
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from .plot_annotations import probabilities_on_graph
from . import ontology_utils as ou
from . import cello as ce

def cello(
        adata: AnnData, 
        clust_key: str = 'leiden',
        rsrc_loc: str = '.',
        algo: str = 'IR',
        out_prefix: str = None,
        model_file: str = None,
        log_dir: str = None,
        term_ids: bool = False,
        remove_anatomical_subterms: list = None
    ):
    """\
    CellO [Bernstein21]_.
    
    Hierarchical cell type classification of human cells with the Cell Ontology.
    
    For more information, tutorials, and bug reports, visit the `CellO
    GitHub page <https://github.com/deweylab/CellO>`__. 
    
    Parameters
    ----------
    adata
        Annotated data matrix. CellO requires that expression data has
        been normalized using log(TPM+1). For droplet-based assays, this
        is equivalent to log(CPM+1).
    clust_key
        Key-name of the cluster annotations in `adata.obs`.
    rsrc_loc
        The path to the CellO resources file. The CellO resources
        contain pre-trained models, gene symbol mappings, and 
        training sets for constructing CellO classifiers. If the 
        CellO resources are not located at this provided location,
        they will be downloaded automatically. The resources require
        approximately 5GB of disk space.
    algo
        The name of the algorithm to use for hierarchical classification
        against the Cell Ontology. Use `IR` for Isotonic Regression and 
        `CLR` for Cascaded Logistic Regression.
    out_prefix
        The output prefix of the trained model. If the pre-trained models
        are not compatible with the input dataset do to model expecting
        different genes than that included in the input data, then a new
        model will be trained using CellO's training set. The output model
        will be written to `<output_pref>.model.dill`.  If `None` the 
        newly trained model will not be saved to disk.
    model_file
        The path to the a trained CellO classifier to use for classification.
        CellO model files end in the suffix `model.dill`.
    log_dir
        Directory in which to write log files. If `None`, no log files will
        be written.
    term_ids
        If `True`, output will use Cell Ontology term ID's. 
        If `False`, output will use human readable cell type names.
    remove_anatomical_subterms   
        A list of Uberon Ontology term ID's used to filter CellO's output
        according to anatomical entities. For example, to blacklist all
        cell type specific only to lung and liver, one would supply the 
        list `['UBERON:0002048', 'UBERON:0002107']`.
    
    Returns
    -------
    Updates `adata.obs` with CellO's output. Specifically, `adata.obs` will
    have two columns for every cell type. A column `<cell_type> (probability)`
    that stores the probability that each cell is of `<cell_type>` and a column
    `<cell_type> (binary)` that stores a 1 if the cell is predicted to be of 
    `<cell_type>` and 0 otherwise.  `adata.obs` will also have a column called
    `Most specific cell type` with the term ID or name (depending on whether
    `term_ids` is set to True or False respectively) of the most-specific cell
    type classification for each cell.
        
    
    Examples
    --------
    >>> from anndata import AnnData
    >>> import scanpy as sc
    >>> import scanpy.external as sce
    >>> adata = sc.datasets.pbmc3k()
    >>> adata.X = adata.X.todense()
    >>> sc.pp.normalize_total(adata, target_sum=1e6)
    >>> sc.pp.log1p(adata)
    >>> sc.pp.pca(adata)
    >>> sc.pp.neighbors(adata)
    >>> sc.tl.leiden(adata, resolution=2.0) # Perform clustering
    >>> adata.var['gene_symbols'] = adata.var.index 
    >>> adata.var = adata.var.set_index('gene_ids') # Set the Ensembl gene ID's as primary gene identifiers
    >>> sce.tl.cello(adata, 'leiden', '.') # Run CellO
    >>> sc.tl.umap(adata)
    >>> sc.pl.umap(adata, color='Most specific cell type') # Create UMAP plot with cells colored by cell type
    """
    
    try:
        import cello as ce
    except ImportError:
        raise ImportError(
            'You need to install the package `cello`: please run `pip install '
            '--user cello` in a terminal.'
        )
    
    # Load the model
    if model_file:
        print('Loading model from {}...'.format(model_file))
        with open(model_file, 'rb') as f:
            mod=dill.load(f)
    else:
        # Load or train a model
        mod = ce._retrieve_pretrained_model(adata, algo, rsrc_loc)
        if mod is None:
            mod = ce.train_model(
                adata, rsrc_loc, algo=algo, log_dir=log_dir
            )
            if out_prefix:
                out_model_f = '{}.model.dill'.format(out_prefix)
                print('Writing trained model to {}'.format(out_model_f))
                with open(out_model_f, 'wb') as f:
                    dill.dump(mod, f)
            else:
                print("No argument to 'out_prefix' was provided. Trained model will not be saved.")
    
    # Run classification
    results_df, finalized_binary_results_df, ms_results_df = ce.predict(
        adata,
        mod,
        algo=algo,
        clust_key=clust_key,
        rsrc_loc=rsrc_loc,
        log_dir=log_dir,
        remove_anatomical_subterms=remove_anatomical_subterms
    )
        
    # Merge results into AnnData object
    if term_ids:
        column_to_term_id = {
            '{} (probability)'.format(c): c
            for c in results_df.columns
        }
        results_df.columns = [
            '{} (probability)'.format(c)
            for c in results_df.columns
        ]
        finalized_binary_results_df.columns = [
            '{} (binary)'.format(c)
            for c in finalized_binary_results_df.columns
        ]
    else:
        column_to_term_id = {
            '{} (probability)'.format(ou.cell_ontology().id_to_term[c].name): c
            for c in results_df.columns
        }
        results_df.columns = [
            '{} (probability)'.format(
                ou.cell_ontology().id_to_term[c].name
            )
            for c in results_df.columns
        ]
        finalized_binary_results_df.columns = [
            '{} (binary)'.format(
                ou.cell_ontology().id_to_term[c].name
            )
            for c in finalized_binary_results_df.columns
        ]
        ms_results_df['most_specific_cell_type'] = [
            ou.cell_ontology().id_to_term[c].name
            for c in ms_results_df['most_specific_cell_type']
        ]

    drop_cols = [
        col
        for col in adata.obs.columns
        if '(probability)' in str(col)
        or '(binary)' in str(col)
        or col == 'Most specific cell type'
    ]
    adata.obs = adata.obs.drop(drop_cols, axis=1)

    finalized_binary_results_df = finalized_binary_results_df.astype(bool).astype(str).astype('category')

    adata.obs = adata.obs.join(results_df).join(finalized_binary_results_df)
    adata.uns['CellO_column_mappings'] = column_to_term_id
    if term_ids:
        adata.obs['Most specific cell type'] = [
            ou.cell_ontology().id_to_term[c].name
            for c in ms_results_df['most_specific_cell_type']
        ]
    else:
        adata.obs['Most specific cell type'] = ms_results_df['most_specific_cell_type']
   

def normalize_and_cluster(
        adata: AnnData, 
        n_pca_components: int = 50, 
        n_neighbors: int = 15,
        cluster_res: float = 1.0
    ):
    """
    Normalize and cluster an expression matrix in units of raw UMI counts.

    Parameters
    ----------
    adata
        Annotated data matrix. Expected units are raw UMI counts.
    n_pca_components (default 50)
        Number of principal components to use when running PCA. PCA is
        is used to reduce noise and speed up computation when clustering.
    n_neighbors (default 15)
        Number of neighbors to use for computing the nearest-neighbors 
        graph. Clustering is performed using community detection on this
        nearest-neighbors graph.
    cluster_res (default 1.0)
        Cluster resolution for the Leiden community detection algorithm.
        A higher resolution produces more fine-grained, smaller clusters.
    """
    try:
        import scanpy as sc
    except ImportError:
        sys.exit("The function 'normalize_and_cluster' requires that scanpy package be installed. To install scanpy, run 'pip install scanpy'")
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=n_pca_components)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.leiden(adata, resolution=cluster_res)


def cello_probs(adata, cell_or_clust, rsrc_loc, p_thresh, width=10, height=10, clust_key=None, dpi=300):
    results_df = adata.obs[[col for col in adata.uns['CellO_column_mappings']]]
    results_df.columns = [
        adata.uns['CellO_column_mappings'][c] for c in results_df.columns
    ]

    # Plot based on cluster ID 
    if clust_key:
        try:
            assert cell_or_clust in set(adata.obs[clust_key])
        except AssertionError:
            raise KeyError(f"Error plotting probabilities on graph. Cluster {clust_key} not found in `adata.obs` columns.")

        # Take subset of results dataframe for each column
        clust_to_indices = defaultdict(lambda: [])
        for index, clust in zip(adata.obs.index, adata.obs[clust_key]):
            clust_to_indices[clust].append(index)

        clusts = sorted(clust_to_indices.keys())
        results_df = pd.DataFrame(
            [
                results_df.loc[clust_to_indices[clust][0]]
                for clust in clusts
            ],
            index=clusts,
            columns=results_df.columns
        )

    g = probabilities_on_graph(
        cell_or_clust,
        results_df,
        rsrc_loc,
        p_thresh=p_thresh
    )

    f = io.BytesIO(g.draw(format='png', prog='dot', args=f'-Gdpi={dpi}'))

    fig, ax = plt.subplots(figsize=(width, height))
    im = mpimg.imread(f)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im)
    plt.show()
    return fig, ax


def write_to_tsv(adata, filename):
    """
    Write CellO's output to a TSV file.
    """
    keep_cols = [
        col 
        for col in adata.obs.columns
        if '(probability)' in col
        or '(binary)' in col
        or 'Most specific cell type' in col
    ]
    df = adata.obs[keep_cols]
    df.to_csv(filename, sep='\t')



