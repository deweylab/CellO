######################################################################
#   Load a gene expression matrix
######################################################################

from os.path import join
import pandas as pd
from anndata import AnnData
import h5py
from scipy.io import mmread
import numpy as np

def load_data(
        data_loc, 
        data_type, 
        rows_cells=False,
        hdf5_expr_key=None,
        hdf5_cells_key=None,
        hdf5_genes_key=None
    ):
    if data_type == '10x':
        ad = _read_10x(data_loc)
    elif data_type == 'TSV':
        ad = _read_csv(
            data_loc,
            '\t',
            rows_cells
        )
    elif data_type == 'CSV':
        ad = _read_csv(
            data_loc,
            ',',
            rows_cells
        )
    elif data_type == 'HDF5':
        ad = _read_hdf5(
            data_loc,
            hdf5_expr_key,
            hdf5_cells_key,
            hdf5_genes_key
        )
    else:
        raise ValueError(
            """
            Error! You specified the datatype '{}', which is not an 
            available option. Please use one of the following options 
            to specify your data type: '10x', 'TSV', 'CSV', or 'HDF5'.
            """.format(data_type)
        )
    ad.raw = ad
    return ad

def _read_10x(data_loc):
    # Load genes 
    genes_f = join(data_loc, 'genes.tsv')
    with open(genes_f, 'r') as f:
        genes = [
            l.split()[0].strip()
            for l in f
        ]

    # Load barcodes
    barcodes_f = join(data_loc, 'barcodes.tsv')
    barcodes = pd.read_csv(
        barcodes_f,
        header=None,
        index_col=0
    ).index

    # Load counts
    expr_f = join(data_loc, 'matrix.mtx')
    with open(expr_f, 'rb') as f:
        X = mmread(f).todense().T

    # Build AnnData object
    ad = AnnData(
        X=X,
        obs=pd.DataFrame(
            data=barcodes,
            index=barcodes
        ),
        var=pd.DataFrame(
            data=genes,
            index=genes
        )
    )
    return ad

def _read_csv(data_loc, sep, rows_cells=None):
    df = pd.read_csv(
        data_loc,
        sep=sep,
        index_col=0
    )
    if not rows_cells:
        df = df.transpose()
    ad = AnnData(
        X=np.array(df),
        obs=pd.DataFrame(
            index=df.index,
            data=df.index
        ),
        var=pd.DataFrame(
            index=df.columns,
            data=df.columns
        )
    )
    return ad


def _read_hdf5(
        data_loc,
        expr_key,
        cells_key,
        genes_key
    ):
    with h5py.File(data_loc, 'r') as f:
        cells = [
            str(x)[2:-1]
            for x in f[cells_key][:]
        ]
        genes = [
            str(x)[2:-1]
            for x in f[genes_key][:]
        ]  
        X = np.array(f[expr_key][:])
    ad = AnnData(
        X=X,
        obs=pd.DataFrame(
            index=cells,
            data=cells
        ),
        var=pd.DataFrame(
            index=genes,
            data=genes
        )
    )
    return ad

