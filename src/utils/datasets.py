"""
Utils for simulation of datasets
"""
import anndata
import numpy as np


def check_clone_uniqueness(cn_mat):
    norm_mat = np.linalg.norm(cn_mat[:, np.newaxis] - cn_mat, axis=-1)
    non_diagonal = ~np.eye(cn_mat.shape[0], dtype=bool)
    return np.all(norm_mat[non_diagonal] > 0)


def shrink_sequences(data: anndata.AnnData | np.ndarray,
                     by: int = 10, layer: str | None = None) -> np.ndarray:
    """
    Reduces the number of bins by taking the median value over a
    user-defined sized window.
    Parameters
    ----------
    data: AnnData object with chromosomes, or simple numpy array
    by: int, shrinkage factor, i.e. number of bins grouped together (inv-resolution)
    layer: str, if None, adata.X will be used

    Returns
    -------
    numpy array with same number of rows (cells) and reduced number of columns (bins)
    """
    full_sequences: list
    if isinstance(data, anndata.AnnData):
        mat = data.X if layer is None else data.layers[layer]
        full_sequences.append()
        # TODO: complete!

