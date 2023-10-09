import logging

import numpy as np
import pandas as pd
import anndata


def convert_adata_to_conet_input(adata: anndata.AnnData, cc_layer='copy', cn_layer='state',
                                 cell_perc_thresh: float = 0.8) -> pd.DataFrame:
    """
    Converts an AnnData object to a pd.Dataframe which can be used as input to CONET.
    Parameters
    ----------
    adata: anndata.AnnData object
    cc_layer: str, corrected counts layer
    cn_layer: str, copy number estimate layer (for breakpoint candidates selection)
    cell_perc_thresh: float, if more than thresh % of cells show corr counts
        different in abs value more than 3 from previous bin count, select as
        candidate breakpoint. the lower the percentage, the more the breakpoint candidates
    Returns
    -------
    pandas.DataFrame with columns: chr, start, end, candidate_brkp, [<cell_name(s)>, ...]
    """
    # read corrected counts
    obs_df = adata.to_df(layer=cc_layer).transpose()
    # create candidate breakpoints on:
    #   1. copy number change
    clone_cn = adata.to_df(layer=cn_layer).drop_duplicates().transpose()
    # initialize 'candidate_brkp' column with copy number changes
    cn_change = np.zeros(adata.n_vars)
    for c in clone_cn.columns:
        cn_change = np.logical_or(cn_change, clone_cn[c].diff().values != 0)

    # create dataframe
    conet_df = pd.concat([
        adata.var[['chr', 'start', 'end']].reset_index(drop=True),
        pd.DataFrame({'candidate_brkp': cn_change.astype(int)}),
        obs_df.reset_index(drop=True)
    ], axis=1)
    logging.debug(f"copy number changes new breakpoint candidates: {conet_df['candidate_brkp'].sum()}")

    # reduced/synth dataset
    chr_codes = {str(i): i for i in range(1, adata.var['chr'].unique().size + 1)}
    if 'X' in conet_df['chr'].values or 'Y' in conet_df['chr'].values:
        logging.debug(f"mapping autosomes X->23, Y-> 24")
        # real data
        chr_codes['X'] = 23
        chr_codes['Y'] = 24

    conet_df['chr'] = conet_df['chr'].map(chr_codes)
    conet_df['width'] = conet_df['end'] - conet_df['start']

    #   2. beginning and end of chromosomes
    beg_end_chr_loc = (conet_df['chr'].shift(1) != conet_df['chr']) | (conet_df['chr'].shift(-1) != conet_df['chr'])
    conet_df.loc[beg_end_chr_loc, 'candidate_brkp'] = 1
    logging.debug(f'beginning-end chr new candidates: {beg_end_chr_loc.sum()}')

    #   3. high corr counts abs diff evidence (abs difference > 3 in > 80% cells)
    high_diff_mask = (obs_df.diff().abs() > 3).sum(axis=1) > (cell_perc_thresh * obs_df.shape[1])
    conet_df.loc[high_diff_mask.values, 'candidate_brkp'] = 1
    logging.debug(f"high count diff new candidates: {high_diff_mask.sum()}")

    #   4. locus to the right of previously computed candidate
    shifted_cand_brkp_mask = conet_df['candidate_brkp'].shift(1, fill_value=0) == 1
    conet_df.loc[shifted_cand_brkp_mask.values, 'candidate_brkp'] = 1
    logging.debug(f"shift to right new candidates: {shifted_cand_brkp_mask.sum()}")

    return conet_df

