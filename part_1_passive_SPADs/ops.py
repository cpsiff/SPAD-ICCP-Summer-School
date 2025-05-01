import numpy as np
import torch
from scipy import spatial


def nearest_neighbor_inpaint(array: torch.Tensor, hot_pixel_mask: np.ndarray):
    """
    Inpaint based on nearest neighbors

    :param array: of shape (h, w) or (h, w, c)
    :param hot_pixel_mask: of shape (h, w). Binary. 1 indicates where to inpaint
    :return: inpainted array of shape (h,w) or (h, w, c)
    """
    i_ll, j_ll = np.where(1 - hot_pixel_mask)
    tree = spatial.KDTree(list(zip(i_ll.ravel(), j_ll.ravel())))
    query_i_ll, query_j_ll = np.where(hot_pixel_mask)
    query_ll = np.stack([query_i_ll, query_j_ll], axis=-1)

    dd, ii = tree.query(query_ll, workers=-1)
    nearest_ll = tree.data[ii]
    nearest_ll = torch.from_numpy(nearest_ll).int().to(array.device)
    nearest_i_ll, nearest_j_ll = nearest_ll[:, 0], nearest_ll[:, 1]

    array[query_i_ll, query_j_ll] = array[nearest_i_ll, nearest_j_ll]
    return array
