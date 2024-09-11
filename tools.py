"""
This file contains the useful functions for the greenlist project.
"""
import torch
from torch import Tensor


def tensor_rank_positions(tensor: Tensor) -> Tensor:
    """
    A function to compute the ranks of a tensor based on the position of the elements (in descending order).
    For example, given the input tensor [5, 7, 1, 2, -7], the ranks would be [1, 0, 2, 3, 4].
    :args tensor: A 1D tensor.
    :return: A 1D tensor of ranks.
    """
    assert(tensor.dim() == 1) # Tensor must be 1D.
    
    sorted_indices = torch.argsort(tensor, descending=True)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(tensor))
    return ranks


def rank_difference(ranks_big_model: Tensor, ranks_small_model: Tensor) -> Tensor:
    """
    A function to compute the difference in ranks of tokens between two models.
    :args ranks_big_model: A 1D tensor of ranks for the big model.
    :args ranks_small_model: A 1D tensor of ranks for the small model.
    :return: A 1D tensor of rank differences.
    """
    assert(ranks_big_model.dim() == 1) # Ranks for big model must be 1D.
    assert(ranks_small_model.dim() == 1) # Ranks for small model must be 1D.
    return ranks_big_model - ranks_small_model


def z_score(n_green: int, seq_length: int, fraction: float) -> float:
    """
    A function to compute the z-score of the number of greenlisted tokens in a sequence.
    :args n_green: The number of greenlisted tokens in the sequence.
    :args seq_length: The length of the sequence.
    :args fraction: The fraction of the vocabulary to greenlist.
    :return: The z-score of the number of greenlisted tokens.
    """
    assert(n_green >= 0) # Number of greenlisted tokens must be non-negative.
    assert(seq_length > 0) # Sequence length must be positive.
    assert(n_green <= seq_length) # Number of greenlisted tokens must be less than or equal to sequence length
    assert(0 <= fraction <= 1) # Fraction must be between 0 and 1.
    
    numerator = n_green - fraction * seq_length
    denominator = (fraction * (1 - fraction) * seq_length) ** 0.5
    return numerator / denominator