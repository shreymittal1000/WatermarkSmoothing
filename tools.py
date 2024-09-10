import random
import torch
from torch import Tensor

def generate_soft_greenlist_watermark(vocab_size: int, fraction: float, constant: float) -> Tensor:
    """
    A function to generate a soft greenlist watermark tensor.
    :args vocab_size: The size of the vocabulary.
    :args fraction: The fraction of the vocabulary to greenlist.
    :args constant: The constant value to assign to the greenlisted tokens.
    :return: A 1D tensor of the watermark.
    """
    assert(0 <= fraction <= 1) # Fraction must be between 0 and 1.
    watermarked_tokens = random.sample(range(vocab_size), int(fraction * vocab_size))
    watermark = torch.zeros(vocab_size)
    
    watermark[watermarked_tokens] = constant
    return watermark

def tensor_rank_positions(tensor: Tensor) -> Tensor:
    """
    A function to compute the ranks of a tensor based on the position of the elements (in descending order).
    For example, given the input tensor [5, 7, 1, 2, -7], the ranks would be [2, 1, 3, 4, 5].
    :args tensor: A 1D tensor.
    :return: A 1D tensor of ranks.
    """
    sorted_indices = torch.argsort(tensor, descending=True)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(1, len(tensor) + 1)
    return ranks