"""
This file contains useful functions where the watermark is context independent.
"""
import random
import torch
from torch import Tensor
from tools import rank_difference, z_score


def generate_soft_greenlist_watermark_context_independent(vocab_size: int, fraction: float, constant: float) -> Tensor:
    """
    A function to generate a soft greenlist watermark tensor, where the watermark is context independent.
    :args vocab_size: The size of the vocabulary.
    :args fraction: The fraction of the vocabulary to greenlist.
    :args constant: The constant value to assign to the greenlisted tokens.
    :return: A 1D tensor of the watermark.
    """
    assert(vocab_size > 0) # Vocabulary size must be positive.
    assert(0 <= fraction <= 1) # Fraction must be between 0 and 1.
    assert(constant >= 0) # Constant must be non-negative. While this isnt technically needed, negative values would make it a redlist watermark
    
    watermarked_tokens = random.sample(range(vocab_size), int(fraction * vocab_size))
    watermark = torch.zeros(vocab_size)
    
    watermark[watermarked_tokens] = constant
    return watermark


def watermark_checker(watermark: Tensor, sequence: Tensor, threshold: float) -> bool:
    """
    A function to check whether a given sequence is greenlisted based on a watermark and a threshold value.
    :args watermark: A 1D tensor of the watermark.
    :args sequence: A 1D tensor of the sequence.
    :args threshold: The threshold value for the z-score.
    :return: A boolean indicating if the sequence is greenlisted.
    """
    assert(watermark.dim() == 1) # Watermark must be a 1D tensor.
    assert(sequence.dim() == 1) # Sequence must be a 1D tensor.
    assert(len(watermark) == len(sequence)) # Watermark and sequence must have the same length.
    
    # Get the first nonzero value of the watermark token
    val = watermark[watermark.nonzero(as_tuple=True)[0]]
    
    n_green = (watermark/val * sequence).sum().item()
    return z_score(n_green, len(sequence), (watermark/val).sum().item() / len(watermark)) >= threshold


def predict_greenlist_absolute(ranks_big_model: list[Tensor], ranks_small_model: list[Tensor], threshold: int) -> list[bool]:
    """
    A function to predict if a token is greenlisted based on the rank difference between two models.
    :args ranks_big_model: A list of 1D tensors of ranks for the big model.
    :args ranks_small_model: A list of 1D tensors of ranks for the small model.
    :args threshold: The threshold for the rank difference.
    :return: A list of booleans indicating if the token is greenlisted.
    """
    assert(len(ranks_big_model) == len(ranks_small_model)) # The number of ranks must be the same for both models.
    assert(threshold >= 0) # Threshold must be non-negative.
    
    mean_rank_difference = sum([rank_difference(ranks_big_model[i], ranks_small_model[i]) for i in range(len(ranks_big_model))]) / len(ranks_big_model)
    return mean_rank_difference > threshold


def predict_greenlist_confidence(ranks_big_model: list[Tensor], ranks_small_model: list[Tensor]) -> float:
    """
    A function to predict the confidence of the greenlist prediction based on the rank difference between two models.
    :args ranks_big_model: A list of 1D tensors of ranks for the big model.
    :args ranks_small_model: A list of 1D tensors of ranks for the small model.
    :return: The confidence of the greenlist prediction.
    """
    assert(len(ranks_big_model) == len(ranks_small_model)) # The number of ranks must be the same for both models.
    
    confidence = Tensor.new_zeros(ranks_big_model[0].size())
    for rank_big, rank_small in zip(ranks_big_model, ranks_small_model):
        confidence += (rank_difference(rank_big, rank_small) > 0).float().mean().item() / len(ranks_big_model)
        
    return confidence