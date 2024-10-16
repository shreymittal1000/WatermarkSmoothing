"""
This file contains useful functions where the watermark is context independent.
"""
import random
import torch
from torch import Tensor
from typing import List
from tools import n_smaller, rank_difference, z_score


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


def watermark_checker(watermark: torch.Tensor, sequence: torch.Tensor, threshold: float) -> bool:
    """
    A function to check whether a given sequence is greenlisted based on a watermark and a threshold value.
    :args watermark: A 1D tensor of the watermark.
    :args sequence: A 1D tensor of the sequence.
    :args threshold: The threshold value for the z-score.
    :return: A boolean indicating if the sequence is greenlisted.
    """
    assert(watermark.dim() == 1)  # Watermark must be a 1D tensor.
    assert(sequence.dim() == 1)   # Sequence must be a 1D tensor.
    # assert(len(watermark) == len(sequence))  # Watermark and sequence must have the same length.
    
    # Extract the corresponding non-zero values from the watermark
    corresponding_watermark_values = watermark[sequence]

    # Calculate n_green based on the watermark and sequence values
    n_green = 0
    for vals in corresponding_watermark_values:
      if vals != 0:
        n_green += 1
    
    # Return whether the sequence is greenlisted based on the threshold
    return n_green / sequence.shape[0]


def predict_greenlist_confidence(ranks_big_model: List[Tensor], ranks_small_model: List[Tensor]) -> Tensor:
    """
    A function to predict the confidence scores of the greenlist prediction based on the rank difference between two models.
    :args ranks_big_model: A list of 1D tensors of ranks for the big model.
    :args ranks_small_model: A list of 1D tensors of ranks for the small model.
    :return: The confidence scores of the greenlist prediction.
    """
    assert(len(ranks_big_model) == len(ranks_small_model)) # The number of ranks must be the same for both models.
    
    confidence = torch.zeros(ranks_big_model.shape[1], dtype=torch.float16)
    for rank_big, rank_small in zip(ranks_big_model, ranks_small_model):
        confidence += n_smaller(rank_difference(rank_big, rank_small)) / (ranks_big_model.shape[0] * ranks_big_model.shape[1])
        
    return confidence


def smoothed_logits(confidence: Tensor, logits_big: Tensor, logits_small: Tensor) -> Tensor:
    """
    A function to compute the smoothed logits.
    :args confidence: A 1D tensor of greenlist confidence scores.
    :args logits_big: A 2D tensor of logits for the big model.
    :args logits_small: A 2D tensor of logits for the small model.
    :return: A 2D tensor of smoothed logits.
    """
    assert(confidence.dim() == 1) # Confidence must be 1D.
    assert(logits_big.dim() == 2) # Logits for big model must be 2D.
    assert(logits_small.dim() == 2) # Logits for small model must be 2D.
    assert(logits_big.size() == logits_small.size()) # Logits for big and small model must have the same shape.
    assert(len(confidence) == logits_big.size(1)) # Confidence must have the same length as the number of tokens.
    
    return confidence * logits_small + (1 - confidence) * logits_big
