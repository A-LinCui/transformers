# -*- coding: utf-8 -*-
"""
Sparse pruning implement in PyTorch.
Copyright (c) 2024 Junbo Zhao
"""

# pylint: disable=no-member,invalid-name,redefined-builtin,not-callable,too-many-arguments,unused-argument,missing-function-docstring,abstract-method,cell-var-from-loop,too-many-locals,arguments-differ,unsupported-assignment-operation,too-many-statements,too-many-instance-attributes

from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.utils.prune import BasePruningMethod


class PatchBasedNM(BasePruningMethod):
    """
    Patch-based N:M sparsity strategy.

    Args:
        M (int): The weight unit.
        patch_size (int): Size of the patch for option determination.
        candidate_N (List[int]): List of candidate N choices.
        bidirection (bool): Whether apply bi-directional pruning.
    """

    EPS = 1e-8

    def __init__(
        self, M: int, patch_size: int, candidate_N: List[int], bidirection: bool, **kwargs
    ) -> None:
        self.M = M
        self.patch_size = patch_size
        self.candidate_N = candidate_N
        self.bidirection = bidirection

        # Possible number of non-pruned weights per patch
        self.candidate_left_num = np.array([n / M * patch_size ** 2 for n in candidate_N])

    def _get_sparse_mask(self, weight: Tensor) -> Tensor:
        """
        Calculate the sparsity mask.

        Args:
            weight (Tensor): The weight to be pruned.

        Returns:
            Tensor: The corresponding N:M pruning binary mask.
        """
        # Step 1: Generate the mask from unstructured pruning
        # The scheme utilises this mask to determine N:M pruning settings
        unstructured_mask = (weight.abs() < self.EPS).int()

        # Step 2: Record the original shape of the weight
        if len(weight.shape) == 4:  # The weight belongs to a 2D-convolution
            C_out, C_in, H, W = weight.shape
            reshaped_W = C_in * H * W
        elif len(weight.shape) == 2:  # The weight belongs to a linear layer
            C_out, C_in = weight.shape
            reshaped_W = C_in
        else:  # The weight is invalid
            raise ValueError(f"Invalid weight shape: {weight.shape}")
        reshaped_H = C_out

        # Step 3: Reshape the weight to a Tensor with shape (C_out, X)
        weight_mtx = weight.reshape(reshaped_H, reshaped_W)
        unstructured_mask_mtx = unstructured_mask.reshape(reshaped_H, reshaped_W)

        # Step 4: Pad the reshaped matrix to integral multiple of the patch size
        W_pad = self.patch_size - reshaped_W % self.patch_size
        H_pad = self.patch_size - reshaped_H % self.patch_size
        weight_padded = F.pad(
                weight_mtx.unsqueeze(0), (0, W_pad, 0, H_pad), value = 0.).squeeze(0)
        unstructured_mask_padded = F.pad(
                unstructured_mask_mtx.unsqueeze(0), (0, W_pad, 0, H_pad), value = 0.).squeeze(0)

        # Step 5: Generate the N:M pruning mask
        mask = torch.zeros_like(weight_padded)  # Initialize the mask
        H_block_num = int((reshaped_H + H_pad) / self.patch_size)
        W_block_num = int((reshaped_W + W_pad) / self.patch_size)

        for i in range(H_block_num):
            for j in range(W_block_num):

                h_left, h_right = i * self.patch_size, (i + 1) * self.patch_size
                w_left, w_right = j * self.patch_size, (j + 1) * self.patch_size

                # Get the weight and mask patches
                weight_sub_mtx = weight_padded[h_left:h_right, w_left:w_right]
                unstructured_mask_sub_mtx = unstructured_mask_padded[h_left:h_right, w_left:w_right]

                # Point 1: Get the best sparsity choice
                preserved_num = unstructured_mask_sub_mtx.sum()
                N_choice_idx = np.argmin(np.abs(preserved_num.item() - self.candidate_left_num))
                N = self.candidate_N[N_choice_idx]

                # Point 2: Generate the mask
                def get_n_m_sparse_mask(transpose: bool = False) -> Tuple[Tensor, float]:
                    """
                    Calculate the sparse mask of the patch.

                    Args:
                        transpose (bool): Whether to prune in the reverse direction.

                    Returns:
                        Tensor: The generated mask.
                        float: The similarity between the newly generated mask and
                               unstructured mask.
                    """
                    frac_weight = weight_sub_mtx.T if transpose else weight_sub_mtx
                    frac_weight = torch.abs(frac_weight.reshape(-1, self.M))
                    _, sorted_indices = torch.sort(frac_weight, descending = True)
                    sub_mask = torch.zeros_like(frac_weight)
                    for k, indices in enumerate(sorted_indices):
                        sub_mask[k][indices[:N]] = 1.
                    sub_mask = sub_mask.reshape(self.patch_size, self.patch_size)
                    sub_mask = sub_mask.T if transpose else sub_mask
                    confidence = (sub_mask == unstructured_mask_sub_mtx).float().mean().item()
                    return sub_mask, confidence

                sub_mask, confidence = get_n_m_sparse_mask(False)

                if self.bidirection:
                    reverse_sub_mask, reverse_confidence = get_n_m_sparse_mask(True)
                    sub_mask = sub_mask if confidence > reverse_confidence else reverse_sub_mask

                # Update the mask
                mask[h_left:h_right, w_left:w_right] = sub_mask

        # Step 7: Recover the original shape
        mask = mask[:reshaped_H, :reshaped_W]
        if len(weight.shape) == 4:
            mask = mask.reshape(C_out, C_in, H, W)
        else:
            mask = mask.reshape(C_out, C_in)

        return mask.data

    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """
        Compute the mask.

        Returns:
            Tensor: The N:M pruning mask.
        """
        return self._get_sparse_mask(t)

    @classmethod
    def apply(cls, module, name, M, patch_size, candidate_N, bidirection):
        return super(PatchBasedNM, cls).apply(
            module, name, M, patch_size, candidate_N, bidirection
        )
