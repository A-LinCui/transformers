# -*- coding: utf-8 -*-
"""
Sparsity functions in PyTorch.
Copyright (c) 2024 Junbo Zhao
"""

# pylint: disable=no-member,invalid-name,redefined-builtin,not-callable,too-many-arguments,unused-argument,missing-function-docstring,abstract-method,cell-var-from-loop,too-many-locals,arguments-differ,unsupported-assignment-operation,too-many-statements,too-many-instance-attributes

from typing import Tuple
import abc

import numpy as np
import torch
from torch import autograd, nn, Tensor
import torch.nn.functional as F
import torch.utils.checkpoint


# PRE-DEFINED HYPER-PARAMETERS
weight_unit = 8
block_w = 8
block_h = 8
sparsity_option = [0, 1, 2, 4, 8]
# The possible preserved weights of the block
sparsity_array = np.array([0, 8, 16, 32, 64])


weight_unit = 4
block_w = 4
block_h = 4
sparsity_option = [0, 1, 2, 4]
sparsity_array = np.array([0, 4, 8, 16])

weight_unit = 8
block_w = 16
block_h = 16
sparsity_option = [0, 1, 2, 4, 8]
sparsity_array = np.array([0, 32, 64, 128, 256])


# ---- Sparsity Utils ----
# ------------------------

def unstructured_weight_prune(weight: Tensor, ratio: float) -> Tensor:
    """
    Unstructured weight pruning based on the absolute value with a predefined ratio.

    Args:
        weight (Tensor): The weight to be pruned.
        ratio (float): The pruning ratio.

    Returns:
        Tensor: The generated mask.

    Examples::
        >>> weight = torch.randn((3, 16, 16))
        >>> mask = unstructured_weight_prune(weight, ratio = 0.5)
    """

    if ratio == 0.:
        return torch.ones_like(weight).type_as(weight)

    num_weight = weight.numel()
    num_prune = int(num_weight * ratio)
    abs_weight = weight.abs()
    threshold = torch.topk(abs_weight.view(-1), num_prune, largest = False)[0].max()
    mask = torch.gt(abs_weight, threshold).type_as(weight)
    return mask


def get_sparse_mask(
    weight: Tensor,
    ratio: float,
    bi_direction: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Calculate the sparsity mask.

    Args:
        weight (Tensor): The weight to be pruned.
        ratio (float): Unstructured pruning ratio.
        bi_direction (bool): Whether apply bi-directional pruning. Default: `True`.

    Returns:
        Tensor: The N:M pruned weight.
        Tensor: The corresponding N:M pruning binary mask.

    Examples::
        >>> module = nn.Conv2d(64, 128, (3, 3))
        >>> pruned_weight, mask = get_sparse_mask(module.weight, ratio = 0.5)
        >>> actual_sparsity = 1 - mask.sum().item() / mask.numel()
    """

    # Step 1: Generate the mask for unstructured pruning
    # The scheme utilises this mask to determine N:M pruning settings
    unstructured_mask = unstructured_weight_prune(weight, ratio = ratio)

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

    # Step 4: Pad the reshaped matrix to integral multiple of the block size
    W_pad = block_w - reshaped_W % block_w
    H_pad = block_h - reshaped_H % block_h
    weight_padded = F.pad(weight_mtx.unsqueeze(0), (0, W_pad, 0, H_pad), value = 0.).squeeze(0)
    unstructured_mask_padded = F.pad(
        unstructured_mask_mtx.unsqueeze(0), (0, W_pad, 0, H_pad), value = 0.).squeeze(0)

    # Step 5: Generate the N:M pruning mask
    mask = torch.zeros_like(weight_padded)  # Initialize the mask
    H_block_num = int((reshaped_H + H_pad) / block_h)
    W_block_num = int((reshaped_W + W_pad) / block_w)

    for i in range(H_block_num):
        for j in range(W_block_num):

            h_left, h_right = i * block_h, (i + 1) * block_h
            w_left, w_right = j * block_w, (j + 1) * block_w

            # Get the weight and mask patches
            weight_sub_mtx = weight_padded[h_left : h_right, w_left : w_right]
            unstructured_mask_sub_mtx = unstructured_mask_padded[h_left : h_right, w_left : w_right]

            # Point 1: Get the best sparsity choice
            preserved_num = unstructured_mask_sub_mtx.sum()
            sparsity_choice_idx = np.argmin(np.abs(preserved_num.item() - sparsity_array))
            sparsity_choice = sparsity_option[sparsity_choice_idx] # The selected N for this patch

            # Point 2: Generate the mask
            def get_n_m_sparse_mask(transpose: bool = False) -> Tuple[Tensor, float]:
                """
                Calculate the sparse mask of the patch.

                Args:
                    transpose (bool): Whether to prune in the reverse direction.

                Returns:
                    Tensor: The generated mask.
                    float: The similarity between the newly generated mask and unstructured mask.
                """
                frac_weight = weight_sub_mtx.T if transpose else weight_sub_mtx
                frac_weight = torch.abs(frac_weight.reshape(-1, weight_unit))
                _, sorted_indices = torch.sort(frac_weight, descending = True)
                sub_mask = torch.zeros_like(frac_weight)
                for k, indices in enumerate(sorted_indices):
                    sub_mask[k][indices[:sparsity_choice]] = 1.
                sub_mask = sub_mask.reshape(block_h, block_w)
                sub_mask = sub_mask.T if transpose else sub_mask
                confidence = (sub_mask == unstructured_mask_sub_mtx).sum().item() / sub_mask.numel()
                return sub_mask, confidence

            if bi_direction:
                sub_mask_1, confidence_1 = get_n_m_sparse_mask(False)
                sub_mask_2, confidence_2 = get_n_m_sparse_mask(True)
                # Select the pruning scheme that has larger similarity with the unstructured pruning
                sub_mask = sub_mask_1 if confidence_1 > confidence_2 else sub_mask_2

            else:
                sub_mask, _ = get_n_m_sparse_mask(False)

            # Update the mask
            mask[h_left : h_right, w_left : w_right] = sub_mask

    # Step 7: Recover the original shape
    mask = mask[: reshaped_H, : reshaped_W]
    mask = mask.reshape(C_out, C_in, H, W) if len(weight.shape) == 4 else mask.reshape(C_out, C_in)
    mask = mask.data

    pruned_weight = mask * weight
    return pruned_weight, mask

# ---- End of Sparsity Utils ----
# -------------------------------


# ---- Sparsity Strategy ----
# ---------------------------

class UnstructuredSparseStrategy(autograd.Function):
    """
    Unstructured sparsity strategy.
    """

    @staticmethod
    def forward(
        ctx,
        weight: Tensor,
        ratio: float,
        mask: Tensor = None,
        update: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Prune the weight in the forward phase.

        Args:
            weight (Tensor): The weight to be pruned.
            ratio (float): The pruning ratio.
            mask (Tensor): The existing generated mask. If not given, a new mask
                           will be generated and utilised. Default: `None`.
            update (bool): Whether to update the mask. Default: `False`.

        Returns:
            Tensor: The weight pruned with the previous mask.
            Tensor: The mask for the backward phase.
        """

        if mask is None or update:
            mask = unstructured_weight_prune(weight, ratio)
        return weight * mask, mask

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_mask: Tensor = None):
        """
        The backward function.
        """
        return grad_output, None, None, None, None, None


class PatchNMSparseStrategy(autograd.Function):
    """
    Patch-based N:M sparsity strategy.
    """

    @staticmethod
    def forward(
        ctx,
        weight: Tensor,
        ratio: float,
        mask: Tensor = None,
        update: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Prune the weight in the forward phase.

        Args:
            weight (Tensor): The weight to be pruned.
            ratio (float): The pruning ratio.
            mask (Tensor): The existing generated mask. If not given, a new mask
                           will be generated and utilised. Default: `None`.
            update (bool): Whether to update the mask. Default: `False`.
            bi_direction (bool): Whether apply bi-directional pruning. Default: `True`.

        Returns:
            Tensor: The weight pruned with the previous mask.
            Tensor: The mask for the backward phase.
        """
        if mask is None or update:
            return get_sparse_mask(weight, ratio, bi_direction = False)
        return weight * mask, mask

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_mask: Tensor = None):
        """
        The backward function.
        """
        return grad_output, None, None, None, None, None


class BidirectionPatchNMSparseStrategy(PatchNMSparseStrategy):
    """
    Bi-directional Patch-based N:M sparsity strategy.
    """

    @staticmethod
    def forward(
        ctx,
        weight: Tensor,
        ratio: float,
        mask: Tensor = None,
        update: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Prune the weight in the forward phase.

        Args:
            weight (Tensor): The weight to be pruned.
            ratio (float): The pruning ratio.
            mask (Tensor): The existing generated mask. If not given, a new mask
                           will be generated and utilised. Default: `None`.
            update (bool): Whether to update the mask. Default: `False`.
            bi_direction (bool): Whether apply bi-directional pruning. Default: `True`.

        Returns:
            Tensor: The weight pruned with the previous mask.
            Tensor: The mask for the backward phase.
        """
        if mask is None or update:
            return get_sparse_mask(weight, ratio, bi_direction = True)
        return weight * mask, mask

# ---- End of Sparsity Strategy ----
# ----------------------------------


# ---- Sparse Module ----
# -----------------------

class SparseConv2d(nn.Conv2d):
    """
    2D sparse convolution.

    Args:
        sparsity_strategy (str): The applied sparsity strategy.
                                 Choices: ["unstructured", "patch_n_m", "bidirection_patch_n_m"].
                                 Default: "unstructured".
        sparsity_schedule (dict): Sparsity schedule. Default: `None`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        # Sparsity Settings
        sparsity_strategy: str = "unstructured",
        sparsity_schedule: dict = None,
        **kwargs
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size,
                         stride = stride,
                         padding = padding,
                         dilation = dilation,
                         groups = groups,
                         bias = bias,
                         padding_mode = padding_mode,
                         **kwargs)

        # Sparsity Settings
        self.sparsity_strategy = sparsity_strategy
        assert sparsity_schedule, "Sparsity schedule not given!!"
        self.sparsity_schedule = sparsity_schedule

        if sparsity_strategy == "unstructured":
            self.sparsity_func = UnstructuredSparseStrategy
        elif sparsity_strategy == "patch_n_m":
            self.sparsity_func = PatchNMSparseStrategy
        elif sparsity_strategy == "bidirection_patch_n_m":
            self.sparsity_func = BidirectionPatchNMSparseStrategy
        else:
            print(f"Invalid sparsity strategy: {sparsity_strategy}")

        self.step = -1
        self.sparsity_rate = None
        self.mask = nn.Parameter(None)

    def forward(self, input: Tensor) -> Tensor:
        """
        Feed-forward phase.

        Args:
            input (Tensor): The input features.

        Returns:
            Tensor: The output feature.
        """
        self.step += self.training
        w = self.get_sparse_weights()
        return F.conv2d(input, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def get_sparse_weights(self) -> Tensor:
        """
        Get the sparse weight.

        Returns:
            Tensor: The pruned weight.
        """
        self.sparsity_rate = self.sparsity_schedule.get(self.step, self.sparsity_rate)
        weight, mask = self.sparsity_func.apply(self.weight,
                                                self.sparsity_rate,
                                                self.mask,
                                                self.step in self.sparsity_schedule)
        self.mask = nn.Parameter(mask)
        return weight

    @property
    def actual_sparse_ratio(self) -> float:
        """
        The current actual sparse ratio.

        Returns:
            float: Current sparsity ratio.
        """
        return 1. - sum(self.mask).sum().item() / self.mask.numel()

    def __return_sparse_weights__(self) -> Tensor:
        """
        Get the sparse weight.

        Returns:
            Tensor: The pruned weight.
        """
        return self.mask.data * self.weight


class SparseLinear(nn.Linear):
    """
    Sparse Linear.

    Args:
        in_features (int): Input channel number.
        out_features (int): Output channel number.
        bias (bool): Default: `True`.
        sparsity_strategy (str): The applied sparsity strategy.
                                 Choices: ["unstructured", "patch_n_m", "bidirection_patch_n_m"].
                                 Default: "unstructured".
        sparsity_schedule (dict): The sparsity schedule settings. Default: `None`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        # Sparsity Settings
        sparsity_strategy: str = "unstructured",
        sparsity_schedule: dict = None,
        **kwargs
    ) -> None:
        super().__init__(in_features, out_features)

        # Sparsity Settings
        self.sparsity_strategy = sparsity_strategy
        assert sparsity_schedule, "Sparsity schedule not given!!"
        self.sparsity_schedule = sparsity_schedule

        if sparsity_strategy == "unstructured":
            self.sparsity_func = UnstructuredSparseStrategy
        elif sparsity_strategy == "patch_n_m":
            self.sparsity_func = PatchNMSparseStrategy
        elif sparsity_strategy == "bidirection_patch_n_m":
            self.sparsity_func = BidirectionPatchNMSparseStrategy
        else:
            print(f"Invalid sparsity strategy: {sparsity_strategy}")

        self.sparsity_rate = None
        self.mask = nn.Parameter(None)
        self.step = -1

    def forward(self, input: Tensor) -> Tensor:
        """
        Feed-forward phase.

        Args:
            input (Tensor): The input features.

        Returns:
            Tensor: The output feature.
        """
        self.step += self.training
        w = self.get_sparse_weights()
        return F.linear(input, w)

    def get_sparse_weights(self) -> Tensor:
        """
        Get the sparse weight.

        Returns:
            Tensor: The pruned weight.
        """
        self.sparsity_rate = self.sparsity_schedule.get(self.step, self.sparsity_rate)
        weight, mask = self.sparsity_func.apply(self.weight,
                                                self.sparsity_rate,
                                                self.mask,
                                                self.step in self.sparsity_schedule)
        self.mask = nn.Parameter(mask)
        return weight

    @property
    def actual_sparse_ratio(self) -> float:
        """
        The current actual sparse ratio.

        Returns:
            float: Current sparsity ratio.
        """
        return 1. - sum(self.mask).sum().item() / self.mask.numel()

    def __return_sparse_weights__(self) -> Tensor:
        """
        Get the sparse weight.

        Returns:
            Tensor: The pruned weight.
        """
        return self.mask.data * self.weight

# ---- End of Sparse Module ----
# ------------------------------
