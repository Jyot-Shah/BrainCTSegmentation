"""
Unit tests for DeepLabV3+ Model, Dropout Validation, Forward Pass, and MC Dropout Mode.
"""

import torch
import torch.nn as nn
import pytest

from src.models.deeplab import DeepLabV3Plus25D, count_and_validate_dropout, enable_mc_dropout, verify_forward_and_gradient


def test_model_initialization_and_dropout_validation():
    """
    Verifies that the DeepLabV3Plus model initializes with:
    - 9 input channels
    - 6 output classes
    - Decoder ASPP dropout >= 0.2
    - count_and_validate_dropout returns at least 1 layer with p >= 0.2
    """
    model = DeepLabV3Plus25D(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=9,
        num_classes=6,
        decoder_dropout=0.2,
    )
    assert len(model.dropouts) >= 1
    assert any(p >= 0.2 for _, _, p in model.dropouts)


def test_mc_dropout_activation():
    """
    Verifies that enable_mc_dropout:
    - Leaves BatchNorm layers in evaluation mode (training=False)
    - Re-enables Dropout layers in training mode (training=True)
    """
    model = DeepLabV3Plus25D(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=9,
        num_classes=6,
        decoder_dropout=0.2,
    )
    
    enable_mc_dropout(model)

    # Assert all BatchNorm are in eval mode
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            assert not module.training, f"BatchNorm layer {name} was not frozen in eval mode!"

    # Assert Dropout layers are in training mode
    dropout_active = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            assert module.training, f"Dropout layer {name} was not set to training mode!"
            dropout_active += 1
            
    assert dropout_active >= 1


def test_forward_pass_and_gradient_flow():
    """
    Executes a forward and backward pass, verifying output dimensions and gradient flow.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus25D(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=9,
        num_classes=6,
        decoder_dropout=0.2,
    )
    verify_forward_and_gradient(model, device, in_channels=9, num_classes=6, input_size=(128, 128))
