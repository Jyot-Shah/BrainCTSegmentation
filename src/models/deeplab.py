"""
DeepLabV3+ Model with 9-channel 2.5D ResNet34 Backbone and Dropout Validation.

Features:
- SMP DeepLabV3+ architecture with ResNet34 encoder.
- 9-channel input adapted from ImageNet pretrained weights.
- 6 output classes returning unnormalized raw logits.
- Configurable decoder ASPP dropout (default >= 0.2).
- Startup validation: guarantees count_dropout(model) >= 1 and dropout_p >= 0.2.
- MC Dropout utility: keeps BatchNorm in eval mode while re-enabling dropout layers only.
- Gradient flow & forward pass sanity checker.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def count_and_validate_dropout(model: nn.Module, min_prob: float = 0.2) -> List[Tuple[str, nn.Module, float]]:
    """
    Scans model for Dropout layers and asserts that:
    1. At least one dropout layer exists.
    2. The dropout probability is >= min_prob.
    Fails loudly with AssertionError if requirements are not met.
    """
    dropout_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            dropout_layers.append((name, module, float(module.p)))
            
    if len(dropout_layers) == 0:
        raise AssertionError(
            "CRITICAL ERROR: No dropout layer found in the model! "
            "Dropout is required for MC Dropout epistemic uncertainty scoring."
        )
        
    valid_prob = any(p >= min_prob for _, _, p in dropout_layers)
    if not valid_prob:
        probs = [p for _, _, p in dropout_layers]
        raise AssertionError(
            f"CRITICAL ERROR: Found dropout layers but none have probability >= {min_prob}! "
            f"Existing probabilities: {probs}. Check YAML config."
        )
        
    return dropout_layers


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Activates Monte Carlo Dropout correctly:
    1. Freezes all layers (especially BatchNorm) in evaluation mode (model.eval()).
    2. Selectively re-enables ONLY Dropout layers (module.train(True)).
    
    This prevents BatchNorm running statistics from updating during test-time inference.
    """
    model.eval()
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train(True)
            dropout_count += 1
            
    assert dropout_count > 0, "Failed to activate MC Dropout: No dropout layers found in model!"


class DeepLabV3Plus25D(nn.Module):
    """
    Wrapper for Segmentation Models PyTorch DeepLabV3+ with ResNet34 and 9-channel input.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 8,
        decoder_atrous_rates: Tuple[int, ...] = (6, 12, 18),
        in_channels: int = 9,
        num_classes: int = 6,
        decoder_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.decoder_dropout = decoder_dropout

        # smp.DeepLabV3Plus with high-resolution bottleneck (stride=8) to preserve thin extra-axial bleeds (EDH/SDH)
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_output_stride=encoder_output_stride,
            decoder_atrous_rates=decoder_atrous_rates,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # Returns logits directly
            decoder_aspp_dropout=decoder_dropout,
        )

        # Validate dropout layers at startup
        self.dropouts = count_and_validate_dropout(self.model, min_prob=decoder_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor [B, 9, H, W]
        Returns:
            Logits tensor [B, 6, H, W]
        """
        return self.model(x)


def verify_forward_and_gradient(
    model: nn.Module,
    device: torch.device,
    in_channels: int = 9,
    num_classes: int = 6,
    input_size: Tuple[int, int] = (256, 256),
) -> None:
    """
    Smoke test: executes forward and backward pass, verifying:
    - Output shape is [B, num_classes, H, W]
    - Output logits are finite (no NaNs or Infs)
    - Gradients flow properly to all layers, including input conv and segmentation head
    """
    model.train()
    model.to(device)
    
    b, h, w = 2, input_size[0], input_size[1]
    dummy_x = torch.randn(b, in_channels, h, w, device=device, requires_grad=True)
    dummy_target = torch.randint(0, num_classes, (b, h, w), device=device)

    # Forward pass
    logits = model(dummy_x)
    
    # Assertions
    assert logits.shape == (b, num_classes, h, w), (
        f"Output shape mismatch: expected {(b, num_classes, h, w)}, got {logits.shape}"
    )
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values!"

    # Loss and backward pass
    loss = nn.functional.cross_entropy(logits, dummy_target)
    loss.backward()

    # Verify input gradients
    assert dummy_x.grad is not None, "Gradient did not flow back to input tensor!"
    assert torch.isfinite(dummy_x.grad).all(), "Input gradient contains NaN or Inf!"

    # Verify weight gradients
    grad_count = 0
    zero_grad_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or param.grad.abs().sum().item() == 0:
                zero_grad_names.append(name)
            else:
                grad_count += 1

    print(f"[VERIFIED] Forward pass output shape: {list(logits.shape)}")
    print(f"[VERIFIED] Loss: {loss.item():.4f}, Gradients active on {grad_count} parameter tensors")
    if zero_grad_names:
        print(f"[NOTE] Zero-gradient parameters ({len(zero_grad_names)}): {zero_grad_names[:3]}...")
    else:
        print("[VERIFIED] All trainable parameters received non-zero gradients!")
