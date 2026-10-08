import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
from pathlib import Path


#Since Encoder and Decoder are symmetric, I define a dataclass to make architectural changes easier to handle
@dataclass
class ConvAEConfig:
    """
    Configuration for the Convolutional Autoencoder.

    The input_* parameters define a single sample's shape (batch dimension is handled
    separately by the DataLoader's batch_size parameter).
    """

    # Input shape for a single sample: (channels, height, width)
    input_channels: int = 1
    input_height: int = 128  # n_mels
    input_width: int = 64    # time frames (2s window, padded to power of 2)

    # Architecture
    latent_dim: int = 64            # Latent space dimension
    base_channels: int = 32         # First conv layer channels (doubles each layer)
    num_layers: int = 4             # Number of conv/deconv layers
    kernel_size: int = 3            # Kernel size for all conv layers
    use_batch_norm: bool = True     # Use batch normalization

    # Regularization
    dropout_rate: float = 0.0       # Disabled: AE should fit normal data precisely

    # Training hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5      # L2 regularization
    epochs: int = 100
    patience: int = 15              # Early stopping patience
    min_delta: float = 1e-4         # Minimum improvement for early stopping

    # Scheduler
    use_scheduler: bool = True
    scheduler_factor: float = 0.5   # LR reduction factor
    scheduler_patience: int = 5     # Epochs before reducing LR

    @property
    def encoder_channels(self) -> List[int]:
        """Channel progression for encoder: [32, 64, 128, 256] for num_layers=4."""
        return [self.base_channels * (2 ** i) for i in range(self.num_layers)]

    @property
    def bottleneck_shape(self) -> Tuple[int, int, int]:
        """Shape before flattening: (channels, height, width)."""
        #h = self.input_height // (2 ** self.num_layers)
        h = self.input_height // (2 ** (self.num_layers - 1))  # First layer stride (1,2): works better to preserve harmonics details
        w = self.input_width // (2 ** self.num_layers)
        c = self.encoder_channels[-1]
        return (c, h, w)

    @property
    def bottleneck_size(self) -> int:
        """Flattened size before latent projection."""
        c, h, w = self.bottleneck_shape
        return c * h * w


class Encoder(nn.Module):
    """
    Convolutional encoder that compresses spectrograms to a latent vector.

    Architecture:
        Conv2d(stride=2) -> BatchNorm -> ReLU (repeated num_layers times)
        Flatten -> Linear -> latent_dim
    """

    def __init__(self, config: ConvAEConfig):
        super().__init__()
        self.config = config

        layers = []
        in_channels = config.input_channels

        for i, out_channels in enumerate(config.encoder_channels):
            stride = (1, 2) if i == 0 else (2, 2)  # Preserve freq resolution on first layer

            layers.append(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=config.kernel_size,
                    stride=stride,
                    padding=config.kernel_size // 2
                )
            )

            if config.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # Add dropout after activation (except last layer to preserve information flow to bottleneck)
            if config.dropout_rate > 0 and i < len(config.encoder_channels) - 1:
                layers.append(nn.Dropout2d(config.dropout_rate))

            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(config.bottleneck_size, config.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    """
    Convolutional decoder that reconstructs spectrograms from latent vectors.

    Architecture:
        Linear -> Reshape
        ConvTranspose2d(stride=2) -> BatchNorm -> ReLU (repeated num_layers-1 times)
        ConvTranspose2d(stride=2) -> Sigmoid (final layer)
    """

    def __init__(self, config: ConvAEConfig):
        super().__init__()
        self.config = config

        self.fc = nn.Linear(config.latent_dim, config.bottleneck_size)
        self.unflatten_shape = config.bottleneck_shape

        layers = []
        channels = list(reversed(config.encoder_channels))

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            layers.append(
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=config.kernel_size,
                    stride=2,
                    padding=config.kernel_size // 2,
                    output_padding=1
                )
            )
            if config.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))

            # Add dropout (except last deconv layer before final reconstruction)
            if config.dropout_rate > 0 and i < len(channels) - 2:
                layers.append(nn.Dropout2d(config.dropout_rate))

        self.deconv_layers = nn.Sequential(*layers)

        self.final_conv = nn.ConvTranspose2d(
            channels[-1], config.input_channels,
            kernel_size=config.kernel_size,
            stride=(1, 2),
            padding=config.kernel_size // 2,
            output_padding=(0, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, *self.unflatten_shape)
        x = self.deconv_layers(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x


class ConvAE(nn.Module):
    """
    Convolutional Autoencoder combining encoder and decoder.

    Usage:
        config = ConvAEConfig(latent_dim=128)
        model = ConvAE(config)

        # Forward pass (reconstruction)
        x_recon = model(x)

        # Get latent representations
        z = model.encode(x)

        # Compute per-sample reconstruction error
        errors = model.reconstruction_error(x)
    """

    def __init__(self, config: ConvAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Compute reconstruction error (MSE) for each sample.

        Args:
            x: Input tensor of shape (batch, 1, height, width)
            reduction: "mean" (default), "sum", or "none"

        Returns:
            Error tensor of shape (batch,) for mean/sum, or full shape for none
        """
        x_recon = self.forward(x)
        error = (x - x_recon) ** 2

        if reduction == "none":
            return error
        elif reduction == "sum":
            return error.view(x.size(0), -1).sum(dim=1)
        else:
            return error.view(x.size(0), -1).mean(dim=1)


# Training Helper Functions

def train_epoch(model: ConvAE, train_loader, optimizer, criterion, device) -> float:
    """
    Train the model for one epoch.

    Args:
        model: ConvAE model
        train_loader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to use

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        x = batch['spectrogram'].to(device)

        optimizer.zero_grad()
        x_recon = model(x)
        loss = criterion(x_recon, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def validate(model: ConvAE, val_loader, criterion, device) -> float:
    """
    Compute validation loss.

    Args:
        model: ConvAE model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    for batch in val_loader:
        x = batch['spectrogram'].to(device)
        x_recon = model(x)
        loss = criterion(x_recon, x)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(val_loader.dataset)


def save_checkpoint(
    model: ConvAE,
    optimizer,
    config: ConvAEConfig,
    history: Dict,
    path: Path,
    scheduler=None
):
    """
    Save model checkpoint.

    Args:
        model: Trained ConvAE model
        optimizer: Optimizer instance
        config: Model configuration
        history: Training history dict
        path: Path to save checkpoint
        scheduler: Optional LR scheduler
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'history': history,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    device: torch.device
) -> Tuple[ConvAE, ConvAEConfig, Dict]:
    """
    Load model from checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Tuple of (model, config, history)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = ConvAEConfig(**checkpoint['config'])
    model = ConvAE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    history = checkpoint.get('history', {})

    return model, config, history


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[Path] = None):
    """
    Plot training and validation loss curves.

    Args:
        history: Dict with 'train_loss' and 'val_loss' lists
        save_path: Optional path to save the figure

    Returns:
        Tuple of (figure, axes)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    # Mark best epoch
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    best_val = min(history['val_loss'])
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax.scatter([best_epoch], [best_val], color='g', s=100, zorder=5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('ConvAE Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax
