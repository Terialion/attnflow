"""Simple demo model for testing attention tracking."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    """
    Simplified multi-head self-attention module.
    
    Implements basic scaled dot-product attention for demonstration purposes.
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        """
        Initialize attention module.
        
        Args:
            hidden_dim: Dimension of hidden representation
            num_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear projection
        output = self.fc_out(context)
        return output


class SimpleTransformerLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, ff_dim: int = 512):
        """
        Initialize transformer layer.
        
        Args:
            hidden_dim: Model dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
        """
        super().__init__()
        self.self_attention = SimpleAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Self-attention with residual
        attn_out = self.self_attention(x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class SimpleTransformerModel(nn.Module):
    """
    Simple Transformer model for demonstration.
    
    Stacks multiple transformer layers for testing attention tracking.
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 512,
    ):
        """
        Initialize model.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Embedding(max_seq_len, hidden_dim)
        
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Embeddings + positional encoding
        x = self.embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_encoding(positions)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output projection
        output = self.fc_out(x)
        return output
