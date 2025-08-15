import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

"""  Phase-Based Rhythm Feature Extraction in Sec 3.2  """
class LinearPhaseEmbedding_1(nn.Module):
    def __init__(self, hidden_dim=768, n_fft=512, hop_length=256, seq_len=29):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seq_len = seq_len

        # self.linear1 = nn.LazyLinear(hidden_dim)
        self.linear2 = nn.LazyLinear(hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, waveform):
        batch, time, features = waveform.shape
        waveform_flat = waveform.view(batch, -1)

        # Use return_complex=True for torch.stft
        stft_output = torch.stft(
            waveform_flat, n_fft=self.n_fft, hop_length=self.hop_length,
            return_complex=True
        )

        # Calculate phase data
        phase_data = torch.angle(stft_output)  # Shape: (batch, freq_bins, time_steps)
        # print("phase_data.shape:", phase_data.shape)
        phase_data = phase_data.permute(0, 2, 1)  # Shape: (batch, time_steps, freq_bins)            

        if phase_data.size(1) > self.seq_len:
            start = (phase_data.size(1) - self.seq_len) // 2
            end = start + self.seq_len
            phase_data = phase_data[:, start:end, :]
        # print("phase_data.shape:", phase_data.shape)
        # x = self.linear1(phase_data)
        # x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        # x = F.relu(x)
        x = self.linear2(phase_data)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        return x


class LPE_1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_emb_1 = LinearPhaseEmbedding_1(config.n_embd)
        self.pos_emb_2 = LinearPhaseEmbedding_1(config.n_embd)
        self.pos_emb_3 = LinearPhaseEmbedding_1(config.n_embd)

    def forward(self, inputs):
        pos_1 = self.pos_emb_1(inputs)
        pos_2 = self.pos_emb_2(inputs)
        pos_3 = self.pos_emb_3(inputs)
        pos_emb = torch.cat([pos_1, pos_2, pos_3], dim=1)
        return pos_emb
    

class GatedConditionalCausalAttention(nn.Module):
    """
    Args:
        config (Config): Configuration containing the embedding dimension (n_embd),
                          number of attention heads (num_heads), sequence length (seq_len),
                          and whether to use bias in query, key, and value (qkv_bias).
    """
    def __init__(self, 
                 n_embd,
                 num_heads,
                 block_size=29,
                 attn_drop=0.1,
                 resid_drop=0.1,
                 ):
        super(GatedConditionalCausalAttention, self).__init__()
        
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.n_head = self.num_heads
        
        # Linear projections for query, key, and value
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)

        # Dropout layers for attention and residual connection
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        # Projection layer to map back to the embedding dimension
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask to prevent looking ahead during attention
        # Create a lower triangular matrix (causal mask)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                .view(1, 1, block_size, block_size))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (B, T, C), where B is the batch size,
                        T is the sequence length, and C is the embedding dimension.
        
        Returns:
            Tensor: Output tensor after applying linear attention and RoPE, with shape (B, T, C).
        """

        B, T, C = x.shape  # B: batch size, T: sequence length, C: embedding dimension
        
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x)) # Residual Connection
        x = self.in_proj(x)
        x = self.act(x)

        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # (B, nh, T, hs)
        v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        t = T // 3
        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:t,:t].repeat(1, 1, 3, 3) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        x = self.out_proj(y * act_res)
        return x

