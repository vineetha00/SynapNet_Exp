import torch
import torch.nn as nn
import torch.nn.functional as F

########################################
# SimpleSSM: local temporal dynamics
########################################
class SimpleSSM(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.dwconv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, D)
        x_t = x.transpose(1, 2)          # (B, D, T)
        conv_out = self.dwconv(x_t)      # (B, D, T)
        conv_out = conv_out.transpose(1, 2)  # (B, T, D)
        gate_vals = torch.sigmoid(self.gate(x))  # (B, T, D)
        return x + gate_vals * conv_out


########################################
# SparseEventAttention:
# salience-gated global mixing
########################################
class SparseEventAttention(nn.Module):
    def __init__(self, dim, heads=4, k_frac=0.25):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0, "dim must be divisible by heads"

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # salience score per token
        self.salience = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

        self.k_frac = k_frac

    def forward(self, x):
        # x: (B, T, D)
        b, n, d = x.shape
        h = self.heads
        hd = self.head_dim

        # salience / event mask
        scores = self.salience(x).squeeze(-1)  # (B, T)
        k = max(1, int(self.k_frac * n))
        topk_vals, _ = torch.topk(scores, k, dim=1)
        thresh = topk_vals[:, -1].unsqueeze(-1)    # (B,1)
        soft_mask = torch.sigmoid((scores - thresh) * 10.0)  # (B,T)

        # project Q,K,V
        q = self.to_q(x)
        k_ = self.to_k(x)
        v = self.to_v(x)

        def split_heads(t):
            # (B,T,D) -> (B,H,T,HD)
            return t.view(b, n, h, hd).transpose(1, 2)
        q = split_heads(q)
        k_ = split_heads(k_)
        v = split_heads(v)

        # scaled dot-prod attention
        attn_logits = torch.matmul(q, k_.transpose(-2, -1)) / (hd ** 0.5)  # (B,H,T,T)

        # bias attention toward salient tokens as keys
        event_bias = torch.log(soft_mask + 1e-6).unsqueeze(1).unsqueeze(1)  # (B,1,1,T)
        attn_logits = attn_logits + event_bias

        attn = F.softmax(attn_logits, dim=-1)  # (B,H,T,T)
        out = torch.matmul(attn, v)            # (B,H,T,HD)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(b, n, d)  # (B,T,D)
        out = self.to_out(out)  # (B,T,D)
        return out, soft_mask   # mask for interpretability


########################################
# ExternalMemory:
# global episodic memory slots
########################################
class ExternalMemory(nn.Module):
    def __init__(self, dim, num_slots=16):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(num_slots, dim))
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B,T,D)
        b, n, d = x.shape
        mem = self.mem.unsqueeze(0).expand(b, -1, -1)  # (B,S,D)

        q = self.to_q(x)    # (B,T,D)
        k = self.to_k(mem)  # (B,S,D)
        v = self.to_v(mem)  # (B,S,D)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)  # (B,T,S)
        attn_weights = F.softmax(attn_logits, dim=-1)                    # (B,T,S)
        mem_context = torch.matmul(attn_weights, v)                      # (B,T,D)
        fused = self.out(mem_context)                                    # (B,T,D)
        return fused


########################################
# SynapBlock:
# combines SSM, sparse attention, memory,
# with learned gates α, β
########################################
class SynapBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, heads=4, k_frac=0.25, mem_slots=16):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)

        self.ssm = SimpleSSM(dim)
        self.attn = SparseEventAttention(dim, heads=heads, k_frac=k_frac)
        self.mem  = ExternalMemory(dim, num_slots=mem_slots)

        # gates
        self.alpha_gate = nn.Linear(dim, dim)
        self.beta_gate  = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        # x: (B,T,D)
        residual = x
        x_norm = self.norm_in(x)

        ssm_out = self.ssm(x_norm)            # (B,T,D)
        attn_out, mask = self.attn(x_norm)    # (B,T,D),(B,T)
        mem_out  = self.mem(x_norm)           # (B,T,D)

        alpha = torch.sigmoid(self.alpha_gate(x_norm))  # (B,T,D)
        beta  = torch.sigmoid(self.beta_gate(x_norm))   # (B,T,D)

        mixed = ssm_out + alpha * attn_out + beta * mem_out  # (B,T,D)

        out = residual + mixed
        out = out + self.ff(out)
        return out, mask


########################################
# SynapNet backbone:
# stacked SynapBlocks + token/pos embedding.
# Supports LM mode or CLS mode.
########################################
class SynapNet(nn.Module):
    def __init__(self,
                 dim=128,
                 depth=4,
                 vocab_size=30522,
                 max_len=512,
                 num_classes=None,
                 heads=4,
                 k_frac=0.25,
                 mem_slots=16):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim))

        self.blocks = nn.ModuleList([
            SynapBlock(dim, heads=heads, k_frac=k_frac, mem_slots=mem_slots)
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(dim)

        if num_classes is None:
            self.head = nn.Linear(dim, vocab_size)
            self.is_lm = True
        else:
            self.head = nn.Linear(dim, num_classes)
            self.is_lm = False

    def forward(self, idx):
        # idx: (B,T) token ids
        b, n = idx.shape
        x = self.token_embed(idx) + self.pos_embed[:, :n, :]  # (B,T,D)

        debug_masks = []
        for block in self.blocks:
            x, mask = block(x)
            debug_masks.append(mask)

        x = self.norm_out(x)

        if self.is_lm:
            logits = self.head(x)         # (B,T,V)
        else:
            cls_vec = x[:, 0, :]          # (B,D)
            logits = self.head(cls_vec)   # (B,C)

        return logits, debug_masks


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, seq_len, batch_size = 1000, 32, 2
    model = SynapNet(dim=64, depth=3, vocab_size=vocab_size, max_len=512,
                     heads=4, k_frac=0.3, mem_slots=8)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, debug_masks = model(dummy_input)
    print("logits shape:", logits.shape)
    print("mask shape:", debug_masks[0].shape)
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
    print("loss:", loss.item())
