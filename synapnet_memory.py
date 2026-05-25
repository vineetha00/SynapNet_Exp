import torch
import torch.nn as nn
import torch.nn.functional as F


class WriteableMemory(nn.Module):
    """
    Episodic memory that is written during the forward pass.
    For each sample in the batch:
      - take token states (B,T,D) and salience mask (B,T)
      - pick top-k salient tokens
      - store their representations into fixed-size memory slots
    Later tokens can attend to those slots.
    """

    def __init__(self, dim, num_slots=8, k_frac=0.05):
        super().__init__()
        self.num_slots = num_slots
        self.k_frac = k_frac
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.val_proj = nn.Linear(dim, dim, bias=False)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def write(self, x, salience):
        """
        x: (B,T,D)
        salience: (B,T)
        returns:
          mem_bank: (B,S,D) where S = num_slots
          topk_idx: (B,k) indices chosen before padding/trunc
        """
        B, T, D = x.shape
        k_tokens = max(1, int(self.k_frac * T))

        topk_vals, topk_idx = torch.topk(salience, k_tokens, dim=1)  # (B,k)

        gathered = []
        mem_bank_list = []
        for b in range(B):
            idxs = topk_idx[b]              # (k,)
            token_states = x[b, idxs]       # (k,D)
            gathered.append(idxs)

            # pad or truncate to num_slots
            if token_states.size(0) < self.num_slots:
                pad = torch.zeros(
                    self.num_slots - token_states.size(0),
                    D,
                    device=token_states.device,
                    dtype=token_states.dtype,
                )
                token_states = torch.cat([token_states, pad], dim=0)
            else:
                token_states = token_states[: self.num_slots]

            mem_bank_list.append(token_states.unsqueeze(0))  # (1,S,D)

        mem_bank = torch.cat(mem_bank_list, dim=0)  # (B,S,D)
        topk_idx = torch.stack(gathered, dim=0)     # (B,k)
        return mem_bank, topk_idx

    def read(self, x, mem_bank):
        """
        x: (B,T,D)
        mem_bank: (B,S,D)
        returns:
          episodic context per timestep: (B,T,D)
        """
        B, T, D = x.shape
        S = mem_bank.size(1)

        Q = self.query_proj(x)         # (B,T,D)
        K = self.key_proj(mem_bank)    # (B,S,D)
        V = self.val_proj(mem_bank)    # (B,S,D)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # (B,T,S)
        attn_w = F.softmax(attn_logits, dim=-1)                          # (B,T,S)
        ctx = torch.matmul(attn_w, V)                                    # (B,T,D)
        return self.out_proj(ctx)


class SimpleSSM(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.dwconv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B,T,D)
        x_t = x.transpose(1, 2)                  # (B,D,T)
        conv_out = self.dwconv(x_t).transpose(1, 2)  # (B,T,D)
        gate_vals = torch.sigmoid(self.gate(x))      # (B,T,D)
        return x + gate_vals * conv_out


class SparseEventAttention(nn.Module):
    """
    Salience-gated attention.
    Produces:
      - attended features
      - a salience mask in [0,1] per token
    """

    def __init__(self, dim, heads=4, k_frac=0.25):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # salience predictor
        self.salience_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

        self.k_frac = k_frac

    def forward(self, x):
        # x: (B,T,D)
        B, T, D = x.shape
        H = self.heads
        Hd = self.head_dim

        sal_raw = self.salience_mlp(x).squeeze(-1)  # (B,T)
        k = max(1, int(self.k_frac * T))
        topk_vals, _ = torch.topk(sal_raw, k, dim=1)
        thresh = topk_vals[:, -1].unsqueeze(-1)     # (B,1)
        soft_mask = torch.sigmoid((sal_raw - thresh) * 10.0)  # (B,T)

        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)

        def split_heads(t):
            # (B,T,D) -> (B,H,T,Hd)
            return t.view(B, T, H, Hd).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (Hd ** 0.5)  # (B,H,T,T)

        bias = torch.log(soft_mask + 1e-6).unsqueeze(1).unsqueeze(1)      # (B,1,1,T)
        attn_logits = attn_logits + bias

        attn = F.softmax(attn_logits, dim=-1)  # (B,H,T,T)
        out = torch.matmul(attn, V)            # (B,H,T,Hd)

        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B,T,D)
        out = self.to_out(out)
        return out, soft_mask  # (B,T,D), (B,T)


class SynapBlockWithEpisodic(nn.Module):
    """
    One block:
    - local temporal modeling (SSM)
    - SparseEventAttention (global bursts + salience mask)
    - write top-k salient tokens into episodic memory
    - read episodic memory back into every timestep
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        heads=4,
        k_frac=0.25,
        episodic_slots=8,
        episodic_write_frac=0.05,
    ):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.ssm = SimpleSSM(dim)
        self.attn = SparseEventAttention(dim, heads=heads, k_frac=k_frac)

        self.epmem = WriteableMemory(
            dim,
            num_slots=episodic_slots,
            k_frac=episodic_write_frac,
        )

        # gates for mixing branches
        self.alpha_gate = nn.Linear(dim, dim)
        self.beta_gate = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        # x: (B,T,D)
        residual = x
        x_norm = self.norm_in(x)

        # local dynamics
        ssm_out = self.ssm(x_norm)  # (B,T,D)

        # sparse event attention -> salience mask
        attn_out, sal_mask = self.attn(x_norm)  # (B,T,D), (B,T)

        # episodic write/read
        mem_bank, topk_idx = self.epmem.write(x_norm, sal_mask)  # (B,S,D), (B,k)
        epi_ctx = self.epmem.read(x_norm, mem_bank)              # (B,T,D)

        alpha = torch.sigmoid(self.alpha_gate(x_norm))  # (B,T,D)
        beta = torch.sigmoid(self.beta_gate(x_norm))    # (B,T,D)

        mixed = ssm_out + alpha * attn_out + beta * epi_ctx  # (B,T,D)
        out = residual + mixed
        out = out + self.ff(out)

        return out, sal_mask, mem_bank, topk_idx


class SynapEpisodicNet(nn.Module):
    """
    Stacked episodic blocks.
    At the end, we form logits for classification by combining:
    - the last token's hidden state
    - a pooled readout of the last block's episodic memory
    This gives the classifier DIRECT access to what was stored earlier.
    """

    def __init__(
        self,
        dim=128,
        depth=4,
        vocab_size=30522,
        max_len=4096,
        heads=4,
        k_frac=0.25,
        episodic_slots=8,
        episodic_write_frac=0.05,
        num_classes=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim))

        self.blocks = nn.ModuleList(
            [
                SynapBlockWithEpisodic(
                    dim,
                    heads=heads,
                    k_frac=k_frac,
                    episodic_slots=episodic_slots,
                    episodic_write_frac=episodic_write_frac,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = nn.LayerNorm(dim)

        # classification head: map combined state -> class logits
        # num_classes can be smaller than vocab_size in v3
        if num_classes is None:
            num_classes = vocab_size
        self.final_proj = nn.Linear(dim, num_classes)

    def forward(self, idx):
        """
        idx: (B,T) integer tokens.
        returns:
          logits: (B,num_classes)
          debug_masks: list[ (B,T) ]  salience per block
          debug_mems:  list[ (B,S,D) ] episodic memory slots per block
          debug_topk:  list[ (B,k) ] indices written into memory per block
        """
        B, T = idx.shape
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]  # (B,T,D)

        debug_masks = []
        debug_mems = []
        debug_topk_idx_list = []

        for block in self.blocks:
            x, sal_mask, mem_bank, topk_idx = block(x)
            debug_masks.append(sal_mask)
            debug_mems.append(mem_bank)
            debug_topk_idx_list.append(topk_idx)

        x = self.norm_out(x)           # (B,T,D)
        final_state = x[:, -1, :]      # (B,D)

        # pull from the *last* block memory
        last_mem_bank = debug_mems[-1]        # (B,S,D)
        mem_pooled = last_mem_bank.mean(dim=1)  # (B,D)

        combined = final_state + mem_pooled   # (B,D)
        logits = self.final_proj(combined)    # (B,num_classes)

        return logits, debug_masks, debug_mems, debug_topk_idx_list
