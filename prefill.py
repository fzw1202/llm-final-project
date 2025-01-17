import math
import torch
from torch import nn
from transformers.activations import gelu_new
from transformers import GPT2Config
from sampler import Sampler

CONFIG_PATH = "model/config.json"
BIN_PATH = "model/gpt2_pytorch_model.bin"


def last_position(attention_mask: torch.Tensor):
    reversed_attention_mask = attention_mask.flip(dims=[1])
    last_positions = reversed_attention_mask.argmax(dim=1).tolist()
    seq_length = attention_mask.size(1)
    last_positions = torch.tensor([seq_length - pos - 1 for pos in last_positions])
    return last_positions


class Prefill(nn.Module):
    def __init__(self, device1, device2):
        super().__init__()
        self.device = device1
        self.returnDevice = device2
        self.config = GPT2Config.from_pretrained(CONFIG_PATH)
        self.Param = torch.load(
            BIN_PATH, map_location=torch.device(self.device), weights_only=True
        )
        self.embed_dim = self.config.n_embd
        self.wte = nn.Embedding(
            self.config.vocab_size, self.embed_dim, device=self.device
        )
        self.wpe = nn.Embedding(
            self.config.max_position_embeddings, self.embed_dim, device=self.device
        )
        self.h = nn.ModuleList(
            [
                PrefillBlock(self.config, self.Param, layer_idx=i, device=self.device)
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(
            self.embed_dim, eps=self.config.layer_norm_epsilon, device=self.device
        )
        self.lm_head = nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False, device=self.device
        )
        self.sampler = Sampler(0.1, 0.2, 30)

        self.wte.weight = nn.Parameter(self.Param["wte.weight"])
        self.wpe.weight = nn.Parameter(self.Param["wpe.weight"])
        self.lm_head.weight = self.wte.weight
        self.ln_f.weight = nn.Parameter(self.Param["ln_f.weight"])
        self.ln_f.bias = nn.Parameter(self.Param["ln_f.bias"])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        position_ids = torch.arange(
            0, input_ids.size(1), dtype=torch.long, device=self.device
        )
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        states = inputs_embeds + position_embeds
        kvCache = []
        for layer in self.h:
            states, kv = layer(states)
            kvCache.append(kv)
        states = self.ln_f(states)
        logits = self.lm_head(states)
        last_positions = last_position(attention_mask)
        last_token_logits = logits[torch.arange(len(last_positions)), last_positions]
        probabilities = torch.softmax(last_token_logits, dim=-1)
        chosed_index = self.sampler.topK(probabilities)
        for i, (key, value) in enumerate(kvCache):
            key = key.to(self.returnDevice)
            value = value.to(self.returnDevice)
            kvCache[i] = (key, value)
        return chosed_index, kvCache


class PrefillBlock(nn.Module):
    def __init__(self, config, modelParameters, layer_idx, device):
        super().__init__()
        self.Param = modelParameters
        self.config = config
        self.embed_dim = self.config.n_embd
        self.device = device
        self.layer_idx = layer_idx
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon, device=self.device
        )
        self.attn = PrefillAttention(
            config, modelParameters, layer_idx=self.layer_idx, device=self.device
        )
        self.ln_2 = nn.LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon, device=self.device
        )
        self.mlp = MLP(
            inner_dim, config, layer_idx, modelParameters, device=self.device
        )

        ln_1_ParaName = f"h.{self.layer_idx}.ln_1"
        self.ln_1.weight = nn.Parameter(self.Param[f"{ln_1_ParaName}.weight"])
        self.ln_1.bias = nn.Parameter(self.Param[f"{ln_1_ParaName}.bias"])
        ln_2_ParaName = f"h.{self.layer_idx}.ln_2"
        self.ln_2.weight = nn.Parameter(self.Param[f"{ln_2_ParaName}.weight"])
        self.ln_2.bias = nn.Parameter(self.Param[f"{ln_2_ParaName}.bias"])

    def forward(self, states):
        residual = states
        states = self.ln_1(states)
        attn_out, kv = self.attn(states)
        states = attn_out + residual
        residual = states
        states = self.ln_2(states)
        states = self.mlp(states)
        states = states + residual
        return states, kv


class PrefillAttention(nn.Module):
    def __init__(self, config, modelParameters, layer_idx, device):
        super().__init__()
        self.Param = modelParameters
        self.config = config
        self.embed_dim = config.n_embd
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.layer_idx = layer_idx
        self.device = device
        self.attn = nn.Linear(
            self.config.n_ctx, self.config.n_ctx, bias=True, device=self.device
        )
        self.c_attn = nn.Linear(
            self.embed_dim, 3 * self.embed_dim, bias=True, device=self.device
        )
        self.c_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=True, device=self.device
        )
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        attn_bias_ParaName = f"h.{self.layer_idx}.attn.bias"
        self.attn.weight = nn.Parameter(
            torch.ones(self.config.n_ctx, self.config.n_ctx)
        )
        self.attn.bias = nn.Parameter(
            self.Param[attn_bias_ParaName].squeeze(0).squeeze(0)
        )
        c_attn_ParaName = f"h.{self.layer_idx}.attn.c_attn"
        self.c_attn.weight = nn.Parameter(self.Param[f"{c_attn_ParaName}.weight"].T)
        self.c_attn.bias = nn.Parameter(self.Param[f"{c_attn_ParaName}.bias"])
        c_proj_ParaName = f"h.{self.layer_idx}.attn.c_proj"
        self.c_proj.weight = nn.Parameter(self.Param[f"{c_proj_ParaName}.weight"].T)
        self.c_proj.bias = nn.Parameter(self.Param[f"{c_proj_ParaName}.bias"])

    def forward(self, hidden_states):
        batch_size, seq_len, embed_dim = hidden_states.size()
        hidden_states_reshaped = hidden_states.view(-1, hidden_states.size(-1))
        qkv = self.c_attn(hidden_states_reshaped)
        query, key, value = torch.split(qkv, self.embed_dim, dim=-1)
        cacheKey = key.view(batch_size, seq_len, embed_dim)
        cacheValue = value.view(batch_size, seq_len, embed_dim)
        kvCache = (cacheKey, cacheValue)
        query = query.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.attn.bias[:seq_len, :seq_len]
        bias = bias.unsqueeze(0).unsqueeze(0)
        bias = bias.expand(batch_size, self.num_heads, seq_len, seq_len)
        scores = scores + bias
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        batch_mask = (
            mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        )
        scores = scores.masked_fill(batch_mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, value)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        )
        context = self.c_proj(context)
        output = self.resid_dropout(context)
        return output, kvCache


class MLP(nn.Module):
    def __init__(self, intermediate_size, config, layer_idx, modelParameters, device):
        super().__init__()
        self.embed_dim = config.n_embd
        self.Param = modelParameters
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.c_fc = nn.Linear(intermediate_size, self.embed_dim, bias=True)
        self.c_proj = nn.Linear(self.embed_dim, intermediate_size, bias=True)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

        c_fc_ParaName = f"h.{self.layer_idx}.mlp.c_fc"
        self.c_fc.weight = nn.Parameter(self.Param[f"{c_fc_ParaName}.weight"].T)
        self.c_fc.bias = nn.Parameter(self.Param[f"{c_fc_ParaName}.bias"])
        c_proj_ParaName = f"h.{self.layer_idx}.mlp.c_proj"
        self.c_proj.weight = nn.Parameter(self.Param[f"{c_proj_ParaName}.weight"].T)
        self.c_proj.bias = nn.Parameter(self.Param[f"{c_proj_ParaName}.bias"])

    def forward(self, states):
        states = self.c_fc(states)
        states = self.act(states)
        states = self.c_proj(states)
        states = self.dropout(states)
        return states
