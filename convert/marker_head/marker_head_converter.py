"""Convert a marker-head decision checkpoint into a Verdict package.

Usage: python marker_head_converter.py <model_dir> <out_dir> <config_json> <graph_name>
Inputs : input_ids [B,L] int64, attention_mask [B,L] int64, marker_pos [B,K] int64,
         marker_mask [B,K] bool, qtype [B] int64
Outputs: logits [B,K] float32 (masked slots = -1e4), act_probs [B,n_act] float32
"""

import json
import os
import sys

import torch
import torch.nn as nn

import compare
import marker_head_probe


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(
            d, nhead, 4 * d, dropout, batch_first=True, norm_first=True
        )
        self.head = (
            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers > 0
            else None
        )
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act)
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


class Wrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act = self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits, torch.softmax(act.float(), -1)


def build_model(cfg, encoder_dir=None):
    from transformers import AutoConfig, AutoModel

    if encoder_dir:
        ecfg = AutoConfig.from_pretrained(encoder_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    return DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)


def export(wrapper, example, out_dir, graph_name):
    out = os.path.join(out_dir, graph_name)
    batch, seq, opts = (
        torch.export.Dim("batch"),
        torch.export.Dim("seq", min=8),
        torch.export.Dim("options", min=2),
    )
    prog = torch.onnx.export(
        wrapper,
        example,
        opset_version=18,
        dynamo=True,
        optimize=True,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_probs"],
        dynamic_shapes={
            "input_ids": {0: batch, 1: seq},
            "attention_mask": {0: batch, 1: seq},
            "marker_pos": {0: batch, 1: opts},
            "marker_mask": {0: batch, 1: opts},
            "qtype": {0: batch},
        },
    )
    prog.save(out, external_data=False)
    return out


def main():
    model_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    cfg_path = os.path.abspath(sys.argv[3])
    graph_name = sys.argv[4] if len(sys.argv) > 4 else "model.onnx"

    from safetensors.torch import load_file

    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(cfg_path))
    encoder_dir = os.path.join(model_dir, "encoder")
    if not os.path.isdir(encoder_dir):
        encoder_dir = model_dir
    model = build_model(cfg, encoder_dir=encoder_dir)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.eval()
    model.encoder.config.reference_compile = False
    wrapper = Wrapper(model).eval()

    n_act = len(cfg["act_costs"]) + 1
    vocab = int(model.encoder.config.vocab_size)
    ids, amask, pos, pmask, qtype = marker_head_probe.cases(n_act, vocab, 0)[0]
    example = (
        torch.from_numpy(ids),
        torch.from_numpy(amask),
        torch.from_numpy(pos),
        torch.from_numpy(pmask),
        torch.from_numpy(qtype),
    )
    out = export(wrapper, example, out_dir, graph_name)

    import onnxruntime as ort

    session = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    compare.report(marker_head_probe.run(wrapper, session, n_act, vocab, 0))


if __name__ == "__main__":
    main()
