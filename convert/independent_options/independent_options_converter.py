"""Convert an independent-options decision checkpoint into a Verdict package.

Usage: python independent_options_converter.py <model_dir> <out_dir> <graph_name>
Inputs : input_ids [B,S] int64, full_mask [B,1,S,S] float32, sliding_mask [B,1,S,S] float32,
         position_ids [B,S] int64, marker_pos [B,K] int64, marker_mask [B,K] bool
Outputs: logits [B,K] float32 (masked slots = -1e4)
"""

import os
import sys

os.environ.setdefault("VON_CHAINS_DIR", "off")

import torch
import torch.nn as nn

import compare
import independent_options_probe


class VonExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.scorer = model.scorer

    def forward(self, input_ids, full_mask, sliding_mask, position_ids, marker_pos, marker_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask={"full_attention": full_mask, "sliding_attention": sliding_mask},
            position_ids=position_ids,
        )
        h = out.last_hidden_state
        b, k = marker_pos.shape
        reps = torch.gather(h, 1, marker_pos.unsqueeze(-1).expand(b, k, h.size(-1)))
        logits = self.scorer(reps.reshape(b * k, h.size(-1))).reshape(b, k)
        return torch.where(marker_mask, logits, torch.full_like(logits, -1e4))


def export(wrapper, out_dir, graph_name):
    out_path = os.path.join(out_dir, graph_name)
    b, s, k = 2, 64, 3
    example = (
        torch.ones((b, s), dtype=torch.long),
        torch.zeros((b, 1, s, s), dtype=torch.float32),
        torch.zeros((b, 1, s, s), dtype=torch.float32),
        torch.arange(s, dtype=torch.long).unsqueeze(0).repeat(b, 1),
        torch.zeros((b, k), dtype=torch.long),
        torch.ones((b, k), dtype=torch.bool),
    )
    dynamic = {
        "input_ids": {0: "B", 1: "S"},
        "full_mask": {0: "B", 2: "S", 3: "S"},
        "sliding_mask": {0: "B", 2: "S", 3: "S"},
        "position_ids": {0: "B", 1: "S"},
        "marker_pos": {0: "B", 1: "K"},
        "marker_mask": {0: "B", 1: "K"},
        "logits": {0: "B", 1: "K"},
    }
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            out_path,
            input_names=[
                "input_ids",
                "full_mask",
                "sliding_mask",
                "position_ids",
                "marker_pos",
                "marker_mask",
            ],
            output_names=["logits"],
            dynamic_axes=dynamic,
            opset_version=18,
            do_constant_folding=True,
        )

    import onnx

    proto = onnx.load(out_path)
    data_path = f"{out_path}.data"
    os.unlink(out_path)
    if os.path.exists(data_path):
        os.unlink(data_path)
    onnx.save_model(
        proto,
        out_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{graph_name}.data",
        size_threshold=1024,
        # size_threshold keeps tiny tensors (e.g. Unsqueeze axes) inline: ORT shape inference reads them before external data resolves
    )
    size = os.path.getsize(out_path) + os.path.getsize(data_path)
    assert size < 1.7e9, "external data must hold ONE copy of the weights"
    return out_path


def main():
    import contextlib
    import io

    from von.backends.option_marker_backend import OptionMarkerBackend

    model_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    graph_name = sys.argv[3] if len(sys.argv) > 3 else "model.onnx"

    os.makedirs(out_dir, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        backend = OptionMarkerBackend(checkpoint_dir=model_dir, device="cpu")
        model = backend._get_model()
    model.eval()
    wrapper = VonExportWrapper(model).eval()
    out = export(wrapper, out_dir, graph_name)

    import onnxruntime as ort

    session = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    vocab = int(model.encoder.config.vocab_size)
    pad = model.tokenizer.pad_token_id or 0
    sliding = getattr(model.encoder.config, "sliding_window", None)
    compare.report(independent_options_probe.run(wrapper, session, vocab, pad, sliding))


if __name__ == "__main__":
    main()
