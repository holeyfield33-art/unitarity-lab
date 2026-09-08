"""Verify corrected telemetry during actual KV-cached distilgpt2 decoding."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from metric_controls import gap, provenance


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained('distilgpt2', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained('distilgpt2', local_files_only=True,
                                               attn_implementation='eager').float().eval()
    ids = tok('The transformer architecture works by', return_tensors='pt').input_ids
    captured = {}
    def capture(module, args, output):
        captured['source'] = (output[0] if isinstance(output, tuple) else output).detach().clone()
    handle = model.transformer.h[3].register_forward_hook(capture)
    cache = None
    rows = []
    try:
        with torch.inference_mode():
            for step in range(4):
                out = model(input_ids=ids if step == 0 else ids[:, -1:],
                            past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                source = captured['source']
                observed = gap(source)
                flat = source.double().reshape(-1, source.shape[-1])
                if step:
                    reference = float(flat.square().sum())
                    assert flat.shape[0] == 1
                else:
                    eig = torch.linalg.eigvalsh(flat @ flat.T / len(flat))
                    reference = float(eig[-1] - eig[-2])
                relative_error = abs(observed-reference) / max(abs(reference), 1e-12)
                assert relative_error < 1e-4
                # Compare the cached prediction to recomputation on the same prefix.
                full = model(input_ids=ids, use_cache=False)
                cached_logp = out.logits[0, -1].log_softmax(-1)
                full_logp = full.logits[0, -1].log_softmax(-1)
                difference = float((cached_logp-full_logp).abs().max())
                assert difference < 1e-3
                chosen = int(out.logits[0, -1].argmax())
                rows.append({'step': step, 'phase': 'prefill' if step == 0 else 'decode',
                             'source_shape': list(source.shape), 'gap': observed,
                             'exact_reference': reference, 'relative_error': relative_error,
                             'cached_vs_full_max_logprob_difference': difference,
                             'next_token': tok.decode([chosen])})
                ids = torch.cat([ids, torch.tensor([[chosen]])], dim=1)
    finally:
        handle.remove()
    result = {'model': 'distilgpt2', 'model_revision': getattr(model.config, '_commit_hash', None),
              'scope': 'corrected experimental clone, source layer 3, float32 CPU, seed 42',
              'source_file_sha256': provenance(), 'rows': rows}
    result['source_file_sha256']['experiments/decode_probe.py'] = __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
    Path(__file__).with_name('decode_probe.results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
