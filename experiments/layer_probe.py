"""Offline, paired distilgpt2 layer intervention pilot (not a benchmark).

One teacher-forced next-token prediction per case/arm. Does not touch InsideAI.
Uses production unitarity metric functions on captured block outputs.
"""
import json
import subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from metric_controls import gap, provenance
from unitarity_labs.core.metrics import manifold_coherence_zeta

CASES = [
    {"id": "capital", "prompt": "The capital of France is", "expected": " Paris"},
    {"id": "arithmetic", "prompt": "1 + 1 =", "expected": " 2"},
    {"id": "grounded", "prompt": "The note says: The box contains a red ball. According to the note, the ball is", "expected": " red"},
    {"id": "user_completion", "prompt": "The transformer architecture works by", "expected": None},
]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained('distilgpt2', local_files_only=True,
                                               attn_implementation='eager').float().eval()
    rows = []
    for case in CASES:
        inputs = tokenizer(case['prompt'], return_tensors='pt')
        expected_ids = tokenizer.encode(case['expected'], add_special_tokens=False) if case['expected'] else []
        if expected_ids and len(expected_ids) != 1:
            raise ValueError('This pilot scores single-token targets only')
        baseline = None
        # gain is applied to L3's residual update, not its entire hidden state.
        for arm, gain in [('baseline', None), ('sham', 1.0), ('attenuate_L3', .9), ('block_L3', 0.)]:
            captures = {}
            handles = []
            try:
                if gain is not None:
                    def intervene(module, args, output):
                        h = output[0] if isinstance(output, tuple) else output
                        altered = args[0] + gain * (h - args[0])
                        return (altered,) + output[1:] if isinstance(output, tuple) else altered
                    handles.append(model.transformer.h[3].register_forward_hook(intervene))
                for i, block in enumerate(model.transformer.h):
                    def capture(module, args, output, index=i):
                        h = output[0] if isinstance(output, tuple) else output
                        captures[index] = (args[0].detach().clone(), h.detach().clone())
                    handles.append(block.register_forward_hook(capture))
                with torch.inference_mode():
                    output = model(**inputs, use_cache=False)
                    logp = output.logits[0, -1].float().log_softmax(-1)
                if baseline is None:
                    baseline = logp.clone()
                top = int(logp.argmax())
                source, sink = captures[3][1], captures[4][1]
                layers = []
                for i, (before, after) in captures.items():
                    prev, curr = before[0, -1], after[0, -1]
                    layers.append({'layer': i, 'hidden_norm': curr.norm().item(),
                                   'residual_delta': (curr-prev).norm().item(),
                                   'relative_delta': ((curr-prev).norm()/prev.norm().clamp_min(1e-12)).item()})
                row = {'case': case['id'], 'arm': arm, 'residual_gain_L3': gain,
                       'prompt': case['prompt'], 'expected_next_token': case['expected'],
                       'top_token': tokenizer.decode([top]), 'top_probability': logp[top].exp().item(),
                       'expected_probability': logp[expected_ids[0]].exp().item() if expected_ids else None,
                       'expected_rank': int((logp > logp[expected_ids[0]]).sum())+1 if expected_ids else None,
                       'kl_baseline_to_arm': (baseline.exp()*(baseline-logp)).sum().item(),
                       'max_logprob_change': (baseline-logp).abs().max().item(),
                       'zeta_source3_sink4': manifold_coherence_zeta(source, sink),
                       'source3_spectral_gap': gap(source), 'layers': layers}
                rows.append(row)
                print(f"{case['id']} / {arm}: top={row['top_token']!r} target_p={row['expected_probability']} zeta={row['zeta_source3_sink4']:.4f}", flush=True)
            finally:
                for handle in handles:
                    handle.remove()
    report = {'evidence_type': 'real-model single-token intervention pilot; no generated-answer hallucination labels',
              'model': 'distilgpt2', 'model_revision': getattr(model.config, '_commit_hash', None),
              'unitarity_sha': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'source_file_sha256': provenance(),
              'seed': 42, 'dtype': 'float32', 'device': 'cpu', 'phase': 'prefill',
              'metric_scope': 'full prompt block outputs; source L3, sink L4; indices zero-based',
              'rows': rows}
    Path(__file__).with_name('layer_probe.results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    shams = [r['max_logprob_change'] for r in rows if r['arm']=='sham']
    assert max(shams) < 1e-4, f'Sham is not numerically equivalent: {shams}'


if __name__ == '__main__':
    main()
