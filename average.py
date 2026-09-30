"""Validation-only late checkpoint averaging (inspired by SWA, not full SWA).

Reference: https://github.com/timgaripov/swa/blob/master/train.py
Only checkpoints from the same run/initialization may be averaged.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from common import load_data, make_model, setup, sha
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    started = time.perf_counter()
    device, precision = setup(args.device, 'fp32', 4)
    data = load_data()['validation']
    metrics = json.loads((args.run_dir/'metrics.json').read_text())
    best_step = metrics['best_validation_step']
    steps = [v['step'] for v in metrics['validation_history'] if v['step'] <= best_step]
    rows = []
    for count in (1, 3, 5, 7, 9):
        chosen = steps[-count:]
        if len(chosen) < count:
            continue
        paths = [args.run_dir/f'step-{step}.pt' for step in chosen]
        checkpoint = torch.load(paths[-1], map_location='cpu', weights_only=True)
        sums = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in checkpoint['model'].items()}
        for path in paths:
            item = torch.load(path, map_location='cpu', weights_only=True)
            assert item['config'] == checkpoint['config'] and item['seed'] == checkpoint['seed']
            for k, v in item['model'].items():
                sums[k].add_(v.double())
        checkpoint['model'] = {k: (v/count).to(checkpoint['model'][k].dtype) for k, v in sums.items()}
        checkpoint['model']['head.weight'] = checkpoint['model']['token.weight']
        checkpoint['parent_checkpoints'] = [{'path':str(path), 'sha256':sha(path)} for path in paths]
        checkpoint['averaging'] = {'count': count, 'steps': chosen, 'method': 'arithmetic same-trajectory checkpoint average'}
        # Shared trajectory: unique ancestry is max step, NOT sum of checkpoint steps.
        checkpoint['train_tokens'] = best_step*metrics['batch_size']*256
        path = args.out/f'average-{count}.pt'
        torch.save(checkpoint, path)
        model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
        model.load_state_dict(checkpoint['model'])
        result = score(model, *data, device, precision)
        result.pop('window_nll_nats')
        row = {'checkpoint':str(path), 'sha256':sha(path), 'steps':chosen, 'count':count, **result}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    output = {'selected_on':'validation', 'source_run':str(args.run_dir), 'candidates':rows,
              'selected':min(rows,key=lambda row:row['bpb']), 'process_seconds':time.perf_counter()-started}
    (args.out/'selection.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()

