"""Final backbone recipe: 12,000 steps x 32 x 256 = 98,304,000 targets."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/final.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/local-train')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=.2)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--save-best', action='store_true',
                   help='Save best_validation.pt and checkpoints at validation intervals.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.lr <= 0 or args.weight_decay < 0:
        p.error('Require positive lr and nonnegative weight-decay.')
    recipe = dict(lr=args.lr, weight_decay=args.weight_decay, betas=[.9,.999],
                  warmup=100, min_lr_ratio=.1, grad_clip=1.)
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.run_dir/'source_snapshot'
    source_dir.mkdir(exist_ok=True)
    for source in ('student.py', 'model.py', 'train.py', 'common.py', 'evaluate.py'):
        shutil.copy2(ROOT/source, source_dir/source)
    (source_dir/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (args.run_dir/'command.json').write_text(json.dumps(sys.argv, indent=2)+'\n')

    def save_checkpoint(path, trained_steps):
        # Copy tensors without moving the live model off the GPU.
        torch.save({'protocol': PROTOCOL, 'implementation': args.implementation,
                    'config': config, 'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'seed': args.seed, 'training_recipe':recipe,
                    'train_tokens': trained_steps*args.batch_size*256}, path)

    best_bpb, best_step = float('inf'), None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = args.lr * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            if args.save_best:
                save_checkpoint(args.run_dir/f'step-{step+1}.pt', step+1)
                if intermediate['bpb'] < best_bpb:
                    best_bpb, best_step = intermediate['bpb'], step+1
                    save_checkpoint(args.run_dir/'best_validation.pt', step+1)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    if args.save_best and validation['bpb'] < best_bpb:
        best_bpb, best_step = validation['bpb'], args.steps
        save_checkpoint(args.run_dir/'best_validation.pt', args.steps)
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,'training_recipe':recipe,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'steps': args.steps, 'batch_size': args.batch_size, 'training_recipe':recipe,
              'best_validation_bpb': best_bpb if args.save_best else None,
              'best_validation_step': best_step,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
