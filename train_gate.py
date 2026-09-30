"""Train-only fitting of a frozen-GPT cache gate; validation-only selection.

Two predeclared controls: learned constant and eight-feature MLP (161 parameters).
The gate is fitted using mixed NLL, not the base GPT forward loss.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import torch
from common import load_data, setup, sha, windows
from evaluate import score
from student import build_model


@torch.no_grad()
def extract(model, split, device):
    tokens, byte_count = split
    chunks = []
    for ids, target in windows(tokens,32):
        ids, target = ids.to(device), target.to(device)
        hidden = model.features(ids)
        logp = model.head(hidden).float().log_softmax(-1)
        cache = model.cache_distribution(hidden,ids)
        features = model.gate_features(logp,cache)
        base = logp.gather(-1,target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        retrieved = cache.gather(-1,target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        valid = target != -100
        first = torch.zeros_like(valid)
        first[:,0] = True
        chunks.append((features[valid],base[valid],retrieved[valid].log(),first[valid]))
    return [torch.cat([chunk[i] for chunk in chunks]) for i in range(4)], byte_count


def losses(model, arrays, indices=None):
    features, base, cache, first = arrays if indices is None else [x[indices] for x in arrays]
    wb,wc = model.gate_log_weights(features)
    mixed = torch.logaddexp(base+wb,cache+wc)
    return -torch.where(first,base,mixed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=Path('runs/local-average/average-9.pt'))
    p.add_argument('--out',type=Path,default=Path('runs/gated-cache-s17'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--lr',type=float,default=.001)
    args=p.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        p.error('epochs and lr must be positive.')
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    plan={'seed':17,'epochs':args.epochs,'batch_size':4096,'optimizer':'AdamW',
          'lr':args.lr,'weight_decay':0.01,'gate_kinds':['constant','mlp'],
          'gate_cap':0.5,'initial_lambda':0.05,'theta':10,
          'trainable':'gate only; backbone weights and cache unchanged',
          'selection':'full validation after every epoch; test only after freeze',
          'parent':str(args.parent),'parent_sha256':sha(args.parent)}
    (args.out/'plan.json').write_text(json.dumps(plan,indent=2))
    snapshot=args.out/'source_snapshot'
    snapshot.mkdir()
    for filename in ('student.py','model.py','train_gate.py','common.py','evaluate.py'):
        shutil.copy2(filename,snapshot/filename)
    device,_=setup(args.device,'fp32',4)
    data=load_data()
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    config=dict(parent['config'],variant='gated_cache',gate_kind='mlp',gate_cap=.5,gate_initial=.05)
    torch.manual_seed(17)
    model=build_model(config).to(device).eval()
    incompatible=model.load_state_dict(parent['model'],strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith('gate') for k in incompatible.missing_keys)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_arrays,_=extract(model,data['train'],device)
    val_arrays,val_bytes=extract(model,data['validation'],device)
    mean=train_arrays[0].mean(0)
    std=train_arrays[0].std(0).clamp_min(.01)
    feature_seconds=time.perf_counter()-started
    print(json.dumps({'feature_seconds':feature_seconds,'train_targets':len(train_arrays[0])}),flush=True)
    baseline_nll=-val_arrays[1]
    fixed=torch.logaddexp(val_arrays[1]+math.log(.95),val_arrays[2]+math.log(.05))
    fixed_nll=-torch.where(val_arrays[3],val_arrays[1],fixed)
    results={'plan':plan,'feature_seconds':feature_seconds,
             'train_targets_per_epoch':len(train_arrays[0]),
             'base_validation_bpb':baseline_nll.double().sum().item()/math.log(2)/val_bytes,
             'fixed_validation_bpb':fixed_nll.double().sum().item()/math.log(2)/val_bytes,
             'candidates':[]}
    for kind in plan['gate_kinds']:
        torch.manual_seed(17)
        model=build_model(dict(config,gate_kind=kind)).to(device).eval()
        model.load_state_dict(parent['model'],strict=False)
        model.gate_mean.copy_(mean)
        model.gate_std.copy_(std)
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('gate.'))
        optimizer=torch.optim.AdamW(model.gate.parameters(),lr=args.lr,weight_decay=.01)
        generator=torch.Generator(device=device).manual_seed(17)
        for epoch in range(1,args.epochs+1):
            permutation=torch.randperm(len(train_arrays[0]),generator=generator,device=device)
            for indices in permutation.split(4096):
                optimizer.zero_grad(set_to_none=True)
                loss=losses(model,train_arrays,indices).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.gate.parameters(),1.)
                optimizer.step()
            with torch.no_grad():
                nll=losses(model,val_arrays)
                bpb=nll.double().sum().item()/math.log(2)/val_bytes
                weights=model.gate_log_weights(val_arrays[0])[1].exp()
                row={'kind':kind,'epoch':epoch,'validation_bpb':bpb,
                     'lambda_mean':weights[~val_arrays[3]].mean().item(),
                     'lambda_q10_q50_q90':torch.quantile(weights[~val_arrays[3]],torch.tensor([.1,.5,.9],device=device)).tolist()}
            checkpoint=dict(parent,config=model.config,
                            model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            gate_training={'parent_sha256':sha(args.parent),'seed':17,'epochs':epoch,
                                           'lr':args.lr,'weight_decay':.01,
                                           'processed_targets':epoch*len(train_arrays[0]),'trainable_parameters':sum(p.numel() for p in model.gate.parameters())})
            # Verify frozen backbone exactly, including tied embeddings.
            assert all(torch.equal(checkpoint['model'][k],v) for k,v in parent['model'].items())
            path=args.out/f'{kind}-epoch-{epoch}.pt'
            torch.save(checkpoint,path)
            row['checkpoint']=str(path)
            results['candidates'].append(row)
            print(json.dumps(row),flush=True)
    selected=min((r for r in results['candidates'] if r['kind']=='mlp'),key=lambda r:r['validation_bpb'])
    checkpoint=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
    model=build_model(checkpoint['config']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    verified=score(model,*data['validation'],device,'fp32')
    verified.pop('window_nll_nats')
    assert abs(verified['bpb']-selected['validation_bpb'])<1e-6
    results['selected_dynamic']=selected
    results['selected_constant']=min((r for r in results['candidates'] if r['kind']=='constant'),key=lambda r:r['validation_bpb'])
    results['verified_dynamic']=verified
    results['backbone_unchanged']=True
    results['new_gate_training_targets']=2*args.epochs*len(train_arrays[0])
    results['process_seconds']=time.perf_counter()-started
    shutil.copy2(selected['checkpoint'],args.out/'checkpoint.pt')
    (args.out/'selection.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
