"""Train, average, fit gate, freeze and evaluate the final recipe in a new directory."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from common import ROOT,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('runs/reproduce-s17'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--precision',choices=['auto','fp32','bf16'],default='auto')
    args=p.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    def run(script,*arguments):
        print(f'Running {script}',flush=True)
        with (out/(Path(script).stem+'.log')).open('w') as stream:
            subprocess.run([sys.executable,'-B',str(ROOT/script),*map(str,arguments)],cwd=ROOT,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    def read(path):
        return json.loads(path.read_text())
    train=out/'train'
    run('train.py','--config',ROOT/'configs/final.json','--run-dir',train,'--device',args.device,
        '--precision',args.precision,'--threads',4,'--seed',17,'--steps',12000,'--batch-size',32,
        '--eval-every',600,'--save-best','--lr',.001,'--weight-decay',.20)
    average=out/'average'
    run('average.py','--run-dir',train,'--out',average,'--device',args.device)
    selected=read(average/'selection.json')['selected']['checkpoint']
    gate=out/'gate'
    run('train_gate.py','--parent',selected,'--out',gate,'--device',args.device,'--epochs',10,'--lr',.001)
    final=out/'final.pt'
    shutil.copy2(gate/'checkpoint.pt',final)
    frozen=dict(selected_on='validation only',checkpoint_sha256=sha(final),
                files={f:sha(ROOT/f) for f in ('student.py','model.py','common.py','evaluate.py','train.py','average.py','train_gate.py')})
    (out/'FROZEN.json').write_text(json.dumps(frozen,indent=2))
    # CPU FP32 is the course scoring protocol, independent of training device.
    run('evaluate.py','--checkpoint',final,'--split','test','--device','cpu','--precision','fp32',
        '--threads',4,'--output',out/'test_cpu_fp32.json')
    assert sha(final)==frozen['checkpoint_sha256']
    assert all(sha(ROOT/f)==h for f,h in frozen['files'].items())
    print(json.dumps(read(out/'test_cpu_fp32.json'),indent=2))


if __name__=='__main__':
    main()
