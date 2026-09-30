"""Run the UNCHANGED scorer and record Windows whole-process peak working set.

Usage: python measure_eval.py --checkpoint PATH --split validation --output PATH
All command-line options are passed to evaluate.py. Run models sequentially.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    evaluator = Path(__file__).with_name('evaluate.py')
    sys.argv[0] = str(evaluator)
    runpy.run_path(str(evaluator), run_name='__main__')
    result = {'checkpoint_bytes': args.checkpoint.stat().st_size,
              'measurement': 'Windows PeakWorkingSetSize; entire evaluator process including loading'}
    if sys.platform == 'win32':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        result['peak_working_set_bytes'] = counters.PeakWorkingSetSize
        result['peak_working_set_gib'] = counters.PeakWorkingSetSize / 2**30
    else:
        raise RuntimeError('This measurement wrapper currently supports Windows only.')
    args.output.with_suffix('.resources.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
