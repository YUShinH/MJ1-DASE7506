"""Verify packaged files against PACKAGE_MANIFEST.json; does not train or download."""
import json
from common import ROOT,sha


def main():
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    bad=[name for name,info in manifest['files'].items()
         if not (ROOT/name).is_file() or sha(ROOT/name)!=info['sha256']]
    if bad:
        raise SystemExit('Missing/changed packaged files: '+', '.join(bad))
    print(f"PASS: {len(manifest['files'])} packaged files match SHA256.")


if __name__=='__main__':
    main()
