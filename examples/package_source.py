"""Create a clean standalone source archive without local data or build files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=Path(args.out)
    receipt=output.with_suffix('.manifest.json')
    if output.exists() or receipt.exists():raise FileExistsError('Choose a new archive path')
    files=[]
    for directory in ('src','tests','configs','reports','examples','assets','.github'):
        for path in (root/directory).rglob('*'):
            relative=path.relative_to(root)
            if any(part in ('__pycache__','.pytest_cache','build','dist') or part.endswith('.egg-info') for part in relative.parts):continue
            if path.suffix=='.pyc' or path.name=='.DS_Store':continue
            if path.is_symlink():raise ValueError(f'Review symbolic link before packaging: {relative}')
            if path.is_file():files.append(path)
    files.extend(root/name for name in ('README.md','LICENSE','pyproject.toml','.gitignore'))
    output.parent.mkdir(parents=True,exist_ok=True)
    checksums={}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            relative=str(path.relative_to(root))
            archive.write(path,Path('open-graph-epv')/relative)
            checksums[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'archive':output.name,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'bytes':output.stat().st_size,'file_count':len(files),'files':checksums,
              'contains_raw_data_or_trained_checkpoints':False}
    receipt.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__=='__main__':main()
