"""Execute the trusted, bundled notebook's Python cells without Jupyter."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-outputs', action='store_true', help='Update notebook text outputs')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root/'notebooks/01_public_pipeline.ipynb'
    notebook = json.loads(path.read_text())
    os.chdir(root)
    namespace = {'__name__': '__main__'}
    count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(''.join(cell['source']), f'{path.name}:cell{count}', 'exec'), namespace)
        output = buffer.getvalue()
        print(output, end='')
        if args.save_outputs:
            cell['execution_count'] = count
            cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': output.splitlines(True)}] if output else []
    if args.save_outputs:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n')
    print(f'Executed {count} Python cells successfully. No benchmark weights were fitted or saved.')


if __name__ == '__main__':
    main()
