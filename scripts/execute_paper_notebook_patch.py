#!/usr/bin/env python3
"""Execute report-only notebook cells and print an apply_patch patch to stdout.

The caller applies the patch; this script never rewrites the source notebook.
Plots are exported by the notebook itself into results/figures.
"""
import difflib
import argparse
import json
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell-id', action='append', help='Execute selected report cell IDs only')
    args = parser.parse_args()
    path = Path('fast3r_reproduction.ipynb')
    original = path.read_text(encoding='utf-8')
    notebook = nbformat.reads(original, as_version=4)
    indices = [i for i, cell in enumerate(notebook.cells)
               if cell.cell_type == 'code' and cell.id.startswith('paper-')
               and (not args.cell_id or cell.id in args.cell_id)]
    if not indices or (args.cell_id and set(args.cell_id) != {notebook.cells[i].id for i in indices}):
        raise ValueError('Requested cell ID must identify an existing paper analysis code cell')
    report = nbformat.v4.new_notebook(cells=[notebook.cells[i] for i in indices])
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(report, km=manager, timeout=180,
                            resources={'metadata': {'path': str(Path.cwd())}})
    client.execute()
    for i, cell in zip(indices, report.cells):
        # Keep tables/validation text; exported PNG figures have stable paths.
        for output in cell.get('outputs', []):
            if 'data' in output:
                output.data.pop('image/png', None)
        notebook.cells[i] = cell
    notebook.metadata.kernelspec = {'display_name': 'fast3r', 'language': 'python', 'name': 'python3'}
    nbformat.validate(notebook)
    new = nbformat.writes(notebook) + '\n'
    diff = list(difflib.unified_diff(original.splitlines(True), new.splitlines(True), n=3))
    if len(diff) <= 2:
        raise RuntimeError('No notebook changes generated')
    print('*** Begin Patch')
    print(f'*** Update File: {path}')
    for line in diff[2:]:
        if line.startswith('@@'):
            print('@@')
        else:
            print(line, end='')
    print('*** End Patch')


if __name__ == '__main__':
    main()
