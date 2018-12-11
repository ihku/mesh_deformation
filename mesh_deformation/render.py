#!/usr/bin/env python3

import subprocess as sp
import os
import json

RENDER_SETTINGS_FN = 'render_settings.json'


def render(obj_filename, output_filename):
    with open(RENDER_SETTINGS_FN, 'w') as fout:
        settings = {
            'obj_filename': str(obj_filename),
            'output_filename': str(output_filename),
        }
        json.dump(settings, fout)

    render_path = os.path.join(os.path.dirname(__file__), '_render.py')
    try:
        sp.run(['blender', '-b', '--python', render_path], check=True)
    finally:
        os.unlink(RENDER_SETTINGS_FN)
