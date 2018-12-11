#!/usr/bin/env python3

import os
import sys
import bpy

import json


# bpy.context.scene.render.engine = 'CYCLES'

def delete_object(name):
    # deselect all
    bpy.ops.object.select_all(action='DESELECT')
    # selection
    bpy.data.objects[name].select = True
    # remove it
    bpy.ops.object.delete()


def scale_obj(obj, tup):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.ops.transform.resize(value=tup)


def main():
    RENDER_SETTINGS_FN = 'render_settings.json'
    with open(RENDER_SETTINGS_FN, 'r') as fin:
        settings = json.load(fin)

    delete_object('Cube')

    OBJ_FILENAME = settings['obj_filename']
    OUTPUT_FILENAME = settings['output_filename']

    OBJ_NAME = os.path.splitext(os.path.basename(OBJ_FILENAME))[0]

    bpy.ops.import_scene.obj(filepath=OBJ_FILENAME)

    obj = bpy.data.objects[OBJ_NAME]
    scale_obj(obj, (2.0, 2.0, 2.0))

    bpy.data.scenes['Scene'].render.filepath = OUTPUT_FILENAME
    bpy.ops.render.render(write_still=True)

main()
