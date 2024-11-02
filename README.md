Steps (this reduces the orginal steps significantly by elimnating the need to build: a CityEngine scene, ie. a .cej file; a CityEngine rules flie, ie. a .cga file; and a python script):
1. Download CityEngine
2. Prepare assets (3D models) and put them into the assets directory
3. Prepare Maps and put them into the maps directory
4. Make minimal updates and run (via F9 hot key in CityEngine) SIMPL.py to generate data. These updates consist of indicating the asset files, and map files to use for making images.

At this point all data and ground truth annotations are complete

New steps
1. If a YAML dataset is desired, run the SIMPL_TruthYAML.py file, and it will create a complete YAML file from the generated data.
2. If a object detector is desired, a base YOLO11 detector script is avaible in SIMPL_synthetic_detector.py. This can be used to build a detector from a .yaml file.

Blender Support
1. If your 3d assets exist in blender and you need to rotate them, or rotate part of them, our program SIMPL_blender_object_rotater.py can simplify and automate this process, and prepare .obj files that are compatible with CityEngine.

-break-

This repository is a fork/extension of this previous project: 
https://github.com/yangxu351/synthetic_xview_airplanes/

and this paper
[SIMPL: Generating Synthetic Overhead Imagery to Address Zero-shot and Few-Shot Detection Problems](https://arxiv.org/ftp/arxiv/papers/2106/2106.15681.pdf) 
