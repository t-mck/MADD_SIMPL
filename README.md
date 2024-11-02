Steps:
1. Download CityEngine
2. Prepare assets (3D models) and put them into the assets directory
3. Prepare Maps and put them into the maps directory
4. Make minimal updates and run (via F9 hot key in CityEngine) SIMPL.py to generate data. These updates consist of indicating the asset files, and map files to use for making images.

At this point all data and ground truth annotations are complete

If a YAML dataset is desired, run the SIMPL_TruthYAML.py file.

If a object detector is desired, a base YOLO11 detector script is avaible in SIMPL_synthetic_detector.py

-break-

This repository is a fork/extension of this previous project: 

https://github.com/yangxu351/synthetic_xview_airplanes/

and this paper

[SIMPL: Generating Synthetic Overhead Imagery to Address Zero-shot and Few-Shot Detection Problems](https://arxiv.org/ftp/arxiv/papers/2106/2106.15681.pdf) 
