python preprocess/run_gpt.py --workdir test_data --sample 30        # call GPT to identify dynamic objects
python preprocess/run_dino_sam2.py --workdir test_data              # run GroundingDINO SAM2 to segment dynamic objects
cd Tracking-Anything-with-DEVA/                                     # run DEVA to track dynamic objects
python evaluation/eval_with_detections.py --workdir /projects/illinois/eng/cs/shenlong/personals/david/visualsync/test_data --max_missed_detection_count 9000 --output-dir deva
cd ..
python preprocess/vggt_to_colmap.py --workdir ./test_data --vis_path ./vggt_output --save_colmap     # run VGG-T to get pose estimation and visualization as colmap format