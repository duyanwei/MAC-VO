
### Docker
```bash
xhost +local:docker; docker run -u root --gpus all --runtime=nvidia -it --rm  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  -v /mnt/DATA/datasets:/data -v /home/yanwei/slam_ws/src/opensource/dp/MAC-VO:/home/macvo/workspace macvo:latest
```

### EuRoC

- Running
```bash
cd workspace
# Single Sequence
python3 MACVO.py --odom Config/Experiment/MACVO/MACVO_evaluate.yaml --data Config/Sequence/EuRoC_MH01.yaml

# Mapping
python3 MACVO.py --odom Config/Experiment/MACVO/MACVO_mapping.yaml --data Config/Sequence/EuRoC_MH01.yaml --noeval

# Multiple Sequence
python3 Experiment_MACVO.py --config Config/Experiment/MACVO/MACVO_evaluate.yaml

# To enable visualization, add `--useRR`
# To disable evaluation online, add `--noeval`


```

- Evaluation
```bash
# MAC-VO (MAC-VO convert poses to body frame if extrinsic provided)
# Single Sequence
python3 -m Evaluation.EvalSeq --spaces Results/MACVO_evaluate\@MH01/06_06_041831\

# Multiple Sequences
python3 -m Evaluation.EvalSeq --dir Results/MACVO_evaluate/06_06_054355/
# python3 -m Evaluation.EvalSeq --spaces Results/MACVO_evaluate\@MH01/06_06_041831\ Resutls/xxx Results/xxx

# Save to csv file.
python3 -m Evaluation.EvalSeq --dir Results/MACVO_evaluate/06_06_054355/  --csv Results/MACVO_evaluate/06_06_054355/eval_euroc.csv

# Convert npy to txt
python3 Evaluation/convert_npy_to_txt.py --dir Results/MACVO_evaluate/06_06_054355/
```

- Visualization
MAC-VO defaults save estimated poses and visual map as `pose.npz` and `tensor_map.npz`
```bash
python3 vis_map.py --dir Results/MACVO_Mapping\@f4_taskdriven/06_13_163713/
```
