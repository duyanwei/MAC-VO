
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

# Multiple Sequence
python3 Experiment_MACVO.py --config Config/Experiment/MACVO/MACVO_evaluate.yaml

# To enable visualization, add `--useRR`


```

- Evaluation
```bash
# MAC-VO
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
