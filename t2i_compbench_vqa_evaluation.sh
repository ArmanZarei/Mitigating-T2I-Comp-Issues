#!/bin/bash

cd T2I-CompBench
source .venv/bin/activate
cd BLIPvqa_eval
echo "[Bash Script] Starting BLIP_vqa.py"
python BLIP_vqa.py --out_dir="$1"
echo "[Bash Script] Finished BLIP_vqa.py"
