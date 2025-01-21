#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0
python LPPG.py \
    --data_path   './datasets/massive_intent/small_p.json' \
    --mode_gen  'small' \
    --model_name '/root/autodl-tmp/LLM-Research/instructor_large'