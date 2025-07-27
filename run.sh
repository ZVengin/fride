#!/bin/bash
#PJM -L rscgrp=share
#PJM -L jobenv=singularity
#PJM -g gk77
#PJM -j
#PJM -L elapse=08:00:00
#PJM -L gpu=1
#PJM -N eval_gpt
#PJM -o run_eval_gpt_split_0

module load singularity

convert_format="python epub2json.py \
  --epub_dir data/smashwords_epub \
  --json_dir data/smashwords_json"

annotate_mode="python annotate_llm_eval_book.py \
  --sour_dir data/smashwords_json \
  --target_dir data/smashwords_wmode \
  --model_checkpoint /tmp/code/writing_mode_classifier/model_checkpoint"

create_eval_set="python create_eval_dataset.py \
  --sour_dir data/smashwords_wmode \
  --target_dir data/eval_set"

code_dir=$(pwd)

#singularity exec --nv --home ${code_dir}/container/kelvin --workdir ${code_dir} --bind ${code_dir}:/tmp/code  ${code_dir}/python-313-amd64.sif bash -c "export WANDB_DISABLED=true && cd /tmp/code/llm_eval && ${convert_format} && ${annotate_mode}"
singularity exec --nv --home ${code_dir}/container/kelvin --workdir ${code_dir} --bind ${code_dir}:/tmp/code  ${code_dir}/python-313-amd64.sif bash -c "export WANDB_DISABLED=true && cd /tmp/code/llm_eval && ${create_eval_set}"
