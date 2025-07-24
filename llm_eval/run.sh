#!/bin/bash
#PJM -L rscgrp=share-debug
#PJM -L jobenv=singularity
#PJM -g gk77
#PJM -j
#PJM -L elapse=00:30:00
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
  --mode_checkpoint ../writing_mode_classifier"

code_dir=$(pwd)/..

singularity exec --nv --home ${code_dir}/container/kelvin --workdir ${code_dir} --bind ${code_dir}:/tmp/code  ${code_dir}/style_project.sif bash -c "cd /tmp/code && ${convert_format} && ${annotate_mode}"