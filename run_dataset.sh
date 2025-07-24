#!/bin/bash

module load singularity

code_dir=/work/gk77/k77006/repos/writing_mode_project
#repos_dir=~/repos
magictools=/work/gk77/k77006/repos/speaker_identification_project/MagicTools

singularity shell --nv --home ${code_dir}/container/kelvin --workdir ${code_dir} --bind ${code_dir}:/tmp/code,${magictools}:/tmp/magictools  ${code_dir}/python-313-amd64.sif