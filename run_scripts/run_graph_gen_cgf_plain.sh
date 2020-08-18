#!/bin/bash

hostname
whoami

DATASET=$1
DIMS=$2
NBLOCK=$3
LR=$4
LOGIT=$5

LOG="output/graph_generation/${DATASET}/logs/train_dims-${DIMS}_nblock-${NBLOCK}_lr-${LR}_use-logit-${LOGIT}.txt.`date +'%Y-%m-%d_%H-%M-%S'`.log"
OUTDIR="output/graph_generation/${DATASET}/results/dims-${DIMS}_nblock-${NBLOCK}_lr-${LR}_use-logit-${LOGIT}"

mkdir -p "output/graph_generation/${DATASET}/logs"
mkdir -p ${OUTDIR}

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python src/graph_generation/train_graphflow.py \
                        --graph_type ${DATASET} \
                        --note none \
                        --dims ${DIMS} \
                        --num_blocks ${NBLOCK} \
                        --result_dir ${OUTDIR} \
                        --save ${OUTDIR} \
                        --lr ${LR} \
                        --use_logit ${LOGIT}

echo "DONE"

# example command:
# bash run_graph_gen_cgf_plain.sh citeseer_small 32,32 1 1e-4 True
