#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --mem=30G
#SBATCH --time=1-11:59             # time (DD-HH:MM)
#SBATCH --output=cedar_logs/log_%A.out        # %A for main jobID, %a for array id.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --overcommit

hostname
whoami


data=$1
save=$2
rtol=$3
lr=$4
puzzle_size=$5
imagesize=$6
batchsize=$7
graph_method=$8
num_layer=$9
patch_size=${10}
variable=${11}
conv=${12}
epoch_idx=${13}
temp=${14}

python src/puzzle_graph/train_graphflow.py --data $data --dims 64,64 --strides 1,1,1 --num_blocks 2 --layer_type concat --rademacher True --graphflow True --ifpuzzle True --num_layers ${num_layer} --rtol $rtol --atol $rtol --if_graph_variable $variable --save $save --batch_size $batchsize --puzzle_size $puzzle_size --imagesize $imagesize --num_epochs 1000000 --lr $lr --patch_size $patch_size --graph_multiscale True --multiscale_method $graph_method --conv $conv --use_semantic_graph True --visualize False --train True
#python src/puzzle_graph/train_graphflow.py --data $data --dims 64,64 --strides 1,1,1 --num_blocks 2 --layer_type concat --rademacher True --graphflow True --ifpuzzle True --num_layers ${num_layer} --rtol $rtol --atol $rtol --if_graph_variable $variable --save $save --batch_size $batchsize --puzzle_size $puzzle_size --imagesize $imagesize --num_epochs 1000000 --lr $lr --patch_size $patch_size --graph_multiscale True --multiscale_method $graph_method --conv $conv --use_semantic_graph True --visualize True --train False --temp $temp --resume $save/'checkpt'$epoch_idx'.pth' --n_samples 5
#uncomment below for 2x2 puzzles
#python src/puzzle_graph/train_graphflow.py --data $data --dims 32,32 --strides 1,1,1 --num_blocks 2 --layer_type concat --rademacher True --graphflow True --ifpuzzle True --num_layers ${num_layer} --rtol $rtol --atol $rtol --if_graph_variable $variable --save $save --batch_size $batchsize --puzzle_size $puzzle_size --imagesize $imagesize --num_epochs 1000000 --lr $lr --patch_size $patch_size --graph_multiscale True --multiscale_method $graph_method --conv $conv --use_semantic_graph True --visualize True --train False --temp $temp --resume $save/'checkpt'$epoch_idx'.pth' --n_samples 30
#python src/puzzle_graph/train_graphflow.py --data $data --dims 64,64,64 --strides 1,2,1,-2 --num_blocks 2 --layer_type concat --rademacher True --graphflow True --ifpuzzle True --num_layers ${num_layer} --rtol $rtol --atol $rtol --if_graph_variable $variable --save $save --batch_size $batchsize --puzzle_size $puzzle_size --imagesize $imagesize --num_epochs 1000000 --lr $lr --patch_size $patch_size --graph_multiscale True --multiscale_method $graph_method --conv $conv


echo "DONE"
