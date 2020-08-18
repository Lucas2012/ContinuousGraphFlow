#!/bin/bash

rtol=0.00001
num_layer=1
#graph_method='ae'
graph_method='conv'
patch_size=8

for data in 'mnist_puzzle'
do
  for puzzle_size in 3
  do
    for lr in 0.001 
    do
      imagesize=$(($puzzle_size*$patch_size))
      bash run_scripts/run_puzzle_trainer_graph_multiscale.sh $data 'experiment/puzzle_models/'$data'/graphflow_graph-multiscale_conv_tol'$rtol'_lr'$lr'/puzzlesize-'$puzzle_size'_graph-method-'$graph_method'_numlayer-'$num_layer'_patchsize-'$patch_size $rtol $lr $puzzle_size $imagesize 32 $graph_method $num_layer $patch_size True True

    done
  done
done
