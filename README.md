# GraphNeuralODE

This is the official implementation for paper: 

[Continuous Graph Flow](https://arxiv.org/pdf/1908.02436.pdf)

[Zhiwei Deng*](http://www.sfu.ca/~zhiweid/), [Megha Nawhal*](http://www.sfu.ca/~mnawhal/), [Lili Meng](https://lilimeng1103.wixsite.com/research-site) and [Greg Mori](http://www2.cs.sfu.ca/~mori/)

Published in ICML 2020 Workshop on Graph Representation Learning and Beyond

[Webpage](http://www.sfu.ca/~mnawhal/projects/cgf.html)

If you find this code helpful in your research, please cite

```
@inproceedings{deng2020continuous,
  title={Continuous graph flow},
  author={Deng, Zhiwei and Nawhal, Megha and Meng, Lili and Mori, Greg},
  booktitle={Proceedings of the International Conference on Machine Learning Workshop on Graph Representation Learning and Beyond},
  year={2020}
}
```

## Contents
1. [Overview](#overview)
2. [Getting started](#setup)
3. [Sample Usage](#usage)
4. [Results](#results)

## Overview

We propose a flow-based generative model for graph-structured data, termed Continuous Graph Flow (CGF). CGF is formulated as a system of ordinary differential equations (reversible), uses **Continuous Message Passing** to transform node states over time (continuous). Highlights for the model: extending flow models to handle variable input dimensions; ability to model reusable dependencies in among dimensions; reversible and memory-efficient.

<div align='center'>
  <img src='model_fig.png' width='512px'>
</div>

## Getting started
- Clone repository
- Use python 3 and create a virtual environment 
- Install dependencies listed in `requirements.txt`

## Sample Usage
The repository contains code for three applications: image puzzle generation, scene layout generation and graph generation. We provide sample commands here. Please change the parameters according to the experiments.

- For image puzzle generation, run the sample command. Change the parameters for exploring the code.

<pre><code>
python src/puzzle_graph/train_graphflow.py --data /data --dims 64,64 --strides 1,1,1 --num_blocks 2 
--layer_type concat --rademacher True --graphflow True --ifpuzzle True --num_layers 1 --rtol 1e-5 
--atol 1e-5 --save ./output --batch_size 64 --puzzle_size 2 --imagesize 64 --num_epochs 1000000 
--lr 0.00001 --patch_size 16 --graph_multiscale True --multiscale_method conv --conv True
</pre></code>

- For graph generation, run the following command.
<pre><code>
python src/graph_generation/train_graphflow.py  --graph_type community_small --dims 32 --num_blocks 2 
--result_dir ./output --save ./output --lr 1e-5 --use_logit True
</pre></code>

## Contact
For further questions, please contact the authors [Zhiwei Deng](https://www.sfu.ca/~zhiweid/) or [Megha Nawhal](https://www.sfu.ca/~mnawhal/).
