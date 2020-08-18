## Continuous Graph Flow
This repository contains the PyTorch implementation of continuous graph flow. For details, refer to the paper hosted on [arXiv](https://arxiv.org/abs/1908.02436).

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
