# NAGS - Neuro-Adaptive Graph Search Chess Engine

NAGS (“Neuro-Adaptive Graph Search”) is a novel chess engine that marries a hypergraph-based board representation with advanced neural architectures and adaptive search strategies. At its core, NAGS encodes each chess position as a hypergraph capturing high-order relationships among pieces, processed by a Graph Neural Network (GNN) augmented with transformer-style multi-head attention to produce policy and value predictions. These predictions then guide a hybrid search algorithm that interleaves classical search methods.

## Features

- Hypergraph + GNN-based board representation  

- Dual-Head Transformer-GNN position evaluator  

- Hybrid search combining α-β pruning and MCTS with UCT selection  

- Meta-learning for position-specific search optimization  

- UCI compatibility for Arena Chess GUI integration  

## System Architecture

### Hypergraph Neural Network

<img align="center" src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7BZ%7D%20%3D%20%5Csigma%5CBigl%28%5Cmathbf%7BD%7D_e%5E%7B-1%7D%5Cmathbf%7BH%7D%5E%5Ctop%20%5Cmathbf%7BX%7D%5Cmathbf%7BW%7D_e%5CBigr%29%2C%5Cquad%20%5Cmathbf%7BX%7D%27%20%3D%20%5Csigma%5CBigl%28%5Cmathbf%7BD%7D_v%5E%7B-1%7D%5Cmathbf%7BH%7D%5Cmathbf%7BZ%7D%5Cmathbf%7BW%7D_v%5CBigr%29.">

### Message Passing Neural Network (MPNN)

<img align="center" src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7Bm%7D_i%5E%7B%28t%2B1%29%7D%20%3D%20%5Csum_%7Bj%5Cin%20%5Cmathcal%7BN%7D%28i%29%7D%20M%5Cbigl%28%5Cmathbf%7Bh%7D_i%5E%7B%28t%29%7D%2C%5Cmathbf%7Bh%7D_j%5E%7B%28t%29%7D%5Cbigr%29%2C%5Cquad%20%5Cmathbf%7Bh%7D_i%5E%7B%28t%2B1%29%7D%20%3D%20U%5Cbigl%28%5Cmathbf%7Bh%7D_i%5E%7B%28t%29%7D%2C%5Cmathbf%7Bm%7D_i%5E%7B%28t%2B1%29%7D%5Cbigr%29.">

### Multi-Head Attention

<img align="center" src="https://render.githubusercontent.com/render/math?math=%5Cmathrm%7BAttention%7D%28Q%2CK%2CV%29%20%3D%20%5Cmathrm%7Bsoftmax%7D%5CBigl%28%5Cfrac%7BQK%5E%5Ctop%7D%7B%5Csqrt%7Bd_k%7D%7D%5CBigr%29V">

### UCT Selection

<img align="center" src="https://render.githubusercontent.com/render/math?math=i%5E%2A%20%3D%20%5Carg%5Cmax_i%5CBigl%28Q_i%20%2B%20c%20%5Csqrt%7B%5Cfrac%7B%5Cln%20N%7D%7Bn_i%7D%7D%5CBigr%29">
