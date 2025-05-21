# NAGS - Neuro-Adaptive Graph Search Chess Engine

NAGS (“Neuro-Adaptive Graph Search”) is a novel chess engine that marries a hypergraph-based board representation with advanced neural architectures and adaptive search strategies. At its core, NAGS encodes each chess position as a hypergraph capturing high-order relationships among pieces, processed by a Graph Neural Network (GNN) augmented with transformer-style multi-head attention to produce policy and value predictions. These predictions then guide a hybrid search algorithm that interleaves classical α-β pruning with Monte Carlo Tree Search (MCTS). Crucially, NAGS employs meta-learning to adapt its search parameters on-the-fly, optimizing exploration–exploitation balance for different types of positions. Training proceeds in two phases—supervised learning from grandmaster games to initialize the evaluator, followed by self-play reinforcement learning to refine both network weights and search heuristics. Together, these components yield a flexible, end-to-end trainable engine with the potential for superhuman play.

## Features

- Hypergraph + GNN-based board representation
- Dual-Head Transformer-GNN position evaluator
- Hybrid search combining α-β and MCTS approaches with UCT selection
- Meta-learning for position-specific search optimization
- UCI compatibility for Arena chess GUI integration

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- C++17 compatible compiler
- CUDA 11.7+ (for GPU acceleration)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Cossy179/NAGS.git
   cd NAGS
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Build the C++ components:

   ```bash
   cd src/core
   mkdir build && cd build
   cmake ..
   make
   ```

## Training the Engine

1. Supervised learning from grandmaster games:

   ```bash
   python scripts/train_supervised.py --pgn_path AJ-CORR-PGN-000.pgn
   ```

2. Self-play reinforcement learning:

   ```bash
   python scripts/train_rl.py --initial_model models/supervised_model.pt
   ```

## Using NAGS with Arena

1. Configure the UCI executable in Arena:
   - Open Arena Chess GUI
   - Select “Engines” → “Install New Engine”
   - Navigate to `bin/nags_v1.exe` and select it
   - Configure hash size, threads, and other parameters as needed

2. Start playing or analyzing with NAGS as your engine.

## Project Structure

- `src/core/`: Core C++ engine components (board representation, search)
- `src/gnn/`: Graph Neural Network implementation
- `src/mcts/`: Monte Carlo Tree Search implementation
- `src/uci/`: Universal Chess Interface protocol support
- `src/meta/`: Meta-learning components for search adaptation
- `src/training/`: Training pipeline implementations
- `scripts/`: Utility scripts for training and evaluation
- `tests/`: Test suite
- `utils/`: Helper utilities

## System Architecture

### Hypergraph-Based Board Representation

Each chess position is represented as a hypergraph **H** = (V, E), where nodes *V* correspond to pieces or squares and hyperedges *E* capture higher-order relationships (e.g., lines of attack, pawn chains). This hypergraph structure preserves multi-piece interactions beyond simple pairwise edges, enabling richer encoding of tactical motifs.

### Neural Evaluator: Dual-Head Transformer-GNN

The evaluator network *f* takes hypergraph **H** and produces:

- A **policy head** π(a│s) over legal moves.
- A **value head** *v*(s) estimating the expected game outcome.

#### Hypergraph Message Passing

Let each node *i* have initial feature vector **x**_i. For each hyperedge *e* connecting nodes *{i₁,…,iₖ}*, a hyperedge convolution aggregates node features and redistributes messages:

\[
\mathbf{m}_e = \sigmaigl(\mathbf{W}_e \cdot 	frac{1}{|e|}\sum_{i\in e}\mathbf{x}_i + \mathbf{b}_eigr),\quad
\mathbf{x}_i' = \sigma\Bigl(\mathbf{W}_n \,igl[\mathbf{x}_i \mathbin\Vert \sum_{e 
i i} \mathbf{m}_e igr] + \mathbf{b}_n\Bigr),
\]

#### Transformer-Style Multi-Head Attention

To capture global interactions, node embeddings undergo multi-head self-attention:

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}igl(	frac{QK^	op}{\sqrt{d_k}}igr)V,
\]

with \(Q, K, V\) linear projections of node embeddings.

#### Dual-Head Output

A shared encoder feeds two separate MLP heads:

- **Policy head:** Softmax over moves, trained with cross-entropy against expert/self-play target distributions.
- **Value head:** Scalar output in [–1, +1], trained with mean-squared error or temporal-difference loss.

## Hybrid Search Algorithm

### Classical α-β Pruning

Minimax search with α-β bounds prunes branches where β ≤ α, reducing the effective branching factor significantly.

### Monte Carlo Tree Search (UCT)

When depth limits or time constraints are reached, MCTS performs multiple simulations. Child *i* is selected by the UCT rule:

\[
i^* = rg\max_i\Bigl(Q_i + c\sqrt{	frac{\ln N}{n_i}}\Bigr),
\]

balancing exploitation and exploration.

### Hybrid α-β & MCTS Integration

NAGS interleaves α-β search with MCTS rollouts at strategic depths, using the neural evaluator to provide leaf values and prior probabilities for moves.

### Meta-Learning for Adaptive Search

A meta-learner adjusts search hyperparameters (e.g., exploration constant *c*, rollout count, depth cutoff) on a per-position basis. Using a MAML-style update, parameters θ are optimized so that a few gradient steps on position-specific losses yield improved search performance.

## Training Methodology

1. **Supervised Pretraining:** Train the dual-head network on grandmaster games to predict expert moves (cross-entropy loss) and outcomes (mean-squared error).
2. **Self-Play Reinforcement Learning:** Generate games through self-play using the current network and search strategy. Update policy and value targets iteratively in an expert iteration loop, akin to AlphaZero.

## Mathematical Formulation

- **Hypergraph Neural Network:**  
  \[
  \mathbf{Z} = \sigmaigl(\mathbf{D}_e^{-1} \mathbf{H}^	op \mathbf{X}\mathbf{W}_eigr),\quad
  \mathbf{X}' = \sigmaigl(\mathbf{D}_v^{-1} \mathbf{H}\mathbf{Z}\mathbf{W}_vigr).
  \]
- **Message Passing Neural Network (MPNN):**  
  \[
  \mathbf{m}_i^{(t+1)} = \sum_{j\in \mathcal{N}(i)} Migl(\mathbf{h}_i^{(t)},\mathbf{h}_j^{(t)}igr),\quad
  \mathbf{h}_i^{(t+1)} = Uigl(\mathbf{h}_i^{(t)},\mathbf{m}_i^{(t+1)}igr).
  \]
- **Multi-Head Attention:**  
  \[
  \mathrm{MultiHead}(Q,K,V) = igl[\mathrm{head}_1\Vert\cdots\Vert\mathrm{head}_higr]W^O,
  \]
  \[
  \mathrm{head}_k = \mathrm{softmax}igl(Q_kK_k^	op/\sqrt{d_k}igr)V_k.
  \]
- **UCT Selection:**  
  \[
  i^* = rg\max_i\Bigl(Q_i + c\sqrt{	frac{\ln N}{n_i}}\Bigr).
  \]
- **Meta-Learning Update (MAML):**  
  \[
  	heta' = 	heta - lpha
abla_	heta \mathcal{L}_s(	heta),\quad
  	heta \leftarrow 	heta - eta
abla_	heta \mathcal{L}_s(	heta').
  \]

## Discussion & Future Work

NAGS demonstrates the power of combining hypergraph representations, transformer-augmented GNNs, hybrid search, and meta-learning into an end-to-end trainable chess engine. Future directions include large-scale benchmarking against top engines, deeper analysis of adaptation dynamics, and generalization to other strategic board games.

## References

1. Cossy179. *NAGS – Neuro-Adaptive Graph Search Chess Engine*. GitHub repository.  
2. Feng, Y., et al. *Hypergraph Neural Networks*. arXiv (2018).  
3. Gilmer, J., et al. *Neural Message Passing for Quantum Chemistry*. ICML (2017).  
4. Kipf, T. & Welling, M. *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR (2017).  
5. Vaswani, A., et al. *Attention Is All You Need*. NIPS (2017).  
6. Silver, D., et al. *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*. arXiv (2017).  
7. Coulom, R. *Efficient Selectivity and Backup Operators in Monte Carlo Tree Search*. CG (2006).  
8. Finn, C., Abbeel, P. & Levine, S. *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*. ICML (2017).

## License

MIT
