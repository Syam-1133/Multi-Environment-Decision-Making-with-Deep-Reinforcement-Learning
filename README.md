# Multi-Environment Decision Making with Deep Reinforcement Learning

A PyTorch implementation of Deep Q-Network (DQN) variants trained to drive autonomously across **multiple highway environments simultaneously** — Highway, Merge, and Roundabout — using a shared encoder with environment-specific decision heads.

---

## Demo

| Highway-v0 | Merge-v0 | Roundabout-v0 |
|:-----------:|:--------:|:-------------:|
| ![highway](media/highway.gif) | ![merge](media/merge.gif) | ![roundabout](media/roundabout.gif) |
| Drive fast, stay in lanes | Merge into moving traffic | Navigate a circular junction |

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Environments](#environments)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Results](#results)
- [Key Concepts Explained](#key-concepts-explained)

---

## Overview

The core idea of this project is **multi-environment generalization**: instead of training a separate agent for each road scenario, a single agent is trained across all environments simultaneously. At each episode reset, an environment is chosen at random — forcing the agent to develop a general driving policy rather than overfitting to one road type.

**Why this matters:**
- A single model handles diverse driving scenarios
- The shared encoder learns transferable perceptual features
- Separate Q-heads allow per-environment specialization
- Mirrors real-world driving where conditions constantly change

---

## Architecture

```
Observation (5 vehicles × 5 features = 25 numbers)
        │
        ▼
┌───────────────────────┐
│      ENCODER          │  ← Shared across all environments
│  Linear → LayerNorm   │    Learns general road understanding
│  → ReLU → Linear      │
└───────────┬───────────┘
            │  (256-dim embedding)
            ├─────────────────────────────────┐
            ▼                                 ▼
┌─────────────────────┐         ┌─────────────────────────┐
│  Standard Q-head    │   OR    │   Dueling Q-head         │
│  Linear → Q-values  │         │   Value V(s) + Adv A(s,a)│
│  (one per env)      │         │   (one per env)          │
└─────────────────────┘         └─────────────────────────┘
            │
            ▼
    5 Q-values (one per action)
    → Pick action with highest Q
```

### Two Networks (Double Q-Learning)

```
┌─────────────────┐        ┌──────────────────────┐
│  Critic Network │        │  Target Network       │
│  (learns fast)  │        │  (updates slowly)     │
│  every step     │        │  every 50 steps       │
└─────────────────┘        └──────────────────────┘
         │                            │
         │  Action selection          │  Value estimation
         │  "which action is best?"   │  "how good is that action?"
         └────────────────┬───────────┘
                          ▼
               Reduces overestimation bias
               → More stable learning
```

---

## Environments

All environments use the [highway-env](https://github.com/Farama-Foundation/HighwayEnv) library.

### Observation Space
Each environment returns a **5×5 matrix** (25 numbers total):
```
[presence, x_position, y_position, x_velocity, y_velocity]  ← ego vehicle (row 0)
[presence, x_position, y_position, x_velocity, y_velocity]  ← nearby car 1
[presence, x_position, y_position, x_velocity, y_velocity]  ← nearby car 2
[presence, x_position, y_position, x_velocity, y_velocity]  ← nearby car 3
[presence, x_position, y_position, x_velocity, y_velocity]  ← nearby car 4
```

### Action Space
5 discrete actions:
| Action | Meaning |
|--------|---------|
| 0 | Lane change LEFT |
| 1 | IDLE (maintain) |
| 2 | Lane change RIGHT |
| 3 | FASTER |
| 4 | SLOWER |

### Environment Details

| Environment | Goal | Episode Length | Key Rewards |
|-------------|------|---------------|-------------|
| `highway-v0` | Drive fast, stay right | 40 steps | Speed +0.4, Right lane +0.1, Crash -1 |
| `merge-v0` | Merge safely into traffic | Variable | Speed +0.2, Merge penalty -0.5, Crash -1 |
| `roundabout-v0` | Navigate the roundabout | 11 steps | Speed +0.2, Lane change -0.05, Crash -1 |
| `intersection-v0` | Cross the intersection | 13 steps | Arrive +1, Speed +2, Crash -5 |

---

## Algorithms

Four variants are implemented, each building on the previous:

### 1. DQN (Deep Q-Network)
The baseline. A neural network approximates Q(s,a) — the expected future reward for taking action `a` in state `s`.

```
Loss = (r + γ · max_a Q_target(s', a)  -  Q(s, a))²
         └── actual reward ──┘              └── prediction ──┘
```

### 2. DDQN (Double DQN) ← Default
Fixes DQN's overestimation problem by separating action selection from value estimation:

```
DQN:   uses target network for BOTH selection AND evaluation  → overestimates
DDQN:  critic selects action, target evaluates it             → more accurate
```

### 3. Dueling DQN
Splits the Q-value into two learned components:
```
Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
           ↑               ↑
    How good is        How much better is
    this state?        this action vs average?
```
Better at learning which states are inherently good, regardless of action.

### 4. Prioritized Experience Replay (PER)
Instead of sampling memories uniformly, samples experiences proportional to their TD-error (surprise):
```
High error experience  → sampled more often  (robot was wrong here → learn more)
Low error experience   → sampled less often  (robot already knows this)
```
Uses a **Segment Tree** data structure for O(log N) priority sampling.

### Multi-Step Returns
Instead of looking 1 step ahead, the agent looks **n=10 steps ahead**:
```
1-step:  r₁ + γ·V(s₁)
10-step: r₁ + γr₂ + γ²r₃ + ... + γ⁹r₁₀ + γ¹⁰·V(s₁₀)
```
Better credit assignment — the agent learns the consequences of actions further into the future.

---

## Project Structure

```
Multi-Env-Decision-Making/
│
├── run.py                   # Entry point — train or test
├── train.py                 # Training loop (Trainer class)
├── evaluate.py              # Evaluation loop (Evaluator class)
├── highway.py               # Environment wrapper (HighwayEnv class)
├── video.py                 # Video recording utility
├── logger.py                # Metrics logging (CSV + TensorBoard)
├── utils.py                 # Helper functions (MLP builder, soft update, etc.)
├── config.yaml              # Main config (3 environments, DDQN)
├── run.sh                   # Convenience bash commands
├── requirements.txt         # Python dependencies
│
├── policy/
│   ├── agent.py             # Encoder, Critic, DRQLAgent (core RL logic)
│   ├── replay_buffer.py     # ReplayBuffer + PrioritizedReplayBuffer
│   └── segment_tree.py      # SumSegmentTree + MinSegmentTree for PER
│
├── configurations/          # Pre-built configs for experiments
│   ├── dqn/                 # Standard DQN configs
│   ├── ddqn/                # Double DQN configs
│   ├── dueling/             # Dueling DQN configs
│   ├── prioritized_replay/  # PER configs
│   ├── hidden_units_128_256/# Smaller network ablation
│   └── hidden_units_256_512/# Larger network ablation
│
├── env_configs/             # Per-environment reward tuning
│   ├── highway-v0.yaml
│   ├── merge-v0.yaml
│   ├── roundabout-v0.yaml
│   └── intersection-v0.yaml
│
├── experiments/             # Saved models and logs (auto-created)
└── media/                   # Demo GIFs
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Multi-Env-Decision-Making.git
cd Multi-Env-Decision-Making

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install moviepy  # Required for video recording
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- gymnasium 1.x
- highway-env 1.10+

---

## How to Run

### Training

```bash
# Train on all 3 environments (Highway + Merge + Roundabout)
python run.py --config config.yaml

# Train on a single environment
python run.py --config configurations/ddqn/ddqn_highway.yaml

# Train with a specific algorithm
python run.py --config configurations/dueling/dueling_highway_merge_roundabout.yaml

# Output saved to experiments/<experiment_name>/
```

### Testing a Trained Model

```bash
# Evaluate without video
python run.py --config config.yaml -m test -p experiments/ddqn/ddqn.pt

# Evaluate with video recording
python run.py --config config.yaml -m test -p experiments/ddqn/ddqn.pt --render_video
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir experiments/
# Open http://localhost:6006 in your browser
```

### Using the run.sh helper

```bash
bash run.sh train                                    # train all envs
bash run.sh train-highway                            # train highway only
bash run.sh test experiments/ddqn/ddqn.pt           # test model
bash run.sh test-video experiments/ddqn/ddqn.pt     # test with video
bash run.sh tensorboard                              # launch tensorboard
```

---

## Configuration

The main `config.yaml` controls all hyperparameters:

```yaml
experiment_name: ddqn

# Training
num_train_steps: 60000        # Total environment steps
num_eval_steps: 1000          # Steps per evaluation
eval_frequency: 1500          # Evaluate every N steps

# Memory
replay_buffer_capacity: 45000 # How many experiences to store
batch_size: 32                # Experiences per gradient update
multistep_return: 10          # N-step return lookahead

# Agent
agent:
  learning_rate: 0.0005       # Adam optimizer LR
  discount: 0.8               # Future reward discount (γ)
  double_q: true              # Enable Double Q-learning
  prioritized_replay: false   # Enable Prioritized Experience Replay
  critic_tau: 1.0             # Target network update weight (1.0 = hard copy)
  critic_target_update_frequency: 50  # Update target every N steps

# Network sizes
critic:
  hidden_dim: 256             # Q-network hidden layer size
  dueling: false              # Enable Dueling architecture

encoder:
  hidden_dim: 256             # Encoder hidden layer size

# Environments (comma-separated)
environments: highway-v0, merge-v0, roundabout-v0
```

---

## Reading the Training Output

```
| train | E: 24 | S: 376 | R: 29.3 | FPS: 3.1 | BR: 3.7 | CLOSS: 0.33
```

| Column | Full Name | Meaning |
|--------|-----------|---------|
| `E` | Episode | Number of completed episodes |
| `S` | Step | Total environment steps taken |
| `R` | Reward | Total reward this episode (higher = better) |
| `FPS` | Frames Per Second | Training speed |
| `BR` | Batch Reward | Avg reward in the sampled mini-batch |
| `CLOSS` | Critic Loss | Q-network prediction error (lower = better) |

**Healthy training signs:**
- `R` trending upward over episodes
- `CLOSS` trending downward over time
- Reward stabilizing around 28–32 after ~400 steps

---

## Results

Training a DDQN agent on all 3 environments for 60,000 steps:

| Metric | Early (E:10) | Mid (E:40) | Late (E:60) |
|--------|-------------|-----------|------------|
| Reward | ~6–12 | ~28–31 | ~29–31 |
| Critic Loss | 2.70 | 0.37 | 0.20 |
| Behavior | Random | Mostly safe | Consistent |

Evaluation reward at step 1000: **28.8** (agent generalizes, not memorizing)

---

## Key Concepts Explained

### Why Multi-Environment Training?
Training on a single environment can cause overfitting — the agent memorizes patterns specific to that road. By randomly sampling environments at each episode, the agent is forced to learn transferable driving strategies.

### Why Shared Encoder + Separate Heads?
- **Shared encoder**: Detecting a nearby vehicle is the same skill on all roads
- **Separate heads**: The optimal response to that vehicle differs by context (highway vs roundabout)

### Why Experience Replay?
Neural networks assume training data is independent and identically distributed (i.i.d.). Consecutive driving frames are highly correlated. Storing experiences and sampling randomly breaks this correlation, stabilizing training.

### Why Epsilon-Greedy Exploration?
```
At start:  ε = 0.95 → 95% random actions  (explore the environment)
Over time: ε decays  → more greedy actions  (exploit learned knowledge)
At end:    ε = 0.05 → 5% random actions   (mostly exploit, little explore)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Neural network and gradient computation |
| `gymnasium` | RL environment interface |
| `highway-env` | Driving simulation environments |
| `tensorboard` | Training visualization |
| `numpy` | Array operations |
| `imageio` / `moviepy` | Video recording |
| `PyYAML` | Config file parsing |
| `matplotlib` | Plotting results |

---

## Acknowledgements

- Environments provided by [highway-env](https://github.com/Farama-Foundation/HighwayEnv) (Farama Foundation)
- Architecture inspired by [CURL](https://github.com/MishaLaskin/curl) and DrQ
- Reference: [Multi-Env Decision Making — Shantanu Acharya](https://shantanuacharya.notion.site/Multi-Env-Decision-Making-d40e0ad783e64eebbb755756306e8ed9)
