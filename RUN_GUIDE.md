# 🚗 Multi-Environment Decision Making - Quick Start Guide

## ✅ Setup Complete!

All dependencies have been installed and the project is ready to run.

## 🎯 Quick Commands

### 1. Train Agent on All 3 Environments (Recommended)
```bash
/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python run.py --config config.yaml
```

### 2. Train on Single Environment (Faster for Testing)
```bash
# Highway only
/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python run.py --config configurations/ddqn/ddqn_highway.yaml

# Merge only
/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python run.py --config configurations/ddqn/ddqn_merge.yaml

# Roundabout only  
/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python run.py --config configurations/ddqn/ddqn_roundabout.yaml
```

### 3. Try Different Algorithms

**Dueling DQN:**
```bash
/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python run.py --config configurations/dueling/dueling_highway.yaml
```

**Prioritized Experience Replay:**
```bash
/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python run.py --config configurations/prioritized_replay/prioritized_replay_highway.yaml
```

### 4. Evaluate Trained Model
```bash
# After training, evaluate the policy
/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python run.py --config config.yaml -m test -p experiments/ddqn/ddqn.pt

# With video rendering
/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python run.py --config config.yaml -m test -p experiments/ddqn/ddqn.pt --render_video
```

### 5. Monitor Training with TensorBoard
```bash
/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python -m tensorboard.main --logdir experiments/
```
Then open: http://localhost:6006

---

## 📊 What to Expect

### Training Output:
```
Training Policy...
| train          | E: 1 | S: 300 | R: 15.4 | FPS: 42.3 | BR: 0.85 | CLOSS: 0.023
| train          | E: 2 | S: 600 | R: 18.7 | FPS: 45.1 | BR: 0.92 | CLOSS: 0.019
Evaluating agent
| eval           | E: 10 | S: 1500 | R: 28.7
```

### Training Time:
- **Single environment**: ~2-3 hours (40,000 steps)
- **Multi-environment**: ~4-5 hours (60,000 steps)

### Output Location:
```
experiments/
├── <experiment_name>/
│   ├── <experiment_name>.pt          # Best model checkpoint
│   ├── <experiment_name>_last.pt     # Final model
│   ├── train.csv                      # Training metrics
│   ├── eval.csv                       # Evaluation metrics
│   ├── tb/                            # TensorBoard logs
│   └── eval_video/                    # Evaluation videos
```

---

## 🔧 Fixed Issues

1. ✅ Updated `requirements.txt` for Python 3.12+ compatibility
2. ✅ Replaced deprecated `attrdict` with custom implementation
3. ✅ Migrated from old `gym` to `gymnasium`
4. ✅ Fixed video recording with gymnasium's `RecordVideo` wrapper

---

## 🎮 Architecture Summary

**Encoder-Decoder Network:**
- **Shared Encoder**: Learns common features across all environments
- **Separate Decoders**: Environment-specific Q-networks for Highway, Merge, and Roundabout
- **Training**: Randomly samples from all environments to build diverse experience

**DQN Features:**
- Double Q-Learning (reduces overestimation)
- Dueling Networks (optional - separates value & advantage)
- Prioritized Experience Replay (optional - importance sampling)
- Multi-step Returns (n-step TD learning)

---

## 💡 Tips

1. **Start Small**: Train on a single environment first to ensure everything works
2. **Monitor Progress**: Use TensorBoard to visualize learning curves
3. **Adjust Config**: Modify `config.yaml` to change hyperparameters
4. **GPU Acceleration**: CUDA will be used automatically if available

---

## 📚 Configuration Files

All YAML configs are in `configurations/` directory:
- `ddqn/` - Double DQN variants
- `dqn/` - Basic DQN
- `dueling/` - Dueling DQN
- `prioritized_replay/` - With prioritized experience replay

Each config specifies:
- Environments to train on
- Network architecture (encoder/critic dimensions)
- Training hyperparameters
- Evaluation settings
