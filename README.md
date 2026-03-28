# WellSim 🧠

**An Extensible Python Framework for Simulating and Evaluating Wellbeing-Aware Recommender Systems**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/RL-OpenAI%20Gym-orange.svg)](https://gymnasium.farama.org/)

---

## Overview

WellSim is a research-grade simulation framework that enables **safe, ethical evaluation of reinforcement learning recommendation policies** before deployment to real users.

Commercial recommendation systems optimized purely for engagement have been linked to content fatigue, algorithmic burnout, and psychological harm. WellSim solves this by providing:

- A **19-dimensional psychometric state space** grounded in Seligman's PERMA, Ryff's wellbeing model, and Diener's SWB theory
- A **Constrained MDP (CMDP) Safety Shield** that intercepts harmful policy actions at runtime
- A **Bradley-Terry IRL reward engine** for learning wellbeing preferences from offline behavioral logs
- **Multi-domain validation** across KuaiRec 2.0 (video), MovieLens 25M (semantic), and Spotify Podcast (audio)

---

## Architecture

```
wellsim/
├── core/
│   ├── environment.py          # WellBeingEnv — main Gym environment
│   ├── deep_reward_model.py    # Bradley-Terry IRL reward model
│   ├── preference_based_irl.py # Preference-based IRL training
│   ├── irl_methods.py          # Extended IRL utilities
│   ├── maxent_irl.py           # MaxEnt IRL implementation
│   ├── ppo.py                  # PPO agent for wellbeing-aware policies
│   ├── rlhf_ppo.py             # RLHF-PPO integration
│   ├── policy_rollout.py       # Policy evaluation & rollout utilities
│   └── qualitative_examples.py # Illustrative usage examples
├── baselines/
│   ├── reward_shaping_baselines.py  # Heuristic reward baselines
│   └── rlhf_baseline.py             # RLHF comparison baseline
├── evaluation/
│   └── counterfactual.py       # Doubly Robust offline evaluation
├── experiments/
│   ├── comprehensive_comparison_experiment.py  # Full ablation suite
│   └── run_all.py              # Run all experiments
├── demo_usage.py               # Quick-start demo
└── README.md
```

---

## Quick Start

### Installation

```bash
pip install torch numpy gymnasium
git clone https://github.com/kritikGianta/WellSim.git
cd WellSim
```

### Run the Demo

```python
python demo_usage.py
```

### Basic Usage

```python
from wellsim.core.environment import WellBeingEnv
from wellsim.core.deep_reward_model import DeepRewardModel
from wellsim.core.ppo import PPOAgent

# Initialize a CMDP-safe environment
env = WellBeingEnv(
    dataset_path=None,        # plug in KuaiRec / MovieLens path here
    max_steps=50,
    enable_safety_shield=True,  # blocks burnout-inducing recommendations
    fatigue_threshold=0.8
)

# Initialize IRL reward model + PPO agent
reward_model = DeepRewardModel(state_dim=19, action_dim=5)
agent = PPOAgent(state_dim=19, action_dim=5, learning_rate=3e-4)
```

---

## Core Concepts

### 1. Psychometric State Space (19D)
The user state vector `S_t` encodes:
- **8D** — Static user embedding (domain affinities via matrix factorization)
- **5D** — Dynamic interaction history (sliding window)
- **6D** — Live PERMA psychometric trackers (fatigue, engagement, mood drift)

### 2. CMDP Safety Shield
Before executing any agent action `A_t`, the shield computes the expected next-state fatigue level.  
If `Fatigue ≥ 0.8`, the action is overridden with a mandatory `Break` recommendation and a cost penalty is returned — **even if the agent never asked for one**.

### 3. Wellbeing Reward via IRL
Instead of hand-designing a reward function, WellSim learns it from expert behavioral trajectories using the **Bradley-Terry preference model**:

```
P(i > j) = exp(R(i)) / (exp(R(i)) + exp(R(j)))
```

This extracts continuous relative ranking preferences, sidestepping the unreliability of absolute psychometric labels.

### 4. Action Space
Actions are discretized into psychologically-grounded strategy classes:

| Action Class | Grounding |
|---|---|
| High-Arousal | Zillmann's Mood Management Theory [excitation] |
| Break-Inducing | Gross's Emotion Regulation Theory [down-regulation] |
| Diverse-Longform | Cognitive absorption envelope (PERMA: Engagement) |

---

## Empirical Validation

| Domain | Dataset | Pearson r (sim vs. real) |
|---|---|---|
| Video | KuaiRec 2.0 | 0.86 ± 0.04 |
| Semantic | MovieLens 25M | 0.81 ± 0.05 |
| Audio | Spotify Podcast | 0.79 ± 0.06 |

Ablation studies (95% CI, 50 random seeds) confirm both the CMDP shield and IRL module are mission-critical.

---

## Limitations

- The transition model is trained on offline logs — distribution drift over time requires periodic retraining
- IRL-learned rewards may reflect latent demographic biases present in historical interaction data
- The simulator is a sandbox; it does not replace final real-world deployment evaluation

---

## Citation

If you use WellSim in your research, please cite:

```
@misc{wellsim2024,
  title  = {WellSim: An Extensible Framework for Simulating and Evaluating Wellbeing-Aware Recommender Systems},
  author = {Gianta, Kritik},
  year   = {2024},
  url    = {https://github.com/kritikGianta/WellSim}
}
```

---

## License

MIT License. See `LICENSE` for details.
