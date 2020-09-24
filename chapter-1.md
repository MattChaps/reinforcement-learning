# 1 Introduction

Humans learn by interacting with our environment.

Reinforcement learning is focused on goal-directed learning from interaction.

## 1.1 Reinforcement Learning 

RL is learning what to do to maximise a numerical reward signal. The learner discovers which action yields the most reward by trial-and-error. Actions may affect immediate reward and all subsequent rewards (delayed rewards).

The RL problem is the optimal control of incompletely-known Markov Decisions Processes (MDPs).

A challenge in RL is the trade-off between exploration and exploitation.

## 1.3 Elements of Reinforcement Learning 

### Policy 

Defines learning agent's way of behaving at a given time.

### Reward Signal

Defines goal of a RL problem.

Objective of agent is to maximise total reward it receives over the long run. 

### Value function

Specifies what is good in the long run. 

**Value of a state:** total amount of reward an agent can expect to accumulate over the features, starting from that state. 

### (Model)

Mimics behaviour of the environment and allows inferences to be made about how the environment will behave.

Used for planning.

## 1.4 Limitations and Scope 

Policy and value function take state as input and output another state.

## 1.5 An Extended Example: Tic-Tac-Toe

Let $S_t$ denote the state before the greedy move, and $S_{t+1}$ the state after that move. The update to the estimated value of $S_t$, $V(S_t)$, can be written as $$V(S_t) \leftarrow V(S_t) + \alpha \left[ V(S_{t+1}) - V(S_t) \right]$$ where alpha, the *step-size parameter* is a small positive fraction which influences the rate of learning. 

**Temporal-difference** learning: changes are based on a difference between estimates at two successive times. 