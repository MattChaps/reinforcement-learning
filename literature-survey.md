# Reinforcement Learning: A Literature Survey

## Introduction

Reinforcement learning is the class of problems concerned with an agent learning behaviour through trial-and-error interactions with a dynamic environment [1]. 

An example is an aspiring tightrope artist (the agent) learns to walk from one end of a rope to another without falling (the behaviour) by repeatedly correcting their balance (the trial-and-error interactions) whenever the rope wobbles beneath them (the dynamic environment).

In this literature survey we:
1. describe reinforcement learning problems, and explain why they are interesting to solve; 
2. describe historical and current work in the field, including the kinds of problems solved and the approaches; as well as,
3. state what we will be using in our own project.

## Motivation

Reinforcement learning is useful in solving problems where there is a measure of an optimal solution. 

A problem might be, "How do I play to have the greatest chance of winning a game of Chess?" In which case, the optimal solution is an optimal strategy which tells you the best move to make each turn. (Mention Deep Mind and AlphaZero)

Beyond games, reinforcement learning problems exist in practical areas, such as in control tasks in robotics [3] and scheduling problems in memory management [4]. 

We are interested in solving these problems because in doing so we develop systems that are more efficient and lead to improvements in quality of life. 

## Historical work

Reinforcement Learning has been successfully applied to various problems, including [ ]. The following sections detail each problem, and the method used to solve them.

### Markov Decision Processes

To abstract a real problem as a reinforcement learning problem, the environment must also have a notion of state, which the agent can sense and take actions accordingly to achieve a goal relating to the state of its environment [2].

### The Gambler's Problem (Dynamic Programming)

### Blackjack (Monte Carlo Methods)

### Backgammon (Temporal-Difference Learning)

### (Policy Gradient Methods)

## Current work

### (Q-Learning)

Q-learning is a value-based class of algorithms that aim to build a value function, which subsequently lets us define a policy.

$Q$-learning keeps a lookup table of values $Q(s,a)$ with one entry for every state-action pair. In order to learn the optimal $Q$-value function, the $Q$-learning algorithm makes use of the Bellman equation for the $Q$-value function (Bellman and Dreyfus,1962) whose unique solution is $Q^*(s,a)$.

This is often inapplicable with a high-dimensional state-action space.

### Atari 2600 (Deep Q-networks)

The DQN algorithm obtains strong performance in an online setting for a variety of ATARI games, directly by ;earning from the pixels.

Rewards are clipped between $-1$ and $+1$.

## Our project

> Pull it all together, and say what we are taking to use in the project

In this project, we want to look at deep reinforcement learning algorithms, implement them, and analyse their performance in those environments provided by OpenAI Gym.

## References

[1] [Reinforcement Learning: A Survey](https://www.jair.org/index.php/jair/article/view/10166)

[2] Reinforcement Learning: An Introdution, 1.1

[3] [Reinforcement Learning in Robotics: A Survey](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Kober_IJRR_2013.pdf)

[4] [Self-Optimizing Memory Controllers: A Reinforcement Learning Approach](https://dl.acm.org/doi/10.1109/ISCA.2008.21)

Watkins, C. J. C. H. 1989. “Learning from delayed rewards”.PhD thesis.King’s College, Cambridge.

https://towardsdatascience.com/applications-of-reinforcement-learning-in-real-world-1a94955bcd12