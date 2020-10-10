# Chapter 3: Finite Markov Decision Processes

The problem of finite MDPs involves evaluative feedback, as in bandits, but also an associative aspect &mdash; choosing different actions in different situations. They are a classical formalisation of seqeunential decisions making, where actions influence not just immediate rewards, but also subsequent situations, or states, and through those future rewards.

$$
\begin{aligned}
    F = ma % this is a comment
\end{aligned}
$$



## 3.1 The Agent-Environment Interface

MDPs are meant to be a straightforward framing of the problem of learning from interaction to acheive a goal. 

![](images/fig-3.1.png)

The MDP and agent together give rise to a sequence or _trajectory_ like this:

$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots .
$$

In a _finite_ MDP, the sets of states, actions, and rewards ($\mathcal{S}$, $\mathcal{A}$, and $\mathcal{R}$) all have a finite number of elements. 

Given $s' \in \mathcal{S}$ and $r \in \mathcal{R}$:

$$
p(s', r | s, a) \doteq Pr\{S_t = s', R_t = r | S_{t-1} = s, A_{t-1} = a \},
$$

for all $s' \in \mathcal{S}$, $r \in \mathcal{R}$, and $a \in \mathcal{A(s)}$.

The function $p$ defines the _dynamics_ of the MDP. 

__State-transition probabilities__, $p : \mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$:

$$
\begin{aligned}
p(s' | s, a) & \doteq Pr\{S_t = s' | S_{t-1} = s, A_{t-1} = a \} \\
& = \sum_{r \in \mathcal{R}} p(s', r | s, a).
\end{aligned}
$$

__Expected rewards for state-action pairs__, $r : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$:

$$
\begin{aligned}
r(s, a) & \doteq \mathbb{E}[R_t | S_{t-1} = s, A_{t-1} = a] \\
& = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r | s, a).
\end{aligned}
$$

__Expected rewards for state-action-next-state triples, $r : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$__:

$$
\begin{aligned}
r(s, a, s') & \doteq \mathbb{E}[R_t | S_{t-1} = s, A_{t-1} = a, S_t = s'] \\
& = \sum_{r \in \mathcal{R}} r \frac{p(s', r | s, a)}{p(s' | s, a)}
\end{aligned}
$$

Sensory receptors of an agent should be considered part of the environment rather than part of the agent. Rewards, too, are computed inside the artificial learning system but are considered external to the agent. 

Anything that cannot be changed arbitrarily by the agent is considered to be outside of it and this part of its environment. The agent-environment boundary represents the limit of the agent's _absolute control_, not of its knowledge 

## 3.5 Policies and Value Functions 

__State-value function for policy $\pi$__ (value function of a state $s$ under a policy $\pi$):  

$$
\begin{aligned}
v_\pi(s) & \doteq \mathbb{E}_\pi[G_T | S_t = s] \\
& = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \middle| S_t = s \right], \text{ for all $s \in \mathcal{S}$,}
\end{aligned}
\tag{3.12} 
$$

is the expected return when starting in $s$ and following $\pi$ thereafter. The value of the terminal state, if any, is always zero.

$\mathbb{E}_\pi[\cdot]$: the expected value of a random variable given that the agent follows policy $\pi$

__Action-value function for policy $\pi$__ (value of taking action $a$ in state $s$ under a policy $\pi$):

$$
\begin{aligned}
q_\pi(s, a) & \doteq \mathbb{E}_\pi[G_t | S_t = s, A_t = a] \\
& = \mathbb{E} \left[ \sum_{k=0}^{\infty}\gamma^k R_{t+k+1} \middle| S_t = s, A_t = a \right],
\end{aligned}
\tag{3.13}
$$

is the expected return starting from $s$, taking the action $a$, and thereafter following policy $\pi$.

__Bellman equation for $v_\pi$__:

$$
\begin{aligned}
v_\pi(s) & \doteq \mathbb{E}_\pi \left[ G_T \middle| S_t = s \right] \\
& = \sum_{a} \pi \left( a \middle| s \right) \sum_{s', r} p \left( s', r \middle| s, a \right) \left[ r + \gamma v_\pi(s') \right], \text{ for all $s \in \mathcal{S}$}, 
\tag{3.14}
\end{aligned}
$$

the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way. 

### Example 3.6: Golf

- Reward: -1 for each stroke until we hit the ball into the hole.
- State: location of the ball.
- Value of a state: negative of the number of strokes to the hole from that location.
- Actions: how we aim and swing at the ball and which club we select.

### Exercise 3.17

__Bellman equation for action values, $q_\pi$__:

$$
\begin{aligned}
q_\pi(s, a) & \doteq \mathbb{E}_\pi \left[ G_t \middle| S_t=s, A_t=a \right] \\ 
& = \sum_{s', r} p(s', r | s, a)\left[r + \gamma \sum_{a'}\pi(a'|s')q_\pi(s',a') \right] 
\end{aligned}
$$

### Exercise 3.18

__Value of a state__:

$$
\begin{aligned}
v_\pi(s) & = \mathbb{E}_\pi[q_\pi(s,a)] \\
& = \sum_a \pi(a|s) q_\pi(s,a)
\end{aligned}
$$

depends on the values of the actions possible in that state and how likely each action is to be taken under the current policy.

### Exercise 3.19

__Value of an action (state-action pair)__:

$$
\begin{aligned}
q_\pi(s,a) & = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a] \\
& = \sum_{s', r} p(s', r | s, a)[r + \gamma v_\pi(s')]
\end{aligned}
$$

depends on the expected next reward and the expected sum of the remaining rewards.

## 3.6 Optimal Policies and Optimal Value Functions 

$\pi \geq \pi'\iff v_\pi(s) \geq v_{\pi'}(s)$ $\forall$ $s \in \mathcal{S}$

__Optimal policy__, $\pi_*$: the one or more policy that is better than or equal to all other policies.

__Optimal state-value function__:

$$v_*(s) \doteq \max_\pi v_\pi(s), \quad \forall \: s \in \mathcal{S}, \tag{3.15}$$

is the shared state-value function of the optimal policies.

__Optimal action-value function__:

$$q_*(s,a) \doteq \max_\pi q_\pi(s,a), \quad \forall \: s \in \mathcal{S} \: \text{and} \: a \in \mathcal{A}, \tag{3.16}$$

is also shared by the optimal policies.

For the state-action pair $(s,a)$, $q_*$ gives the expected return for taking action $a$ in state $s$ and thereafter following an optimal policy. Hence, 

$$q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t = s, A_t = a ]. \tag{3.17}$$

__Bellman optimality equation__ (the Bellman equation for $v_*$):

$$
\begin{aligned}
v_*(s) 
& = \max_{a \in \mathcal{A}(s)} q_{\pi_*}(s,a) \\
& = \max_a \mathbb{E}_{\pi_*}[G_t|S_t=s,A_t=a] \\
& = \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t=t, A_t=a] \\
& = \max_a \sum_{s', r} p(s',r|s,a)[r + \gamma v_*(s')]. \tag{3.18}
\end{aligned}
$$

Because $v_*$ is the value function for a policy, it must satisfy the self-consistency condition given by the Bellman equation for state values (3.14). The equation expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state.

__Bellman optimality equation for $q_*$__:

$$\begin{aligned} 
q_*(s,a) & = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} q_*(S_{t+1},a') \middle| S_t=s, A_t=a \right] \\
& = \sum_{s', r}p(s',r|s,a) \left[r + \gamma \max_{a'} q_*(s',a')\right] \tag{3.20}
\end{aligned}$$

![](images/fig-3.4.png)

For finite MDPs, the Bellman optimality equation for $v_*$ (3.19) has a unique solution. The bellman optimality equation is actually a system of equations, one for each state, so if there are $n$ states, then there are $n$ equations in $n$ unknowns. If the dynamics $p$ of the environment are known, then in principle one can solve this system of equations for $v_*$.

Once one has $v_*$, it is relatively easy to determine an optimal policy. Any policy that is _greedy_ with respect to the optimal evlauaton function $v_*$ is an optimal policy. The agent chooses optimal actions by, for any state $s$, finding any action that maximises $q_*(s,a)$.

Explictly solving the Bellman optimality equations provides one route to finding an optimal policy, and thus to solving the reiforcement learning problem. It is akin to an exhaustive search. This solution relies on at least three assumptions that are rarely true in practice: 

1. the dynamics of the environment are accurately known

2. computational resources are sufficient to complete the calculation 

3. the states have the Markov property 

For backgammon, the first and third assumptions present no problems but the third is an impediment. The game has about $10^{20}$ states.

Many different decision-making methods can be viewed as ways of approximately solving the Bellman optimality equation 

## 3.7 Optimality and Approximation 

In practice, an agent rarely learns an optimal policy. Computational costs are too extreme.

Memory is an important constraint. In small tasks, it is possible to approximate with tables with one entry for each state, *tabular* case, with corresponding tabular methods. 

The online nature of reinforcement learning makes it possible to approximate optimal policies in ways that put more effort into learning to make good decisions for frequently encountered states, at the expense of less effort for infrequently encountered states

# 4 Dynamic Programming

The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov decision process (MDP).

We usually assume that the environment is a finite MDP. A common way of obtaining approximate solutions for tasks with continuous state and actions is to quantise the state and action spaces adn then apply finite-state DP methods.

The key idea of DP, and of reinforcement learning generally, is the use of value functions to organise and structure the search for good policies. 

## 4.1 Policy Evaluation (Prediction)

**Policy evaluation/The prediction problem:** how to compute the state-value function $v_\pi$ for an arbitrary policy $\pi$.

If the environment's dynamics are completely known, then (3.14) is a system of $|\mathcal{S}|$ simultaneous linear equations in $|\mathcal{S}|$ unknowns (the $v_\pi(s), s \in \mathcal{S})$. Iterative solution methods are most suitable.

**Iterative policy evaluation:**

1. Initial approximation, $v_0$, is chosen arbitrarily.

2. Each successive approximation is obtained by using the Bellman equation for $v_\pi$ as an update rule.

  $$\begin{aligned}
v_\pi(s) & \doteq \mathbb{E}_\pi \left[ R_{t+1} + \gamma v_k(S_{t+1}) \middle| S_t = s \right] \\
& = \sum_{a} \pi \left( a \middle| s \right) \sum_{s', r} p \left( s', r \middle| s, a \right) \left[ r + \gamma v_\pi(s') \right], \text{ for all $s \in \mathcal{S}$}. 
\tag{4.5}
\end{aligned}$$

**Expected update:** To produce each successive approximation, $v_{k+1}$ from $v_k$, iterative policy evaluation applies the same operation to each state $s$: it replaces the old value of $s$ with a new value obtained from the old values of the successor states of $s$, and the expected immediate rewards, along all the one-step transitions possible under the policy being evaluated. 

To write a sequential computer program to implement iterative policy evaluation as
given by (4.5) you would have to use two arrays, one for the old values, $v_k(s)$, and one for the new values, $v_{k+1}(s)$.

![](images/iter-pol-eval.png)