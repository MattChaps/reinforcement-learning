# 2 Multi-armed Bandits

## 2.1 A k-armed bandit Problem

**k-armed bandit problem:** k levers on a slot machine; each of k actions has an expected reward given that that action is selected. 

**Value of an action:** expected reward given that that action is selected.

$A_t$: action selected on time step $t$.

$R_t$: corresponding reward of $A_t$.

$q_*(a)$: value of an action $a$.

$$q_*(a) \doteq \mathbb{E}[R_t|A_t=a]$$

**Greedy action(s):** action(s) whose estimated value is greatest.

**Exploiting:** selecting one of greedy action(s).

**Exploring:** selecting non-greedy action.

## 2.2 Action-value Methods

**Action-value methods:** methods for estimating the values of actions and for using the estimates to make action selection decisions.

**True value of an action:** mean reward when the action is selected.

**Sample-average method $Q_t(a)$:** average of rewards received.

$$Q_t(a) \doteq \frac{\text{sum of rewards when $a$ taken prior to $t$}}{\text{number of times $a$ taken prior to $t$}}$$

**Greedy method:** $A_t \doteq \argmax_a Q_t(a)$.

**Epsilon-greedy method:** with probability $\epsilon$ select randomly from all actions with equal probability, otherwise behave greedily. 

- as number of steps increases, every action will be sampled an infinite number of times, ensuring all $Q_t(a)$ converge to their respective $q_*(a)$.

## Exercise 2.2: Bandit example

Consider a $k$-armed bandit problem with $k=4$ actions, denoted 1, 2, 3, and 4. 

Consider applying to this problem a bandit algorithm using $\epsilon$-greedy action selection, sample-average action-value estimates, and initial estimates of $Q_1(a)=0$, for all $a$. 

Suppose the initial sequence of actions and rewards is $A_1=1$, $R_1=-1$, $A_2=2$, $R_2=1$, $A_3=2$, $R_3=-2$, $A_4=2$, $R_4=2$, $A_5=3$, $R_5=0$. 

On some of these time steps the $\epsilon$ case may have occurred, causing an action to be selected at random. 

1) On which time steps did this definitely occur? 

2) On which time steps could this possibly have occurred?

### Solution:

Build a table for $Q_t(a)$ for each time step $t$:

|     | a=1 | a=2 | a=3 | a=4 |
|-----|-----|-----|-----|-----|
| t=1 | 0   | 0   | 0   | 0   |
| t=2 | -1  | 0   | 0   | 0   |
| t=3 | -1  | 1   | 0   | 0   |
| t=4 | -1  | -0.5| 0   | 0   |
| t=5 | -1  | 0.33| 0   | 0   |

- $A_1=1$: random or greedy
- $A_2=2$: random or greedy
- $A_3=2$: random or greedy
- $A_4=2$: definitely $\epsilon$
- $A_5=3$: definitely $\epsilon$

## 2.4 Incremental Implementation

Let $R_i$ denote the reward received after the $i$th selection of an action, and let $Q_n$ denote the estimate of its action value after it has been selected $n-1$ times, where

$$
Q_n \doteq \frac{R_1 + … + R_{n-1}}{n-1}.
$$

If we record all rewards then computate whenever the estimated value was needed, memory and computational requirements would grow over time. 

Instead, do

$$
Q_{n+1} = \frac{1}{n} \sum_{i=1}^{n} R_i
= …
= Q_n + \frac{1}{n}[R_n - Q_n]
$$

Requires memory only for $Q_n$ and $n$ and the small computation for each new reward.

General form:

$$
NewEstimate \leftarrow OldEstimate + Stepsize  [ Target - OldEstimate ]
$$

$[Target - OldEstimate]$ is the _error_ in the estimate. 
Denote step-size parameter by $\alpha$ or $\alpha_t(a)$

## 2.5 Tracking a Nonstationary Problem

Averaging methods so far are appropriate for stationary bandit problems — reward probabilities do not change over time. For non stationary problems, makes more sense to give more weight to recent rewards than to long-past rewards. Can do this by using a constant step-size parameter, e.g. $\alpha \in (0, 1]$

$$
Q_{n+1} = Q_n + \alpha[R_n - Q_n]
…
= (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1 - \alpha)^{n - i} R_i.
$$

Results in $Q_{n+1}$ being a weighted average of past rewards and initial estimate $Q_1$ since sum of the weights is $(1 - \alpha)^n + \sum_{i=1}^{n} \alpha (1 - \alpha)^(n - i) = 1$. Weight given to $R_i$ decays exponentially.

## 2.6 Optimistic Initial Values 

Methods so far are dependent on initial action-value estimates, $Q_1(a)$ — _biased_ by their initial estimates. For sample-average methods, bias disappears once all actions have been selected at least once, but for methods with constant $\alpha$, the bias not usually a problem and can be helpful. 

Initial action values can be used to encourage exploration by setting the initial estimate far from the true mean. Can be effective on stationary problems but not well-suited to non stationary problems. Any methods that focuses on initial conditions is unlikely to help with nonstationary problems.

## 2.7 Upper-Confidence-Bound Action Selection

In $\epsilon$-greedy action selection, would be better to select among the non-greedy actions according to their potential for actually being optimal, taking into account both how close their estimates are to being maximal and the uncertainties in those estimates. Can select actions according to 

$$
A_t \doteq \argmax_a \left[ Q_t(a) + c \sqrt{\frac{ln(t)}{N_t(a)}} \right] 
$$

- $N_t(a)$ denotes the number of times that action $a$ has been selected prior to time $t$
- $c$ > 0 controls the degree of exploration 
- If $N_t(a) = 0$ then $a$ is considered to be a maximising action

This _upper confidence bound_ (UCB) action selection is that the square-root term is a measure of the uncertainty or variance in the estimate of $a$’s value. $c$ determines the confidence level on the possible true value of action $a$. Each time $a$ is elected, uncertainty is presumably reduced. 

UCB often performs well on bandits, but is more difficult than $\epsilon$-greedy to extend to the more general reinforcement learning settings.

## 2.8 Gradient Bandit Algorithms

Consider a numerical _preference_ for each action $a$, which we denote $H_t(a)$ in R. The larger the preference, the more often that action is taken, but the preference has no interpretation in terms of reward. Only the relative preference of one action over another is important. 

Let $\pi_t(a)$ be the probability of taking action $a$ at time $t$.

$$
Pr\{A_t = a\} \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}} \doteq \pi_t(a)
$$

Initially all action preferences are the same so that all actions have equal probability of being selected 

Without reward baseline term and gradient bandit algorithm, performance degraded. 

## 2.9 Associative Search (Contextual bandits)

So far considered only non associative tasks. In these tasks the learner either tries to find a single best action when the task is stationary, or tries to track the best action as it changes over time when the task is non-stationary. In a general reinforcement learning task there is more than one situation, and the goal is to learn a policy: a mapping from situations to the actions that are best in those situations. 

An _associative search_ task (contextual bandits) involves both trial-and-error learning to _search_ for the best actions, and _association_ of these actions with the situations in which they are best. Associative search tasks are intermediate between $k$-armed bandit problem and full reinforcement learning problem. Full reinforcement learning problem when each action affects immediate reward and _next situation_.

## 2.10 Summary

UCB methods choose deterministically but achieve exploration by subtly favouring at each step the actions that have so far received fewer samples. 

Gradient bandit algorithms estimate not action values, but action preferences, and favour them ore preferred actions in a graded, probabilistic manner using a soft-max distribution. 