# Literature Survey

## The problems we want to solve

We want computers to be able to solve problems for us. Commonly, we can tell them how to do things (supervised); or, we can ask them to spot patterns (unsupervised). How well the computer solves the problem is only as good as how well we can solve it, because we tell the computer what to do. What if computers can learn how to solve problems by themselves, without us having to give each step? After all, that is how humans learn &mdash; by trial and error. It could be that the computer learns to solve the problem in a novel and more efficient way that we hadn't thought of. Thus, by observing the computer we ourselves can learn new things. The field of teaching computers to learn for themselves is called Reinforcement Learning.

There are some problems we want to solve that are too complex to program every rule for a computer to follow (such as those with continuous spaces). An example is $\dots$. 

These problems are interesting because they have practical applications, and developing learning algorithms to solve them brings us closer to building an artificial general intelligence.

## What others have done

### What kind of problems have they solved, and can problems be categorised into classes?

A lot of research has been done in this area. People have 'solved' traditional games by developing algorithms that beat world champions at Checkers, Chess, Backgammon, and Go, to name a few. People have also made programs that play old video games perfectly, such as Atari games. The types of environments can be categorised into controlling, solving, and maximising. An example of controlling is in robotics, of solving is in mazes, and of maximising in games.

### What were their approaches, and what worked/didn't work/could be improved?

Early on, solving was done with the following approaches: Dynamic Programming, Monte Carlo Methods, Temporal Difference Learning, etc.

Recently, there has been advances in deep reinforcement learning; algorithms include: deep Q-Learning, policy gradients, etc.

- Have themes as the development of reinforcement learning algorithm over the years?

## In this project

In this project, we want to look at deep reinforcement learning algorithms, implement them, and analyse their performance in those environments provided by OpenAI Gym.