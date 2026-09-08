# Frugal Inference for Control

How much does an autonomous agent need to infer in order to act effectively?

This project studies the performance of **frugal agents**: agents that balance task performance against the 
information encoded in the beliefs guiding their actions in partially observed environments. 
Inference is among the most resource-intensive components of an autonomous system, creating a 
major bottleneck as today’s machines process high-dimensional observations from multiple sensors. 
Frugal agents address this bottleneck by jointly optimizing inference and control. 
Rather than pursuing task-agnostic, Bayes-optimal estimation, they use the task’s costs/rewards to 
determine which state directions require accurate estimation and when control can compensate for estimation errors. 
This task-directed use of information could help autonomous agents operate within realistic computation, memory, and energy constraints.

The work connects probabilistic inference, optimal control, and computational neuroscience. It is motivated by the ability of biological systems to produce robust behavior under tight 
limits on energy, memory, and computation, and by the increasing need for artificial agents that can do the same.

## Core idea

The agent computes a joint inference and control strategy that minimize an infinite-horizon average objective:

$$
J =\limsup_{T\rightarrow\infty}\frac{1}{T}
\mathbb{E}\left[
\sum_{t=0}^{T}
\left(
s_t^\top C_s s_t
+a_t^\top C_a a_t
+C_b I(s_t; b_t)
\right)
\right].
$$

Here, $s_t$ is the system state, $a_t$ is the action, and $I(s_t;b_t)$ measures how much information the agent's belief contains about the hidden state. The coefficient $C_b$ controls the price of information, revealing how inference and control reorganize as computational resources become more limited.

## Questions explored

- How does optimal behavior change as integrating evidence becomes more costly?
- What principles emerge when agents move beyond certainty equivalence and jointly optimize inference and control?

## Methods

Because the general nonlinear problem does not admit a tractable closed-form characterization, we study the local asymptotic 
trade-off among task performance, control effort, and belief quality near task-relevant equilibria. In this regime, the dynamics, 
costs, and uncertainty can be approximated using a linear–quadratic–Gaussian model. This approximation allows us to derive and 
systematically characterize frugal strategies: coupled inference–control policies that determine both how agents integrate new evidence 
into their beliefs and how they use those beliefs to select effective actions. 

## Key results

### Spend when it counts

As the cost of encoding information in the belief increases, inference undergoes a transition from Bayes-optimal estimation 
to a lossy regime. Beyond this transition, the task’s costs and rewards determine which state directions require accurate 
estimation and how the control policy should adapt to compensate for the errors introduced by cheaper, less precise beliefs.

### Adaptability begins when perfection ends

Beyond the transition from Bayes-optimal inference, the solution to the joint inference–control problem is no longer unique. 
Agents can combine imperfect inference with compensatory control in multiple ways, forming a structured family of locally equivalent 
perception–action strategies. This family provides a principled design space for adaptive reconfiguration, echoing the flexibility of 
biological organisms as they adjust their behavior to changes in body configuration, load, fatigue, and available resources.

### Thinking less, moving more

Control makes imperfect beliefs behaviorally useful in two complementary ways. First, it compensates for estimation errors, 
limiting their effects on task performance. Second, it steers the system toward states with lower variability, where maintaining 
effective behavior requires less information. In this way, control does more than accomplish the immediate task: it creates conditions 
under which frugal inference remains viable.

## Experiments

The repository implements the optimizer used to compute frugal strategies and evaluates their ability to regulate partially observed dynamical systems near task-relevant equilibria:

- **Two-dimensional random walk:** provides a transparent setting for visualizing the transition from Bayes-optimal to lossy inference as information becomes more costly.
- **Cart-pole stabilization:** demonstrates how coupled inference and control can stabilize an underactuated, open-loop unstable system despite imperfect state estimates.
- **Planar drone hover:** extends the analysis to a higher-dimensional nonlinear system, revealing families of locally equivalent strategies and testing their robustness away from the target equilibrium.

Each experiment compares the frugal strategies with a standard LQG baseline that combines Bayes-optimal Kalman filtering with linear-quadratic regulation.

## Repository structure

```text
frugal-inference-for-control/
├── notebooks/       # Analyses, simulations, and figures
├── src/             # Models, controllers, and optimization utilities
├── figures/         # Selected results and visualizations
├── requirements.txt
└── README.md
```

## Getting started

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter lab
```

## Scope and limitations

The theoretical analysis focuses on stationary linear strategies and local linear-Gaussian approximations near task-relevant equilibria. 
The resulting solution families characterize local asymptotic behavior; they do not establish global equivalence for strongly nonlinear dynamics or 
arbitrary initial conditions. Simulations away from equilibrium are therefore used to evaluate robustness and transient behavior.

## Author

**Itzel Olivos-Castillo**  
Postdoctoral Researcher, Carnegie Mellon University

