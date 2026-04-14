# Experiments: Hybrid Offline-Online Follower Manipulation

## Goal

Evaluate whether **offline data reduces the online sample complexity** of learning optimal follower manipulation in Stackelberg bandits.

This directly tests our theoretical claim:

> Offline data reduces the number of rounds where the follower does not play the optimal manipulation (denoted ( T\_{f,w} )).

---

## Experiment 1: Effect of Offline Dataset Size

### Objective

Test how increasing offline data improves learning efficiency.

### Setup

- Environment: Synthetic Stackelberg bandit

- Leader actions: |A| = 5–10

- Follower actions: |B| = 5–10

- Reward functions:
  - ( \mu\_\ell(a,b) \sim \text{Uniform}(0,1) )
  - ( \mu_f(a,b) \sim \text{Uniform}(0,1) )

- Algorithms:
  - **Baseline**: FMUCB (purely online)
  - **Proposed**: Hybrid-FMUCB (offline + online)

### Variable

- Offline dataset size:
  [
  N_{\text{off}} \in {0, 100, 500, 1000, 5000}
  ]

### Metrics

- ( T\_{f,w} ): number of rounds where follower is not playing optimal manipulation
- Convergence time to optimal manipulation
- (Optional) Leader regret

### Expected Outcome

- As ( N\_{\text{off}} ) increases:
  - ( T\_{f,w} \downarrow )

- With sufficient offline data:
  - Immediate identification of optimal manipulation (( T\_{f,w} \approx 0 ))

---

## Experiment 2: Effect of Offline Data Coverage

### Objective

Demonstrate that **quality of offline data matters**, not just quantity.

### Setup

Fix offline dataset size ( N\_{\text{off}} ), vary coverage:

- **Good Coverage**
  - Offline data includes key action pairs relevant to optimal manipulation

- **Poor Coverage**
  - Offline data misses critical action pairs

### Metrics

- ( T\_{f,w} )
- Success rate of identifying correct manipulation

### Expected Outcome

- Good coverage → significant improvement
- Poor coverage → minimal improvement

### Insight

Supports the role of the **transfer coefficient ( C\_{\text{man}} )**:

> Offline data only helps if it covers decision-relevant directions.

---

## Experiment 3 (Optional): Contextual Extension

### Objective

Evaluate performance in contextual Stackelberg bandits.

### Setup

- Add context ( x )
- Define:
  [
  \mu_\ell(x,a,b) = \langle \phi_\ell(x,a,b), \theta_\ell \rangle
  ]
- Compare:
  - Tabular Hybrid-FMUCB
  - Contextual Hybrid-FMUCB

### Metrics

- ( T\_{f,w} )
- Generalization across contexts

### Expected Outcome

- Contextual model learns faster due to shared structure

---

## Visualization

Recommended plots:

1. **Offline Size vs Learning Speed**
   - x-axis: ( N\_{\text{off}} )
   - y-axis: ( T\_{f,w} )

2. **Coverage vs Performance**
   - Bar chart: good vs poor coverage

3. (Optional) Learning curves over time

---

## Key Takeaways

- Offline data reduces exploration requirements
- Benefit depends on **coverage of important actions**
- Hybrid approach interpolates between:
  - Purely online learning
  - Fully offline identification

---

## Implementation Notes

- Use multiple random seeds for stability
- Ensure leader uses EXP3 with exploration parameter ( \alpha )
- Track true optimal manipulation for evaluation
- Keep environment simple (synthetic) for reproducibility

---

## Summary

These experiments validate the core claim:

> Hybrid offline-online learning reduces the sample complexity of follower manipulation —
> **but only when offline data is informative.**
