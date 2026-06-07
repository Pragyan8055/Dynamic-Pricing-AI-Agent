# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

**Open Project 2026 — Society of Business | IIT Roorkee**

> A self-improving, multi-agent pricing engine that autonomously predicts charging demand, recommends real-time tariffs, and continuously learns from operational outcomes — built on 16,304 ACN sessions and 8,640 × 247-station Shenzhen time-series data.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Datasets](#2-datasets)
3. [Repository Structure](#3-repository-structure)
4. [Feature Engineering](#4-feature-engineering)
5. [Agent 1 — Demand Prediction](#5-agent-1--demand-prediction-agent)
   - [Gradient Boosting Regressor](#51-gradient-boosting-regressor-primary-model)
   - [Random Forest Regressor](#52-random-forest-regressor-benchmark)
   - [ST-GNN — Spatio-Temporal Graph Neural Network](#53-st-gnn--spatio-temporal-graph-neural-network)
   - [Model Comparison & Best Model](#54-model-comparison--best-model)
6. [Agent 2 — Tariff Pricing](#6-agent-2--tariff-pricing-agent)
   - [Rule-Based LP Pricing](#61-rule-based--linear-programming-lp-tariff)
   - [Premium Decomposition Model](#62-premium-decomposition-model)
   - [PPO Reinforcement Learning Agent](#63-ppo-reinforcement-learning-tariff-agent)
7. [Agent 3 — Monitoring & Learning](#7-agent-3--monitoring--learning-agent)
8. [Operations Research Layer](#8-operations-research-layer)
   - [MIP Charger Scheduling](#81-mixed-integer-programme-mip--charger-scheduling)
   - [Vehicle Routing Problem](#82-vehicle-routing-problem-vrp)
   - [DEA Station Efficiency](#83-data-envelopment-analysis-dea)
9. [Network & Graph Analysis](#9-network--graph-analysis)
10. [Key Results](#10-key-results)
11. [Deliverables](#11-deliverables)
12. [Assumptions & Limitations](#12-assumptions--limitations)
13. [How to Run](#13-how-to-run)

---

## 1. Project Overview

Static flat-rate EV tariffs (e.g. a fixed ₹15/kWh or ¥1/kWh) are blind to real-world congestion, grid stress, and demand heterogeneity. This project builds a **three-agent agentic AI framework**:

```
Historical + Real-Time Data
        │
        ▼
┌──────────────────────┐
│  Demand Prediction   │  ──►  ûₛ,ₜ  (utilisation forecast)
│       Agent          │  ──►  P(congested)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   Tariff Pricing     │  ──►  τ*ₛ,ₜ  (optimal per-kWh tariff)
│       Agent          │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Monitoring &        │  ──►  KPI tracking + feedback signal
│  Learning Agent      │       to update pricing policy
└──────────────────────┘
```

The causal chain is:

$$\text{Dynamic Tariff} \xrightarrow{\text{demand response}} \text{Charging Demand} \xrightarrow{\text{network propagation}} \text{Load Distribution}$$

---

## 2. Datasets

| Dataset | Coverage | Shape | Use |
|---------|----------|-------|-----|
| **ACN-Data** (Caltech/JPL) | Apr – Dec 2018 | 16,304 sessions × 27 cols | Session-level EDA, user premium modelling |
| **ST-EVCDP / UrbanEV** (Shenzhen) | Jun – Jul 2022 | 8,640 timesteps × 247 stations | Demand modelling, tariff optimisation, network analysis |

**ST-EVCDP files:**

| File | Description |
|------|-------------|
| `volume.csv` | kWh delivered per 5-min slot per station |
| `occupancy.csv` | Busy charger piles per 5-min slot |
| `price.csv` | Observed price (¥/kWh) per slot |
| `duration.csv` | Average session duration per slot |
| `adj.csv` | Binary station adjacency matrix |
| `distance.csv` | Inter-station distances (km) |
| `information.csv` | Station metadata (capacity, area, CBD flag) |

---

## 3. Repository Structure

```
├── ev_tariff_optimization.ipynb       # Core pipeline: EDA + 3 agents (GB/RF)
├── EV_Tariff_Optimization_2.ipynb     # Extended pipeline: network demand model + premium decomposition
├── OR_Network_EV_Analysis.ipynb       # OR layer: LPP, MIP, VRP, DEA, graph analysis
├── graph_modelling.ipynb              # Network EDA: degree centrality, adjacency
├── EV_dynamic_tariff_rl.py            # ST-GNN demand model + PPO pricing agent
├── EV_dynamic_tariff_rl_2.py          # Extended RL: user premium clustering (GBR)
├── outputs/
│   ├── evaluation_summary.csv
│   ├── demand_prediction_metrics.csv
│   ├── tariff_agent_metrics.csv
│   ├── monitoring_agent_episodes.csv
│   └── or_summary.csv
└── README.md
```

---

## 4. Feature Engineering

### 4.1 Core KPIs

| Feature | Definition | Economic Meaning |
|---------|-----------|-----------------|
| `util_rate` | `busy_piles / total_piles` | Capacity pressure at the station |
| `revenue_per_slot` | `kWh × price` | Realised revenue per 5-min window |
| `queue_proxy` | `max(busy_piles − 0.9 × capacity, 0)` | Overflow / excess demand signal |
| `occupancy_density` | `busy_piles / area` | Spatial congestion intensity |
| `charging_util_rate` (ACN) | `charging_hr / session_hr` | Idle waste at the charger |
| `efficiency_index` | Composite score ∈ [0,1] | Station-level operational efficiency |

### 4.2 Temporal Encoding

Raw hour and day-of-week are mapped to continuous cyclical features to remove discontinuities at midnight and week boundaries:

$$h_{\sin} = \sin\!\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad h_{\cos} = \cos\!\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

$$d_{\sin} = \sin\!\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad d_{\cos} = \cos\!\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$

**Intuition:** A linear encoding treats hour 23 and hour 0 as far apart; the cyclic encoding correctly places them adjacent in feature space, which is critical for a neural network or tree model to learn overnight demand patterns.

### 4.3 Lag and Rolling Features

For each station $s$, three temporal lags are computed:

- **Lag-1** (5 min): captures minute-level autocorrelation
- **Lag-12** (1 hour): captures the last-hour state
- **Lag-288** (24 hours): captures same-slot-yesterday (strong daily seasonality)

Rolling means over 12-step (1-hour) and 288-step (24-hour) windows smooth out noise:

$$\bar{u}_{s,t}^{(k)} = \frac{1}{k}\sum_{i=1}^{k} u_{s,t-i}$$

Rolling standard deviation over 12 steps captures **volatility** — a key input for surge pricing decisions.

### 4.4 Spatial Spillover Feature

One of the most novel features is the **network spillover**, which captures how congestion at neighbouring stations redirects demand to station $s$:

$$\text{Spillover}_{s,t} = \sum_{j \in \mathcal{N}(s)} W_{sj} \cdot u_{j,t-1}$$

The spatial weight $W_{sj}$ decays with distance using a power-law kernel:

$$W_{sj} = \frac{A_{sj}}{(d_{sj} + \epsilon)^{\delta}}, \quad \delta = 1.5$$

where $A_{sj} \in \{0,1\}$ is the adjacency indicator and $d_{sj}$ is the inter-station distance in km. Rows are then normalised so $\sum_j W_{sj} = 1$.

**Intuition:** When a nearby station is full, its would-be users overflow to the next available station. Without this feature, the demand model cannot explain why some stations suddenly spike.

---

## 5. Agent 1 — Demand Prediction Agent

**Target variable:** `util_rate` ∈ [0, 1] — the fraction of charger piles in use at a station in a 5-minute slot.

**Train/test split:** chronological 80/20 split (time-respecting; no future data leakage).

**Feature set (21 inputs):** cyclic time encodings, three-lag utilisation and kWh values, rolling means and volatility, spatial spillover, station metadata (capacity, fast-charger ratio, CBD flag, efficiency index), and observed price.

---

### 5.1 Gradient Boosting Regressor (Primary Model)

**How it works:**

Gradient Boosting builds an ensemble of $M$ shallow decision trees sequentially. Each tree $f_m$ fits the **negative gradient** (pseudo-residuals) of the loss from all previous trees:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot f_m(\mathbf{x})$$

For squared-error loss $L = \frac{1}{2}(y - \hat{y})^2$, the pseudo-residual is simply the ordinary residual:

$$r_{m,i} = -\left[\frac{\partial L}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}} = y_i - F_{m-1}(\mathbf{x}_i)$$

The key hyperparameters and their effects:

| Hyperparameter | Value used | Effect |
|---------------|-----------|--------|
| `n_estimators` | 400 | More trees → lower bias, higher training time |
| `learning_rate` | 0.04 | Smaller → more trees needed, more regularisation |
| `max_depth` | 5 | Controls model complexity; depth-5 ≈ 32-leaf trees |
| `subsample` | 0.75 | Stochastic GB: each tree sees 75% of rows, reduces variance |
| `min_samples_leaf` | 30 | Prevents overfitting on small leaf groups |

**Why it works well here:** The demand signal is dominated by strong, non-linear patterns (peak hours, day-of-week, lag effects). Gradient Boosting captures these interactions directly without manual feature crossing. The shallow trees maintain interpretability via feature importance.

**Results:**

| Metric | Value |
|--------|-------|
| **RMSE** | **0.0612** |
| **MAE** | **0.0431** |
| **R²** | **0.8847** |

---

### 5.2 Random Forest Regressor (Benchmark)

**How it works:**

Random Forest builds $B$ deep decision trees independently, each on a bootstrapped sample of training rows and a random subset of $\sqrt{p}$ features at each split. The prediction is the average:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(\mathbf{x})$$

The variance reduction from averaging $B$ uncorrelated trees is:

$$\text{Var}\left(\frac{1}{B}\sum_b f_b\right) = \frac{\rho \sigma^2 + (1-\rho)\sigma^2/B}{1} \approx \rho \sigma^2 \text{ as } B \to \infty$$

where $\rho$ is the average pairwise correlation between trees. Feature randomisation reduces $\rho$, which is the key insight behind RF.

**Results:**

| Metric | Value |
|--------|-------|
| RMSE | 0.0698 |
| MAE | 0.0503 |
| R² | 0.8621 |

---

### 5.3 ST-GNN — Spatio-Temporal Graph Neural Network

The most advanced model in the pipeline. It jointly encodes the spatial graph structure (which stations are connected and how far apart) and the temporal sequence (12-step lookback = 1 hour of history).

#### Architecture

```
Input: X ∈ ℝ^(B × L × N × F)
       B = batch, L = 12 lookback steps, N = 25 stations, F = 6 features

  For each time step t = 1..L:
    ┌─────────────────────────────────┐
    │   GCN Layer 1  (F → 32)        │  ← Graph convolution with edge weights
    │   LayerNorm → ELU → Dropout    │
    │   GCN Layer 2  (32 → 16)       │  ← Residual connection
    └────────────┬────────────────────┘
                 │  spatial embedding hₜ ∈ ℝ^(N × 16)
                 ▼
  Stack L spatial embeddings → (N, L, 16)
                 │
                 ▼
    ┌────────────────────┐
    │   GRU  (16 → 32)  │  ← Captures temporal dependencies
    └────────┬───────────┘
             │  last hidden state enc ∈ ℝ^(N × 32)
             ▼
    ┌────────────────────────────────┐
    │  MLP: 32 → 8 → 1  (ELU)      │  ← Per-station prediction head
    └────────────────────────────────┘

Output: ŷ ∈ ℝ^(B × N)  (predicted kWh per 5-min slot)
```

#### Graph Convolution Mathematics (GCN)

Each GCN layer aggregates information from neighbouring stations:

$$\mathbf{H}^{(l+1)} = \sigma\!\left(\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

where $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ (self-loops added), $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{A}_{ij}$ is the degree matrix, and $\mathbf{W}^{(l)}$ are learnable weights. Edge weights $e_{ij} = 1 / (d_{ij} + \epsilon)$ (distance-decay) are passed to the convolution to give closer stations more influence.

**Why GCN?** Standard models treat each station independently. GCN propagates information along the physical network edges — a station that sees high demand can "warn" its neighbours via the graph, improving their predictions.

#### GRU Mathematics

The GRU processes the sequence of spatial embeddings:

$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t])  \quad \text{(update gate)}$$

$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])  \quad \text{(reset gate)}$$

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

The gating mechanism allows the GRU to selectively remember long-range patterns (same-hour-yesterday demand) and forget transient spikes.

#### Training Configuration

| Setting | Value |
|---------|-------|
| Optimiser | Adam, lr = 5×10⁻⁴, weight decay = 10⁻⁴ |
| LR schedule | CosineAnnealingLR, T_max = 15 |
| Epochs | 15 |
| Batch size | 64 |
| Gradient clipping | max norm = 1.0 |
| Dropout | 0.1 (GCN layers) |
| Subgraph | 25 randomly-sampled stations |
| Train / Val / Test split | 70% / 15% / 15% |

**Results:**

| Metric | Value |
|--------|-------|
| **RMSE** | **0.0489** |
| **MAE** | **0.0341** |
| **R²** | **0.9203** |

---

### 5.4 Model Comparison & Best Model

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Notes |
|-------|--------|-------|------|-------|
| Random Forest | 0.0698 | 0.0503 | 0.8621 | Strong baseline, fast inference |
| Gradient Boosting | 0.0612 | 0.0431 | 0.8847 | Best tree-based model |
| **ST-GNN** | **0.0489** | **0.0341** | **0.9203** | **Best overall** |

**🏆 Best Model: ST-GNN**

The ST-GNN outperforms both tree-based models because:

1. **Spatial propagation** — it explicitly models how demand flows across connected stations via GCN. The two tree models treat each station × timestep as an i.i.d. sample, relying on the handcrafted `spillover_util` feature to approximate this. The GCN learns it end-to-end.

2. **Temporal memory** — the GRU maintains hidden state across the 12-step window, capturing temporal dynamics that the lag features only partially approximate.

3. **Joint representation** — the shared encoder produces a 32-dimensional station embedding `enc` that the PPO pricing agent directly reuses, closing the demand → pricing loop end-to-end.

The 2.3% R² gain over GB (0.920 vs 0.885) translates to materially better congestion-probability estimates at the tails, which is exactly where pricing decisions matter most.

---

## 6. Agent 2 — Tariff Pricing Agent

### 6.1 Rule-Based / Linear Programming (LP) Tariff

**Mathematical formulation:**

Let $S = \{1,\ldots,n\}$ be the set of $n = 247$ stations. The LP finds the revenue-maximising tariff vector $\boldsymbol{\tau}^*$:

$$\max_{\boldsymbol{\tau}} \quad \sum_{s=1}^{n} \bar{v}_s \cdot \tau_s \cdot \bigl(1 + \varepsilon(\tau_s - \bar{\tau}_s)\bigr)$$

where $\bar{v}_s$ = mean kWh/slot, $\bar{\tau}_s$ = historical mean price, $\varepsilon = -0.30$ = own-price demand elasticity.

After first-order linearisation (treating $\varepsilon \tau_s^2$ as small), the objective becomes linear with sensitivity coefficient $c_s = \bar{v}_s(1 - \varepsilon \bar{\tau}_s)$:

$$\min_{\boldsymbol{\tau}} \quad -\mathbf{c}^\top \boldsymbol{\tau} \qquad \text{s.t.}$$

| Constraint | Expression | Rationale |
|-----------|-----------|-----------|
| Price floor | $\tau_s \geq \tau^{\min} = ¥0.60$ | Consumer protection |
| Price ceiling | $\tau_s \leq \tau^{\max} = ¥2.50$ | Regulatory / social acceptance |
| Congestion surcharge | $\tau_s \geq \tau^{\min} + ¥0.40$ when $\hat{u}_s \geq 0.80$ | Pigouvian congestion tax |
| Off-peak discount | $\tau_s \leq \tau^{\min} + ¥0.10$ when $\hat{u}_s \leq 0.30$ | Demand stimulation |
| Network average | $\frac{1}{n}\sum_s \tau_s \leq ¥1.40$ | Grid-wide price stability |

Solved with the HiGHS LP solver via `scipy.optimize.linprog`.

**LP Result:** Revenue gain of **+12.33%** over the ¥1/kWh flat baseline, with mean optimal tariff ¥1.28/kWh.

---

### 6.2 Premium Decomposition Model

The tariff is structured as a **base price plus additive premiums**, each grounded in economic theory:

$$\tau^*_{s,t} = \tau_0 \Bigl(1 + \pi^{\text{cong}}_{s,t} + \pi^{\text{spill}}_{s,t} + \pi^{\text{eff}}_{s} + \pi^{\text{time}}_{t} + \pi^{\text{fast}}_{s}\Bigr)$$

| Premium | Formula | Economic Basis |
|---------|---------|---------------|
| **Congestion** $\pi^{\text{cong}}$ | $+0.40 \cdot \max\!\left(\frac{\hat{u}-0.8}{0.2}, 0\right)$ | Pigouvian tax: internalises congestion externality on queuing users |
| **Spillover** $\pi^{\text{spill}}$ | $+0.15 \cdot \text{Spillover}_{s,t}$ | Network externality: nearby congestion raises this station's scarcity value |
| **Efficiency** $\pi^{\text{eff}}$ | $+0.10 \cdot (E_s - 0.5)$ | Reward efficient stations with higher tariff capture; penalise inefficient ones |
| **Time-of-day** $\pi^{\text{time}}$ | $+0.20$ (evening peak), $+0.12$ (morning ramp), $-0.25$ (off-peak night/late) | Classic time-of-use (TOU) pricing to smooth demand |
| **Fast-charger** $\pi^{\text{fast}}$ | $+0.12 \cdot f_s$ | Speed premium: faster service commands a higher willingness-to-pay |

**Intuition:** Think of $\tau_0$ as the base electricity cost. Each premium is an economically motivated add-on — the congestion premium is effectively a Pigouvian tax that internalises the external cost one user imposes on others by occupying a charger during a peak period.

**Demand response simulation (elasticity):**

$$\Delta Q_{s,t} = \varepsilon \cdot \frac{\tau^*_{s,t} - \tau_0}{\tau_0} \cdot Q_{s,t}, \quad \varepsilon = -0.30$$

---

### 6.3 PPO Reinforcement Learning Tariff Agent

The PPO agent learns a pricing policy end-to-end, using the ST-GNN encoder's output as its state representation.

#### Environment Design

**State** at time $t$:

$$\mathbf{s}_t = \bigl[\underbrace{\bar{\mathbf{e}}_t}_{\text{GNN encoding, dim 32}},\; \underbrace{\mathbf{u}_t}_{\text{utilisation, dim 25}},\; \underbrace{\mathbf{p}_t}_{\text{current prices, dim 25}},\; \underbrace{h_{\sin},h_{\cos},\mathbb{1}_{\text{peak}}}_{\text{time features}}\bigr] \in \mathbb{R}^{85}$$

**Action:** For each of the 25 stations, choose one of 7 discrete price adjustments:

$$a \in \{-0.15,\ -0.10,\ -0.05,\ 0,\ +0.05,\ +0.10,\ +0.15\} \text{ ¥/kWh}$$

**Reward function:**

$$r_t = \underbrace{0.5 \cdot R_n}_{\text{revenue}} + \underbrace{0.3 \cdot U_b}_{\text{utilisation balance}} - \underbrace{0.2 \cdot C_p}_{\text{congestion penalty}}$$

where:

$$R_n = \frac{\sum_s p_{s,t} v_{s,t} - \text{baseline}}{\text{baseline} + \epsilon}, \quad U_b = -\left|\bar{u}_t - 0.55\right|, \quad C_p = \frac{1}{N}\sum_s \max(u_{s,t} - 0.80, 0)$$

The target utilisation of 0.55 reflects the sweet spot between congestion (>0.80) and idle waste (<0.30).

#### PPO Algorithm Mathematics

PPO is an actor-critic policy gradient method. The **actor** outputs action logits; the **critic** estimates the value function $V^\pi(s)$. Both share a backbone network.

**Policy gradient objective (clipped surrogate):**

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\!\left( r_t(\theta)\, \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}})\, \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio and $\epsilon_{\text{clip}} = 0.2$.

**Why clipping?** Without it, a greedy gradient step might overshoot, catastrophically changing the policy. The clip keeps updates inside a trust region without the expensive second-order computation of TRPO.

**Advantage estimation (GAE):**

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

with $\gamma = 0.99$ (discount factor), $\lambda = 0.95$ (GAE parameter, trading variance vs bias).

**Full loss:**

$$\mathcal{L}(\theta) = -\mathcal{L}^{\text{CLIP}} + 0.5 \cdot \mathcal{L}^{\text{VF}} - 0.01 \cdot \mathcal{H}$$

where $\mathcal{L}^{\text{VF}} = \mathbb{E}_t[(V_\theta(s_t) - \hat{R}_t)^2]$ is the value loss and $\mathcal{H} = \mathbb{E}_t[\mathcal{H}(\pi_\theta(\cdot|s_t))]$ is the entropy bonus (encourages exploration).

**Actor architecture:**

```
Input: s ∈ ℝ^85
  → Linear(85, 96) → ELU → LayerNorm(96)
  → Linear(96, 48) → ELU
  → [Action head] Linear(48, 25×7)  → reshape (25, 7) → Categorical
  → [Value head]  Linear(48, 1)
```

**Training:** 50 episodes × 144 steps per episode, 2 PPO update epochs per episode, mini-batch size 64, Adam lr = 1×10⁻⁴, gradient clipping = 0.5.

---

### 6.4 Per-User Premium Model (ACN)

On the ACN dataset (16,304 sessions from 315 unique users), a **Gradient Boosting Regressor** maps user behavioural features to a personalised price multiplier:

$$\mu_u = 1.0 + 0.35 \cdot f^{\text{peak}}_u + 0.15 \cdot \min\!\left(\frac{\bar{E}_u}{20}, 1\right) - 0.05 \cdot \min\!\left(\frac{\bar{D}_u}{10}, 1\right) + 0.10 \cdot \min\!\left(\frac{\bar{r}_u}{5}, 1\right) - 0.05 \cdot w_u$$

Users are first clustered into 4 segments using K-Means on 8 behavioural features; the GBR then refines the multiplier within each cluster. Output range: $\mu_u \in [0.85, 1.50]$.

---

## 7. Agent 3 — Monitoring & Learning Agent

The Monitoring & Learning Agent evaluates each **episode** (one operating day = 288 five-minute slots) against realised outcomes and tracks whether the pricing feedback loop is improving decisions over time.

### Metric Definitions

#### Average Waiting Time Reduction

Congestion creates queuing. Each 5-minute slot where `util_rate ≥ 0.80` is treated as a "congested slot" — one unit of potential waiting. The reduction per episode is:

$$\Delta W_e = \max\!\left(N^{\text{cong}}_{\text{before}} - N^{\text{cong}}_{\text{after}}, 0\right) \times \frac{5}{60} \text{ hours}$$

where $N^{\text{cong}}$ is the count of station-slots at or above the 80% congestion threshold.

**Result:** Mean wait-time reduction of **~0.38 congestion-hours per day** across the evaluation period, with a positive trend as the pricing policy matures. The 7-day moving average of congested slots consistently sits below the static-pricing baseline.

> **Note:** This is a proxy metric — computed from utilisation changes, not observed queue records, which are not available in the dataset. Causal interpretation is therefore avoided.

#### Customer Response Rate (Demand Elasticity Proxy)

Measures the aggregate shift in energy delivered in response to tariff changes:

$$\text{CRR}_e = \frac{Q^{\text{adj}}_e - Q^{\text{static}}_e}{Q^{\text{static}}_e} \times 100\%$$

where $Q^{\text{adj}} = Q^{\text{static}} \times (1 + \varepsilon \cdot \Delta\tau/\tau_0)$ is the elasticity-adjusted demand, $\varepsilon = -0.30$.

**Result:** Mean customer response rate of **−8.2%** — reflecting that higher peak tariffs reduce demand by ~8% on average, while off-peak discounts draw in a roughly equal volume shift, consistent with the assumed elasticity.

**Interpretation:** The response rate tracks demand-smoothing effectiveness. As the agent learns to make more targeted (smaller but better-timed) adjustments, the magnitude decreases while revenue efficiency improves — a sign the feedback loop is working.

#### Pricing Efficiency Score (Revenue per kWh)

Tracks whether the feedback loop is generating more revenue from each kWh delivered over time:

$$\text{PE}_e = \frac{\text{Revenue}_e^{\text{dynamic}}}{Q_e^{\text{adj}} + \epsilon} \quad \left(\frac{¥}{\text{kWh}}\right)$$

**Result:** Mean pricing efficiency of **¥1.107/kWh** vs ¥0.984/kWh baseline, with an upward trend slope of **+0.0012 ¥/kWh per episode**. A positive trend confirms the feedback loop is gradually improving pricing decisions over the evaluation window.

---

## 8. Operations Research Layer

### 8.1 Mixed-Integer Programme (MIP) — Charger Scheduling

**Problem:** Decide how many charger piles $x_s \in \mathbb{Z}$ to activate at each station during the peak period, subject to grid capacity and operating budget.

$$\max_{\mathbf{x}} \quad \sum_{s=1}^{n} \hat{u}_s \cdot r_s \cdot x_s$$

$$\text{s.t.} \quad 1 \leq x_s \leq K_s, \quad \sum_s r_s x_s \leq P^{\max}, \quad \sum_s c_s x_s \leq B$$

LP relaxation solved with HiGHS, followed by integer rounding. **Result:** 1,654 of 2,168 available piles activated (76.3% activation ratio), consuming 2,999 kWh/slot of a 3,000 kWh grid limit.

### 8.2 Vehicle Routing Problem (VRP)

A fleet of 4 service vehicles is dispatched from the highest-centrality depot to visit the 30 most-congested stations for maintenance/rebalancing. Solved via **greedy nearest-neighbour + 2-opt local search**:

**2-opt improvement criterion:** swap edges $(v_i, v_{i+1})$ and $(v_j, v_{j+1})$ if:
$$d(v_i, v_j) + d(v_{i+1}, v_{j+1}) < d(v_i, v_{i+1}) + d(v_j, v_{j+1})$$

**Result:** NN tour = 18.42 km → 2-opt tour = 15.67 km (**14.9% improvement**).

### 8.3 Data Envelopment Analysis (DEA)

DEA scores each station's efficiency using the **CCR (Charnes–Cooper–Rhodes) model** with constant returns to scale.

**Inputs:** charger capacity $K_s$, grid area $a_s$  
**Outputs:** mean kWh delivered, revenue proxy, reliability $(1 - u^{90}_s)$

**Fractional programme (ratio form):**

$$E_k^* = \max_{\mathbf{u},\mathbf{v}} \; \frac{\mathbf{u}^\top \mathbf{y}_k}{\mathbf{v}^\top \mathbf{x}_k} \quad \text{s.t.} \quad \frac{\mathbf{u}^\top \mathbf{y}_j}{\mathbf{v}^\top \mathbf{x}_j} \leq 1 \;\; \forall j, \quad \mathbf{u},\mathbf{v} \geq 0$$

**Linear equivalent (Charnes–Cooper transformation):** set $t = 1 / \mathbf{v}^\top \mathbf{x}_k$, let $\boldsymbol{\mu} = t\mathbf{u}$, $\boldsymbol{\nu} = t\mathbf{v}$:

$$\max_{\boldsymbol{\mu},\boldsymbol{\nu}} \; \boldsymbol{\mu}^\top \mathbf{y}_k \quad \text{s.t.} \quad \boldsymbol{\nu}^\top \mathbf{x}_k = 1, \quad \boldsymbol{\mu}^\top \mathbf{y}_j \leq \boldsymbol{\nu}^\top \mathbf{x}_j \;\; \forall j, \quad \boldsymbol{\mu},\boldsymbol{\nu} \geq \varepsilon$$

**Result:** Mean DEA efficiency = **0.3982**, 9 stations on the efficient frontier (score ≈ 1). Most inefficiency is driven by excess capacity at low-demand stations.

---

## 9. Network & Graph Analysis

The 247 stations form a graph $G = (V, E, w)$ where edges represent physical adjacency and weights are inter-station distances.

| Graph property | Value |
|---------------|-------|
| Nodes | 247 |
| Edges | 503 |
| Connected components | 4 |
| Largest component | 183 nodes |

### Minimum Spanning Tree (MST)

Kruskal's algorithm finds the minimum-weight connected subgraph:

$$T^* = \arg\min_{T \subseteq E,\; T \text{ spans } V} \sum_{(i,j) \in T} d_{ij}$$

**MST backbone weight: 292.43 km** — the minimum physical cable/road infrastructure needed to connect all stations.

### Centrality Analysis

| Metric | Formula | Use in pricing |
|--------|---------|---------------|
| Degree centrality | $C_D(v) = \deg(v)/(n-1)$ | Stations with many neighbours can absorb more redirected demand |
| Betweenness | $C_B(v) = \sum_{s \neq t} \sigma_{st}(v)/\sigma_{st}$ | High-betweenness stations are infrastructure bottlenecks; surge pricing here has outsized network impact |
| Closeness | $C_C(v) = (n-1)/\sum_u d(v,u)$ | Geographically central stations respond faster to price signals from the network |

**Composite hub score** (weighted 0.3 / 0.4 / 0.3) identifies the top 10 critical stations that should receive priority in both surge pricing and maintenance routing.

---

## 10. Key Results

### Demand Prediction Agent

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | 0.0698 | 0.0503 | 0.8621 |
| Gradient Boosting | 0.0612 | 0.0431 | 0.8847 |
| **ST-GNN** *(best)* | **0.0489** | **0.0341** | **0.9203** |

### Tariff Pricing Agent

| Metric | Value |
|--------|-------|
| Revenue gain vs flat baseline | **+12.33%** (LP) / **up to +18% peak episodes** (PPO) |
| LP mean optimal tariff | ¥1.28/kWh |
| PPO final 10-episode avg revenue gain | +14–18% |
| Charger utilisation — before | 0.487 |
| Charger utilisation — after dynamic pricing | 0.461 (↓ off-peak shifted) |
| Off-peak kWh uplift | **+7.5%** |

### Monitoring & Learning Agent

| Deliverable | Value |
|-------------|-------|
| **Avg Waiting Time Reduction** | ~0.38 congestion-hours avoided per day |
| **Customer Response Rate** | −8.2% mean demand shift per tariff change |
| **Pricing Efficiency Score** | ¥1.107/kWh (vs ¥0.984 baseline), +0.0012 ¥/kWh/episode trend |

### Operations Research

| Model | Key Result |
|-------|-----------|
| LP Tariff | +12.33% revenue, mean τ* = ¥1.28/kWh |
| MIP Scheduling | 1,654 piles activated, 99.97% grid utilisation |
| VRP 2-Opt | 14.9% tour-length reduction vs greedy |
| DEA Efficiency | Mean 0.40; 9/183 stations on frontier |
| MST Backbone | 292 km minimum infrastructure |

---

## 11. Deliverables

All code, notebooks, and output CSVs are in this repository.

| Deliverable | File |
|-------------|------|
| Core pipeline notebook | `ev_tariff_optimization.ipynb` |
| Extended network model | `EV_Tariff_Optimization_2.ipynb` |
| OR & graph analysis | `OR_Network_EV_Analysis.ipynb` |
| Graph EDA | `graph_modelling.ipynb` |
| ST-GNN + PPO agent | `EV_dynamic_tariff_rl.py` |
| User premium model | `EV_dynamic_tariff_rl_2.py` |
| Demand metrics | `demand_prediction_metrics.csv` |
| Tariff metrics | `tariff_agent_metrics.csv` |
| Episode KPI history | `monitoring_agent_episodes.csv` / `kpi_history.csv` |
| OR summary | `or_summary.csv` |
| ST-GNN weights | `stgnn_encoder.pt` |
| PPO policy weights | `ppo_policy.pt` |

---

## 12. Assumptions & Limitations

| # | Assumption | Where used | Impact if wrong |
|---|-----------|-----------|----------------|
| 1 | Own-price demand elasticity = −0.30 | Revenue simulation, demand response, customer response rate | If true elasticity is −0.10, revenue gains are understated; if −0.50, overcorrected peak suppression |
| 2 | No cross-price elasticity | Premium decomposition | Stations in the same neighbourhood may cannibalise each other; network tariff interactions are partially but not fully captured |
| 3 | Congestion-slot-hours = wait-time proxy | Monitoring agent wait metric | Actual queue lengths depend on arrival rates and service times; without queue records, causal claims on wait reduction are not made |
| 4 | ACN = fleet site, not public | Session-type classification | ACN users have lower price sensitivity than public charging; user premium multipliers should not be directly applied to public networks |
| 5 | Distance decay δ = 1.5 | Spatial weight matrix | Estimated a priori; spatial panel regression on a longer panel could improve this |
| 6 | Static grid limit = 3,000 kWh/slot | MIP constraints | Real grid capacity is dynamic and depends on feeder loading |

> All results are **associational**. Dynamic pricing effects are simulated using the stated elasticity assumption and do not arise from a natural experiment or randomised trial.

---

## 13. How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn networkx scipy openpyxl
# For RL notebooks:
pip install torch torch-geometric
```

### Data Paths

Update the `DATA_DIR` variable at the top of each notebook/script to point to your local folder containing the `ACN Data_...` and `UrbanEV_ SZ_districts/` sub-directories.

### Execution Order

```bash
# 1. Core pipeline (EDA + 3 agents)
jupyter notebook ev_tariff_optimization.ipynb

# 2. Extended network demand model
jupyter notebook EV_Tariff_Optimization_2.ipynb

# 3. OR & graph analysis
jupyter notebook OR_Network_EV_Analysis.ipynb

# 4. ST-GNN + PPO
python EV_dynamic_tariff_rl_2.py
```

---

*Open Project 2026 — Society of Business | IIT Roorkee. Submission deadline: 7 June 2026.*
