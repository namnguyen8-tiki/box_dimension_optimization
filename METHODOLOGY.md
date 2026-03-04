# Box Dimension Optimization — Methodology

## 1. Problem Statement

Given a warehouse operation that ships orders of varying sizes, we need to determine the optimal set of carton box dimensions (a "Collection") that minimizes packaging cost while maximizing box utilization across all orders.

The core question: **Is the current collection of box sizes optimal, or should we adjust individual box dimensions to reduce cost and improve space utilization?**

### Definitions

- **Order**: A customer order containing one or more items, each with known dimensions (length, width, height) and quantity.
- **Collection**: A fixed set of N box types, each defined by (length, width, height). This is what we optimize.
- **Box Packing Solution (algorithm_BPS)**: A pre-existing algorithm that, given an order and a collection, determines the best-fitting box and the arrangement of items inside it.
- **Region**: A group of orders sharing similar characteristics, defined by the tuple `(volume_bin, length_bin, unit_bin)` — the same quantile bins used in stratified sampling. Enables per-region diagnostics and planned region-based GA enhancements.

## 2. Architecture Overview

The system is structured as a nested optimization:

1. **Inner layer** — `algorithm_BPS`: Solves "given this order and these boxes, which box fits best and how should items be placed?"
2. **Outer layer** — Genetic Algorithm: Searches for the best Collection (set of box dimensions) by evaluating many candidate Collections using historical order data.

The evaluation of each candidate Collection requires running `algorithm_BPS` on all (or a sample of) historical orders, then computing objective function values.

### Key Classes

- **Collection**: Represents a candidate set of N box types with dimensions, volumes, and surface-area-based costs.
- **Bubble**: Computes bubble wrap cost for filling unused box volume.
- **Valuation**: Evaluates a Collection against order data — runs algorithm_BPS in parallel, computes all three objective functions globally and per-region.
- **Sample**: Creates a stratified proportional sample from the full dataset and validates it via KS test and valuation comparison.
- **GeneticAlgorithm**: Evolves Collections over generations using tournament selection, single-point crossover, Gaussian mutation, and elitism.

## 3. Objective Functions

We define three objective functions. f1 and f2 are computed only over **fittable** orders (clean separation). f3 captures the unfittable ratio separately.

### 3.1 Objective Function 1 — Cost Minimization

For each fittable order i assigned to a box:

$$f_1 = \frac{1}{N_{fittable}} \sum_{i=1}^{N_{fittable}} \left( C_{box,i} + C_{bubble,i} \right)$$

Where:
- $C_{box,i}$ = cost of the chosen box, proportional to its surface area: $2(lw + lh + wh) \times \text{unit\_cost}$
- $C_{bubble,i}$ = cost of bubble wrap filling the unused space: $\frac{61000}{35 \times 90 \times 100 \times t} \times V_{bubble,i}$
- $V_{bubble,i} = (V_{box,i} - V_{items,i}) \times \text{filling\_rate}$
- $t$ = bubble thickness
- $N_{fittable}$ = number of orders successfully assigned to a box

### 3.2 Objective Function 2 — Utilization Optimization

$$f_2 = \frac{1}{N_{fittable}} \sum_{i=1}^{N_{fittable}} \left( \frac{V_{items,i}}{V_{box,i}} - u^* \right)^2$$

Where:
- $V_{items,i}$ = total volume of items in order i (after adding reinforcement thickness)
- $V_{box,i}$ = volume of the chosen box
- $u^*$ = target utilization ratio (default: 0.9)

This is the Mean Squared Error (MSE) of utilization relative to the target. Averaged over fittable orders only, making it scale-independent.

### 3.3 Objective Function 3 — Unfittable Ratio

$$f_3 = \frac{N_{unfittable}}{N_{total}}$$

Where:
- $N_{unfittable}$ = number of orders that could not be packed (detected via algorithm_BPS markers: `"No satisfied Box"` or `"Cannot find any satisfied Box in the given time"`)
- $N_{total}$ = total number of orders evaluated

**Design rationale**: Unfittable orders are structurally expected — some orders have too many items (15–20) or volumes exceeding all boxes. These truly-unfittable orders represent a **natural constant** regardless of Collection. The remaining unfittable orders indicate Collection sparsity (the algorithm can't find a fit in time for that box configuration). A good Collection should bring $f_3$ close to this natural baseline.

### 3.4 Combined Objective — Weighted Sum with Exponential Penalty

$$F = w_1 \cdot \frac{f_1}{b_1} + w_2 \cdot \frac{f_2}{b_2} + w_3 \cdot g(f_3)$$

Where:
- $b_1$, $b_2$ = baseline values for normalization (from current/reference Collection)
- $w_1 = 0.6$, $w_2 = 0.35$, $w_3 = 0.05$ (sum = 1.0, prioritizing cost)

**Exponential penalty function for f3:**

$$g(f_3) = e^{k \cdot (f_3/b_3 - 1)} - 1$$

Properties:
- $f_3 = b_3 \Rightarrow g = 0$ (neutral — no penalty, no reward)
- $f_3 \gg b_3 \Rightarrow$ exponential growth (harsh penalty for Collections producing many more unfittable orders)
- $f_3 \ll b_3 \Rightarrow g \to -1$ (reward, bounded since $f_3 \geq 0$)
- Near $b_3$: approximately linear $g \approx k \cdot (f_3/b_3 - 1)$ (mild)
- $k$ controls sensitivity (default: 3.0)

**Why exponential instead of linear?** The natural constant of truly-unfittable orders means $b_3$ represents a baseline. Collections near $b_3$ are normal — only Collections significantly worse need harsh punishment. The exponential's asymmetric shape (unbounded penalty upward, bounded reward downward) matches this physical reality.

### 3.5 Per-Region Metrics

All three objectives are also computed per region `(volume_bin, length_bin, unit_bin)`:
- Each region's f1/f2 is averaged by the number of fittable orders **in that region**
- Each region's f3 is the unfittable ratio **within that region**
- Per-region baselines (optional) enable region-specific exponential penalties in `results_by_region()`

This supports future region-based GA crossover (Section 6.3) and diagnostic analysis.

## 4. Data Pipeline

### 4.1 Input Data

- **Historical order data**: DataFrame with columns `[order_code, item, unit, length, width, height]`. One order may span multiple rows (one per item type).
- **Collection**: A set of box dimensions `[[l1, w1, h1], [l2, w2, h2], ...]`.

### 4.2 Reinforcement Adjustment

Before packing evaluation, item dimensions are increased to account for protective wrapping:

$$l' = l + 2 \times t_{reinforcement}$$
$$w' = w + 2 \times t_{reinforcement}$$
$$h' = h + 2 \times t_{reinforcement}$$

Default reinforcement thickness: 0.5 cm per side.

### 4.3 Parallel Processing

To handle large datasets, the Valuation process uses multiprocessing:

1. Orders are distributed across buckets using round-robin assignment (ensures equal distribution with at most 1-order variance per bucket).
2. All rows of the same order are guaranteed to stay in the same bucket.
3. Each bucket is processed in a separate OS process (`multiprocessing.Process`).
4. Within each bucket, individual orders are processed in threads calling `box_packing_solution`.

### 4.4 Result Parsing

algorithm_BPS writes CSV with columns: `['order', 'box_final', 'item', 'position', 'process_time']`.

For fittable orders, `box_final` contains the box name (e.g., `"Box1"`). For unfittable orders:
- `"No satisfied Box"` — volume exceeds all boxes or no fit found
- `"Cannot find any satisfied Box in the given time"` — timeout (30s default)

## 5. Representative Sampling

### 5.1 Motivation

Each GA fitness evaluation requires running `algorithm_BPS` on all historical orders. With population_size=50, generations=100, and tens of thousands of orders, this is computationally prohibitive. A representative sample reduces evaluation cost while maintaining statistical validity.

### 5.2 Stratified Proportional Sampling

We create a sample that preserves the distribution of order characteristics that influence box selection.

**Stratification features** (computed per order):
- **Total volume**: Sum of (length × width × height × unit) across all items in the order. Determines minimum box size.
- **Max single dimension**: The largest dimension among all items. Forces minimum box length regardless of volume.
- **Total units**: Number of items. Multi-item orders are harder to pack and may require different box choices.

**Procedure**:
1. Compute per-order summary statistics (total_volume, max_length, total_unit).
2. Create quantile-based bins for each feature using `pd.qcut` with `duplicates='drop'`.
3. Define strata as unique combinations of all bins (up to $5^3 = 125$ strata with 5 bins per feature).
4. Sample proportionally from each stratum with minimum 1 order per stratum.
5. All rows belonging to a sampled order are included (orders are never split).
6. Merge bin columns (`volume_bin`, `length_bin`, `unit_bin`) into both sample and full data for Valuation's region-based tracking.

### 5.3 Sample Validation

Two complementary tests validate the sample:

**Test 1 — Kolmogorov-Smirnov (KS) Test**

Checks whether the sample and full data have the same distribution for each stratification feature.

- Metric: KS statistic (maximum distance between CDFs) and p-value.
- Pass criteria: KS statistic < 0.1 AND p-value > 0.05 for all features.
- Note: For large datasets, prioritize KS statistic over p-value, as the test becomes overly sensitive to small, practically meaningless differences.

**Test 2 — Valuation Comparison**

Checks whether the sample produces similar combined fitness as the full dataset.

- Run Valuation on both full data and sample using the same Collection.
- Also compares unfittable ratios between sample and full.
- Pass criteria: Combined F values differ by less than 10%.

### 5.4 Troubleshooting Poor Samples

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| KS test fails for one feature | Bins too coarse due to non-unique values | Reduce bins for that feature or use manual boundaries |
| KS passes but Valuation fails | Missing stratification feature | Add features (e.g., aspect ratio, min dimension) |
| Both fail | Sample too small | Increase sample_size |

## 6. Genetic Algorithm

### 6.1 Encoding

Each individual (chromosome) represents a candidate Collection: a list of N box dimensions `[[l1, w1, h1], ..., [lN, wN, hN]]`.

Constraints maintained throughout:
- Within each box: dimensions sorted descending (L ≥ W ≥ H)
- Boxes sorted by volume ascending (consistent ordering eliminates permutation redundancy)
- All dimensions clamped to `[dim_min, dim_max]` (default: 5–60)

### 6.2 Fitness Function

Fitness = combined objective function $F$ from Section 3.4, evaluated on the representative sample. **Lower is better.**

### 6.3 Operators

**Initialization**: Random box dimensions — each dimension drawn uniformly from `[dim_min, dim_max]` as integers.

**Selection**: Tournament selection with configurable tournament size (default: 3). The individual with the lowest fitness in the tournament wins.

**Crossover**: Single-point crossover at box level. A random cut point splits the parent Collections; boxes before the cut come from parent 1, boxes after from parent 2 (and vice versa for the second child). Applied with probability `crossover_rate` (default: 0.8). Children are re-sorted by volume.

**Mutation**: Per-dimension Gaussian perturbation. Each dimension of each box is independently mutated with probability `mutation_rate` (default: 0.1). Perturbation drawn from $\mathcal{N}(0, \sigma^2)$ where $\sigma$ = `mutation_sigma` (default: 3.0). Result is clamped to bounds and re-sorted.

**Elitism**: Top `elitism_count` (default: 2) individuals are carried forward unchanged to the next generation.

### 6.4 GA Loop

```
1. Initialize population of random Collections
2. Evaluate all individuals (create Valuation per Collection)
3. For each generation:
   a. Carry forward elite individuals
   b. Fill remaining population via:
      - Tournament selection of 2 parents
      - Crossover → 2 children
      - Mutation on each child
   c. Evaluate all new individuals
   d. Track best individual and full fitness log
4. Return best Collection, best fitness, per-generation best history, and full fitness log
```

### 6.5 History & Diagnostics

The GA records two history structures:
- **history**: List of best fitness per generation (length = generations + 1). For plotting convergence curves.
- **fitness_log**: List of lists — all population fitnesses per generation. For scatter-plotting fitness distributions and analyzing diversity over time.

### 6.6 Region-Based Crossover (Planned)

A domain-informed enhancement to accelerate convergence:

1. After each generation, compute per-region fitness using `results_by_region()`.
2. Identify which boxes primarily serve which regions (via box assignment tracking).
3. For well-performing regions, preserve the responsible boxes (reduced mutation).
4. Focus mutation and crossover on boxes serving poorly-performing regions.

This exploits the problem structure: small boxes serve small orders, large boxes serve large orders. If small-order performance is already good, there is no need to disrupt those box dimensions while improving large-order performance.

## 7. Dependencies

- **algorithm_BPS**: External module (Box Packing Solution) providing the core packing algorithm. Imported via `sys.path` configuration pointing to the sibling directory `../Box Packing Solution/`.
- **Python libraries**: pandas, numpy, scipy (KS test), copy, multiprocessing, threading, math, random, os.

## 8. Parameters Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| number_of_boxes | (input) | Number of box types in a Collection |
| utilization_optimal | 0.9 | Target box utilization ratio |
| reinforcement_thickness | 0.5 cm | Protective wrapping thickness per side |
| bubble_thickness | 1.0 cm | Bubble wrap material thickness |
| bubble_filling_rate | 0.5 | Fraction of empty space filled with bubble wrap |
| objective_function_1_baseline | 1,000,000 | Cost normalization baseline |
| objective_function_2_baseline | 0.1 | Utilization deviation normalization baseline |
| objective_function_3_baseline | 0.05 | Natural unfittable ratio baseline |
| w1, w2, w3 | 0.6, 0.35, 0.05 | Objective function weights (sum = 1.0) |
| k | 3.0 | Exponential penalty sensitivity for f3 |
| sample_size | 10,000 | Number of orders in representative sample |
| number_of_buckets_per_aspect | 5 | Number of quantile bins per stratification feature |
| population_size | 50 | GA population size |
| generations | 100 | GA number of generations |
| mutation_rate | 0.1 | GA per-dimension mutation probability |
| crossover_rate | 0.8 | GA crossover probability |
| tournament_size | 3 | GA tournament selection size |
| elitism_count | 2 | GA number of elite individuals carried forward |
| dim_min | 5 | Minimum box dimension (cm) |
| dim_max | 60 | Maximum box dimension (cm) |
| mutation_sigma | 3.0 | Gaussian mutation standard deviation (cm) |
