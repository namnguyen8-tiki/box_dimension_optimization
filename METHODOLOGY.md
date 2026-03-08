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

$$f_1 = \frac{1}{N_{\text{fittable}}} \sum_{i=1}^{N_{\text{fittable}}} \left( C_{\text{box},i} + C_{\text{bubble},i} \right)$$

Where:
- $C\_{box,i}$ = cost of the chosen box, proportional to its surface area: $2(lw + lh + wh) \times \text{unit\_cost}$
- $C\_{bubble,i}$ = cost of bubble wrap filling the unused space: $\frac{61000}{35 \times 90 \times 100 \times t} \times V\_{bubble,i}$
- $V\_{bubble,i} = (V\_{box,i} - V\_{items,i}) \times \text{filling\_rate}$
- $t$ = bubble thickness
- $N\_{fittable}$ = number of orders successfully assigned to a box

### 3.2 Objective Function 2 — Utilization Optimization

$$f_2 = \frac{1}{N_{\text{fittable}}} \sum_{i=1}^{N_{\text{fittable}}} \left( \frac{V_{\text{items},i}}{V_{\text{box},i}} - u^* \right)^2$$

Where:
- $V\_{items,i}$ = total volume of items in order i (after adding reinforcement thickness)
- $V\_{box,i}$ = volume of the chosen box
- $u^*$ = target utilization ratio (default: 0.9)

This is the Mean Squared Error (MSE) of utilization relative to the target. Averaged over fittable orders only, making it scale-independent.

### 3.3 Objective Function 3 — Unfittable Ratio

$$f_3 = \frac{N_{\text{unfittable}}}{N_{\text{total}}}$$

Where:
- $N\_{unfittable}$ = number of orders that could not be packed (detected via algorithm_BPS markers: `"No satisfied Box"` or `"Cannot find any satisfied Box in the given time"`)
- $N\_{total}$ = total number of orders evaluated

**Design rationale**: Unfittable orders are structurally expected — some orders have too many items (15–20) or volumes exceeding all boxes. These truly-unfittable orders represent a **natural constant** regardless of Collection. The remaining unfittable orders indicate Collection sparsity (the algorithm can't find a fit in time for that box configuration). A good Collection should bring $f\_3$ close to this natural baseline.

### 3.4 Combined Objective — Weighted Sum with Exponential Penalty

$$F = w_1 \cdot \frac{f_1}{b_1} + w_2 \cdot \frac{f_2}{b_2} + w_3 \cdot g(f_3)$$

Where:
- $b\_1$, $b\_2$ = baseline values for normalization (from current/reference Collection)
- $w\_1 = 0.6$, $w\_2 = 0.35$, $w\_3 = 0.05$ (sum = 1.0, prioritizing cost)

**Exponential penalty function for f3:**

$$g(f_3) = e^{k \cdot (f_3/b_3 - 1)} - 1$$

Properties:
- $f\_3 = b\_3 \Rightarrow g = 0$ (neutral — no penalty, no reward)
- $f\_3 \gg b\_3 \Rightarrow$ exponential growth (harsh penalty for Collections producing many more unfittable orders)
- $f\_3 \ll b\_3 \Rightarrow g \to -1$ (reward, bounded since $f\_3 \geq 0$)
- Near $b\_3$: approximately linear $g \approx k \cdot (f\_3/b\_3 - 1)$ (mild)
- $k$ controls sensitivity (default: 3.0)

**Why exponential instead of linear?** The natural constant of truly-unfittable orders means $b\_3$ represents a baseline. Collections near $b\_3$ are normal — only Collections significantly worse need harsh punishment. The exponential's asymmetric shape (unbounded penalty upward, bounded reward downward) matches this physical reality.

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

$$l' = l + 2 \times t_{\text{reinforcement}}$$
$$w' = w + 2 \times t_{\text{reinforcement}}$$
$$h' = h + 2 \times t_{\text{reinforcement}}$$

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
- All dimensions clamped to `[dim_min, dim_max]` (default: 4–60)

### 6.2 Fitness Function

Fitness = combined objective function $F$ from Section 3.4, evaluated on the representative sample. **Lower is better.**

### 6.3 Operators

**Initialization**: Random box dimensions — each dimension drawn as a random integer from `[dim_min, dim_max]`. All dimensions are integers throughout the entire pipeline (initialization, crossover, mutation).

**Selection**: Tournament selection with configurable tournament size (default: 3). The individual with the lowest fitness in the tournament wins.

**Crossover**: Single-point crossover at box level. A random cut point splits the parent Collections; boxes before the cut come from parent 1, boxes after from parent 2 (and vice versa for the second child). Applied with probability `crossover_rate` (default: 0.8). Children are re-sorted by volume.

**Mutation**: Per-dimension Gaussian perturbation. Each dimension of each box is independently mutated with probability `mutation_rate` (default: 0.1). Perturbation drawn from $\mathcal{N}(0, \sigma^2)$ where $\sigma$ = `mutation_sigma` (default: 3.0). Result is rounded to integer, clamped to bounds, and re-sorted.

**Elitism**: Top `elitism_count` (default: 2) individuals are carried forward unchanged to the next generation.

**Random Immigrants**: `immigrant_count` (default: 3) fresh random individuals are injected each generation after elites, before selection/crossover. This maintains population diversity and helps escape local optima (Grefenstette, 1992).

### 6.4 GA Loop — `run()` (Naive)

```
1. Initialize population of random Collections
2. Evaluate all individuals (create Valuation per Collection)
3. For each generation:
   a. Carry forward elite individuals (elitism_count)
   b. Inject random immigrants (immigrant_count)
   c. Fill remaining population via:
      - Tournament selection of 2 parents (based on overall fitness)
      - Crossover → 2 children
      - Mutation on each child
   d. Evaluate all new individuals
   e. Track best individual and full fitness log
4. Return best Collection, best fitness, per-generation best history, and full fitness log
```

### 6.5 GA Loop — `run_by_region()` (Region-Informed)

A domain-informed enhancement that uses per-region fitness decomposition to guide crossover. See Section 6.7 for full details.

```
1. Initialize population of random Collections
2. Evaluate all individuals → (overall_fitness, region_fitness_dict, box_usage_by_region)
3. For each generation:
   a. Carry forward elite individuals (by overall fitness)
   b. Inject random immigrants
   c. Build selection pool via per-region tournaments (Section 6.7)
   d. Generate pool_ratio fraction of children from pool (assemble + mutate)
   e. Generate remaining children via traditional crossover (same as run())
   f. Evaluate all new individuals
   g. Track best individual and full fitness log
4. Return best Collection, best fitness, per-generation best history, and full fitness log
```

The `pool_ratio` parameter (default: 0.3) controls the mix:
- `pool_ratio=1.0` → all children from region pool (pure region-informed)
- `pool_ratio=0.0` → all children from traditional crossover (equivalent to `run()`)
- `pool_ratio=0.3` → 30% pool-based, 70% traditional crossover (default)

### 6.6 History & Diagnostics

The GA records two history structures:
- **history**: List of best fitness per generation (length = generations + 1). For plotting convergence curves.
- **fitness_log**: List of lists — all population fitnesses per generation. For scatter-plotting fitness distributions and analyzing diversity over time.

### 6.7 Region-Decomposed Gene Pool Recombination

A domain-informed crossover mechanism that identifies the best-contributing boxes from the best-performing Collections in each region, then assembles new children from that pool. This approach combines ideas from:

- **Cooperative Coevolution** (Potter & De Jong, 2000): Decompose the problem into subcomponents (regions), evolve each independently, assemble solutions from the best parts.
- **Gene Pool Recombination** (Mühlenbein & Voigt, 1996): Instead of pairwise crossover, build a pool of alleles from good individuals, then sample new individuals from the pool.
- **MOEA/D** (Zhang & Li, 2007): Decompose into scalar subproblems and use subproblem-specific selection.

The distinctive aspect of our approach: the decomposition is on the **evaluation side** (order regions defined by stratified bins) rather than the decision variable side — we use evaluation-side decomposition to inform which decision variables (box positions) to inherit from which parents.

#### 6.7.1 Box Usage Tracking

During Valuation, for each fittable order, we record which box was assigned in which region:

```
box_usage_by_region = {region: {box_name: count}}
```

For example: `{(2, 3, 1): {"Box3": 45, "Box4": 30, "Box5": 12}}` means in region (2,3,1), Box3 was used 45 times, Box4 was used 30 times, etc.

#### 6.7.2 Per-Region Tournament Selection

For each region (up to 125), run `region_tournament_rounds` (default: 3) tournaments:

1. Sample `region_tournament_size` (default: 20) individuals from the population.
2. Select the winner — the individual with the **lowest fitness in that specific region** (using `results_by_region()`).
3. From the winner, take the **top `top_boxes_per_region`** (default: 3) most-used boxes in that region.
4. Record these boxes' positions in the selection pool.

#### 6.7.3 Selection Pool

The pool aggregates across all regions into:

```
pool = {box_position (0-indexed): set of collection indices}
```

**Key properties:**
- Medium-sized box positions (3–6) tend to have many candidate Collections (they serve many regions).
- Extreme positions (smallest/largest) have fewer candidates — this correctly reflects that fewer Collections excel at handling extreme orders.
- The asymmetry in pool size across positions is a feature, not a bug: it concentrates selection pressure where it matters.

#### 6.7.4 Child Construction

For each new child:
1. For each box position 0 to N-1:
   - If the pool has candidates at this position: randomly pick one Collection from the candidates, take its box dimensions at that position.
   - If no candidates (uncovered position): generate random box dimensions as fallback.
2. Re-sort the assembled child by volume for consistency.
3. Apply standard Gaussian mutation.

#### 6.7.5 Hybrid Ratio

The `pool_ratio` parameter controls what fraction of children are produced via region pool vs. traditional crossover. This allows empirical tuning:

| pool_ratio | Behavior |
|------------|----------|
| 1.0 | Pure region-informed (all children from pool) |
| 0.3 | 30% pool-based, 70% traditional crossover (default) |
| 0.0 | Pure traditional crossover (equivalent to `run()`) |

#### 6.7.6 Design Considerations

**Epistasis (box synergy)**: Assembling boxes from different Collections creates combinations that weren't optimized together. However, re-evaluation + mutation in subsequent generations can "heal" suboptimal combinations. This is the primary tradeoff vs. `run()`.

**Pool refresh**: The pool is rebuilt from scratch each generation based on newly evaluated individuals. Combined with mutation, this prevents stale selection pressure.

**Comparison with `run()`**: Both approaches will be run on the same data with identical parameters (except `pool_ratio`). Comparing convergence speed and final fitness will determine whether the region-informed approach justifies its added complexity. This comparison is intended for paper publication.

## 7. Dependencies

- **algorithm_BPS**: External module (Box Packing Solution) providing the core packing algorithm. Imported via `sys.path` configuration pointing to the sibling directory `../Box Packing Solution/`.
- **Python libraries**: pandas, numpy, scipy (KS test), copy, multiprocessing, threading, math, random, os.

## 8. Parameters Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| number_of_boxes | (input) | Number of box types in a Collection |
| utilization_optimal | 0.8 | Target box utilization ratio |
| reinforcement_thickness | 0.5 cm | Protective wrapping thickness per side |
| bubble_thickness | 1.0 cm | Bubble wrap material thickness |
| bubble_filling_rate | 0.5 | Fraction of empty space filled with bubble wrap |
| objective_function_1_baseline | 5,000 | Cost normalization baseline |
| objective_function_2_baseline | 0.1 | Utilization deviation normalization baseline |
| objective_function_3_baseline | 0.05 | Natural unfittable ratio baseline |
| w1, w2, w3 | 0.6, 0.35, 0.05 | Objective function weights (sum = 1.0) |
| k | 3.0 | Exponential penalty sensitivity for f3 |
| sample_size | 100,000 | Number of orders in representative sample |
| number_of_regions_per_aspect | 5 | Number of quantile bins per stratification feature |
| population_size | 50 | GA population size |
| generations | 100 | GA number of generations |
| mutation_rate | 0.1 | GA per-dimension mutation probability |
| crossover_rate | 0.8 | GA crossover probability |
| tournament_size | 3 | GA tournament selection size (for `run()` and traditional crossover in `run_by_region()`) |
| elitism_count | 2 | GA number of elite individuals carried forward |
| immigrant_count | 3 | Random immigrants injected per generation |
| region_tournament_size | 20 | Tournament size for per-region selection in `run_by_region()` |
| region_tournament_rounds | 3 | Number of tournaments per region for pool building |
| top_boxes_per_region | 3 | Top contributing boxes taken per region per tournament winner |
| pool_ratio | 0.3 | Fraction of children from region pool vs. traditional crossover |
| dim_min | 4 | Minimum box dimension (cm) |
| dim_max | 60 | Maximum box dimension (cm) |
| mutation_sigma | 3.0 | Gaussian mutation standard deviation (cm) |
