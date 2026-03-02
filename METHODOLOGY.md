# Box Dimension Optimization — Methodology

## 1. Problem Statement

Given a warehouse operation that ships orders of varying sizes, we need to determine the optimal set of carton box dimensions (a "Collection") that minimizes packaging cost while maximizing box utilization across all orders.

The core question: **Is the current collection of box sizes optimal, or should we adjust individual box dimensions to reduce cost and improve space utilization?**

### Definitions

- **Order**: A customer order containing one or more items, each with known dimensions (length, width, height) and quantity.
- **Collection**: A fixed set of N box types, each defined by (length, width, height). This is what we optimize.
- **Box Packing Solution (algorithm_BPS)**: A pre-existing algorithm that, given an order and a collection, determines the best-fitting box and the arrangement of items inside it.

## 2. Architecture Overview

The system is structured as a nested optimization:

1. **Inner layer** — `algorithm_BPS`: Solves "given this order and these boxes, which box fits best and how should items be placed?"
2. **Outer layer** — Genetic Algorithm: Searches for the best Collection (set of box dimensions) by evaluating many candidate Collections using historical order data.

The evaluation of each candidate Collection requires running `algorithm_BPS` on all (or a sample of) historical orders, then computing objective function values.

## 3. Objective Functions

We define two objective functions, both computed as per-order averages across all evaluated orders.

### 3.1 Objective Function 1 — Cost Minimization

For each order i assigned to a box:

$$f_1 = \frac{1}{N} \sum_{i=1}^{N} \left( C_{box,i} + C_{bubble,i} \right)$$

Where:
- $C_{box,i}$ = cost of the chosen box, proportional to its surface area: $2(lw + lh + wh) \times \text{unit\_cost}$
- $C_{bubble,i}$ = cost of bubble wrap filling the unused space: $\frac{61000}{35 \times 90 \times 100 \times t} \times V_{bubble,i}$
- $V_{bubble,i} = (V_{box,i} - V_{items,i}) \times \text{filling\_rate}$
- $t$ = bubble thickness
- $N$ = total number of orders

### 3.2 Objective Function 2 — Utilization Optimization

$$f_2 = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{V_{items,i}}{V_{box,i}} - u^* \right)^2$$

Where:
- $V_{items,i}$ = total volume of items in order i (after adding reinforcement thickness)
- $V_{box,i}$ = volume of the chosen box
- $u^*$ = target utilization ratio (default: 0.9)

This is essentially the Mean Squared Error (MSE) of utilization relative to the target. Using mean instead of sum makes it scale-independent and comparable across different sample sizes.

### 3.3 Combined Objective — Weighted Sum with Normalization

The two objectives are on different scales (cost in monetary units, utilization gap as a small decimal). To combine them, we normalize each by a baseline value derived from the current real Collection:

$$F = w_1 \cdot \frac{f_1}{f_1^{baseline}} + w_2 \cdot \frac{f_2}{f_2^{baseline}}$$

Where:
- $f_1^{baseline}$, $f_2^{baseline}$ = objective values from the current Collection
- $w_1$, $w_2$ = weights reflecting business priority (e.g., 0.5/0.5 for equal importance)

This normalization means $F < 1.0$ indicates improvement over the current Collection.

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
4. Within each bucket, individual orders are processed in threads.

## 5. Representative Sampling

### 5.1 Motivation

Each Genetic Algorithm fitness evaluation requires running `algorithm_BPS` on all historical orders. With population_size=50, generations=100, and tens of thousands of orders, this is computationally prohibitive. A representative sample reduces evaluation cost while maintaining statistical validity.

### 5.2 Stratified Proportional Sampling

We create a sample that preserves the distribution of order characteristics that influence box selection.

**Stratification features** (computed per order):
- **Total volume**: Sum of (length x width x height x unit) across all items in the order. Determines minimum box size.
- **Max single dimension**: The largest dimension among all items. Forces minimum box length regardless of volume.
- **Total units**: Number of items. Multi-item orders are harder to pack and may require different box choices.

**Procedure**:
1. Compute per-order summary statistics (total_volume, max_length, total_unit).
2. Create quantile-based bins for each feature using `pd.qcut` with `duplicates='drop'` (handles cases where many orders share the same value, e.g., most orders have 1 unit).
3. Define strata as unique combinations of all bins (up to $5^3 = 125$ strata with 5 bins per feature).
4. Sample proportionally from each stratum: each stratum contributes to the sample in proportion to its size in the full dataset, with a minimum of 1 order per stratum to ensure representation.
5. All rows belonging to a sampled order are included (orders are never split across sample/non-sample).

### 5.3 Sample Validation

Two complementary tests validate the sample:

**Test 1 — Kolmogorov-Smirnov (KS) Test**

Checks whether the sample and full data have the same distribution for each stratification feature.

- Metric: KS statistic (maximum distance between cumulative distribution functions) and p-value.
- Pass criteria: KS statistic < 0.1 AND p-value > 0.05 for all features.
- Note: For large datasets, prioritize KS statistic over p-value, as the test becomes overly sensitive to small, practically meaningless differences.

**Test 2 — Valuation Comparison**

Checks whether the sample produces similar objective function values as the full dataset.

- Run Valuation on both full data and sample using the same (current) Collection.
- Pass criteria: Combined objective function values differ by less than 10%.
- This is the ultimate validation — if outputs match, the sample is adequate regardless of KS results.

### 5.4 Troubleshooting Poor Samples

If validation fails:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| KS test fails for one feature | Bins too coarse due to non-unique values | Reduce bins for that feature or use manual boundaries |
| KS passes but Valuation fails | Missing stratification feature | Add features (e.g., aspect ratio, min dimension) |
| Both fail | Sample too small | Increase sample_size |

## 6. Genetic Algorithm (Planned)

### 6.1 Encoding

Each individual (chromosome) in the GA population represents a candidate Collection: a list of N box dimensions `[[l1, w1, h1], ..., [lN, wN, hN]]`.

### 6.2 Fitness Function

Fitness = combined objective function $F$ from Section 3.3, evaluated on the representative sample. Lower is better.

### 6.3 Region-Based Crossover (Planned)

A domain-informed approach to accelerate convergence:

1. Define "regions" based on order characteristics (e.g., small/medium/large volume combined with item count).
2. After each generation, compute per-region Valuation metrics.
3. Identify which boxes primarily serve which regions.
4. For well-performing regions, preserve the responsible boxes (reduced mutation rate).
5. Focus mutation and crossover on boxes serving poorly-performing regions.

This exploits the structure of the problem: small boxes serve small orders, large boxes serve large orders. If small-order performance is already good, there is no need to disrupt those box dimensions while improving large-order performance.

### 6.4 Operators

- **Selection**: Tournament or roulette wheel based on fitness.
- **Crossover**: Exchange box dimensions between two parent Collections, informed by region performance.
- **Mutation**: Perturb individual box dimensions (increase/decrease length, width, or height by a small amount).

## 7. Dependencies

- **algorithm_BPS**: External module (Box Packing Solution) providing the core packing algorithm. Imported via `sys.path` configuration pointing to the sibling directory.
- **Python libraries**: pandas, numpy, scipy (KS test), copy, multiprocessing, threading.

## 8. Parameters Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| number_of_boxes | (input) | Number of box types in a Collection |
| utilization_optimal | 0.9 | Target box utilization ratio |
| reinforcement_thickness | 0.5 cm | Protective wrapping thickness per side |
| bubble_thickness | 1.0 cm | Bubble wrap material thickness |
| bubble_filling_rate | 0.5 | Fraction of empty space filled with bubble wrap |
| sample_size | 10,000 | Number of orders in representative sample |
| number_of_buckets_per_aspect | 5 | Number of quantile bins per stratification feature |
| population_size | 50 | GA population size |
| generations | 100 | GA number of generations |
| mutation_rate | 0.01 | GA mutation probability |
| w1, w2 | 0.5, 0.5 | Objective function weights |
