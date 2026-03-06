# pylint: disable=C0103
# pylint: disable=E0102

import sys
import math
import random
import os
from pathlib import Path
from scipy.stats import ks_2samp, chi2

# Goes up to parent of Box Dimension Optimization, then into Box Packing Solution
sys.path.append(str(Path(__file__).parent.parent.parent / "Box Packing Solution"))

from algorithm_BPS import *

class Collection:
    def __init__(
            self,
            number_of_boxes: int,
            # list of lists of dimensions of the boxes, where each list is [length, width, height]
            boxes_dimensions: list
    ) -> None: 
        
        self.number_of_boxes = number_of_boxes
        self.boxes_dimensions = boxes_dimensions

        self.boxes_volumes = []
        self.boxes_cost = []
        for box in boxes_dimensions:
            length, width, height = box
            self.boxes_volumes.append(length * width * height)
            # assume the cost of the box is proportional to its surface area, and the cost per unit area is 1000 --> will update later after checking with Procurement team
            self.boxes_cost.append(2*(length*width + length*height + width*height) * 1000)

        self.df_boxes = pd.DataFrame({
            'id': list(range(1, number_of_boxes+1)),
            'box': ['Box' + str(i) for i in range(1, number_of_boxes+1)],
            'length': [box[0] for box in boxes_dimensions],
            'width': [box[1] for box in boxes_dimensions],
            'height': [box[2] for box in boxes_dimensions],
            'volume': self.boxes_volumes,
            'cost': self.boxes_cost
        })

class Bubble:
    def __init__(
            self,
            bubble_thickness: float = 1.0,
            volume_used: float = 0.0
    ) -> None:
        
        self.bubble_thickness = bubble_thickness
        self.volume_used = volume_used

        self.bubble_cost = 61000/(35*90*100*self.bubble_thickness) * self.volume_used

# F(Order Info, Collection) = Assigned Box
# Assigned Box + Order Info = Bubble

def process(
        df_bucket, 
        df_box,
        filename
        ):
    order_list = df_bucket['order_code'].unique()
    for order in order_list:
        df_order = df_bucket[df_bucket['order_code'] == order]
        k = Thread(
            target=box_packing_solution,
            args=(df_order, df_box, 'balanced', 100, 100, 30, filename)
        )

        k.start()
        k.join()

class Valuation:
    def __init__(
            self,
            data_order: pd.DataFrame,
            number_of_buckets: int,
            collection: Collection,
            filename: str,
            utilization_optimal: float = 0.8,
            reinforcement_thickness: float = 0.5,
            bubble_thickness: float = 1.0,
            bubble_filling_rate: float = 0.5,
            objective_function_1_baseline: float = 5000.0,  # baseline cost for normalization
            objective_function_2_baseline: float = 0.1,  # baseline utilization deviation for normalization,
            objective_function_3_baseline: float = 0.05,
            objective_function_1_by_region_baseline: dict = None,  # optional baseline by region for objective function 1
            objective_function_2_by_region_baseline: dict = None,  # optional baseline by region for objective function 2
            objective_function_3_by_region_baseline: dict = None
    ) -> None:
        self.objective_function_1_baseline = objective_function_1_baseline
        self.objective_function_2_baseline = objective_function_2_baseline
        self.objective_function_3_baseline = objective_function_3_baseline
        self.objective_function_1_by_region_baseline = objective_function_1_by_region_baseline if objective_function_1_by_region_baseline is not None else {}
        self.objective_function_2_by_region_baseline = objective_function_2_by_region_baseline if objective_function_2_by_region_baseline is not None else {}
        self.objective_function_3_by_region_baseline = objective_function_3_by_region_baseline if objective_function_3_by_region_baseline is not None else {}

        df_box = collection.df_boxes
        
        print(f"[Valuation] Starting evaluation with {len(data_order['order_code'].unique())} orders, {number_of_buckets} buckets...", flush=True)
        data_order_mod = copy.deepcopy(data_order)  # to avoid modifying the original dataframe
        # modify data_order to include reinforcement thickness for each item
        data_order_mod['length'] = data_order_mod['length'] + 2*reinforcement_thickness
        data_order_mod['width'] = data_order_mod['width'] + 2*reinforcement_thickness
        data_order_mod['height'] = data_order_mod['height'] + 2*reinforcement_thickness

        # Get unique orders and assign them round-robin bucket numbers
        unique_orders = data_order_mod['order_code'].unique()
        bucket_assignment = {order: i % number_of_buckets for i, order in enumerate(unique_orders)}
        
        # Map those assignments back to all rows (ensures all rows of same order go to same bucket)
        data_order_mod['bucket'] = data_order_mod['order_code'].map(bucket_assignment)
        
        # Create buckets dictionary
        buckets = {}
        for bucket in data_order_mod['bucket'].unique():
            buckets[bucket] = data_order_mod[data_order_mod['bucket'] == bucket]
        
        # Create all processes
        processes = []
        for bucket, df_bucket in buckets.items():
            p = Process(target=process, args=(df_bucket, df_box, filename))
            processes.append(p)
        
        # Start all processes in parallel
        print(f"[Valuation] Launching {len(processes)} parallel processes...", flush=True)
        for p in processes:
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        print(f"[Valuation] All processes completed. Parsing results...", flush=True)

        # After all processes are done, we can read the results from the files and calculate the valuation metrics
        # read results from file
        # CSV columns from algorithm_BPS: ['order', 'box_final', 'item', 'position', 'process_time']
        df_result = pd.read_csv(filename)

        # Unfittable markers from algorithm_BPS
        unfittable_markers = {"No satisfied Box", "Cannot find any satisfied Box in the given time"}

        # Calculate valuation metrics — f1/f2 only over fittable orders, count unfittable separately
        objective_function_1 = 0
        objective_function_2 = 0
        objective_function_3 = 0
        objective_function_1_by_region = {}
        objective_function_2_by_region = {}
        objective_function_3_by_region = {}
        orders_by_region = {}  # total orders per region (for correct per-region averaging)
        box_usage_by_region = {}  # box usage per region for region-based GA: {region: {box_name: count}}

        for order in tqdm(unique_orders):
            df_order_temp = data_order_mod[data_order_mod['order_code'] == order]
            item_volume = (df_order_temp['length'] * df_order_temp['width'] * df_order_temp['height'] * df_order_temp['unit']).sum()
            region = (df_order_temp['volume_bin'].iloc[0], df_order_temp['length_bin'].iloc[0], df_order_temp['unit_bin'].iloc[0])  # region definition based on stratification bins

            orders_by_region[region] = orders_by_region.get(region, 0) + 1

            df_result_temp = df_result[df_result['order'] == order]

            # Check if order is unfittable (no result row, or box_final is a known unfittable marker)
            if len(df_result_temp) == 0 or df_result_temp['box_final'].iloc[0] in unfittable_markers:
                objective_function_3 += 1
                objective_function_3_by_region[region] = objective_function_3_by_region.get(region, 0) + 1
                continue

            # Fittable order — look up the chosen box by name
            chosen_box_name = df_result_temp['box_final'].iloc[0]
            df_box_temp = df_box[df_box['box'] == chosen_box_name]

            if len(df_box_temp) == 0:
                # box name not found in collection (shouldn't happen, but guard against it)
                objective_function_3 += 1
                objective_function_3_by_region[region] = objective_function_3_by_region.get(region, 0) + 1
                continue

            chosen_box_volume = df_box_temp['volume'].iloc[0]
            chosen_box_cost = df_box_temp['cost'].iloc[0]

            filling_bubble_usage_volume = (chosen_box_volume - item_volume) * bubble_filling_rate
            bubble_usage = Bubble(bubble_thickness, filling_bubble_usage_volume)

            # Objective function 1: minimize total cost (box cost + bubble cost)
            objective_function_1 += chosen_box_cost + bubble_usage.bubble_cost
            objective_function_1_by_region[region] = objective_function_1_by_region.get(region, 0) + chosen_box_cost + bubble_usage.bubble_cost
            # Objective function 2: optimize total utilization (item volume / box volume)
            objective_function_2 += (item_volume / chosen_box_volume - utilization_optimal)**2
            objective_function_2_by_region[region] = objective_function_2_by_region.get(region, 0) + (item_volume / chosen_box_volume - utilization_optimal)**2
            # Track box usage per region for region-based GA
            if region not in box_usage_by_region:
                box_usage_by_region[region] = {}
            box_usage_by_region[region][chosen_box_name] = box_usage_by_region[region].get(chosen_box_name, 0) + 1
            # Objective function 3: unfittable ratio, already counted above as objective_function_3 (count of unfittable orders)

        # Average over fittable orders only (avoid division by zero)
        self.objective_function_1 = objective_function_1 / (len(unique_orders) - objective_function_3) if (len(unique_orders) - objective_function_3) > 0 else float('inf')
        self.objective_function_2 = objective_function_2 / (len(unique_orders) - objective_function_3) if (len(unique_orders) - objective_function_3) > 0 else float('inf')
        # Unfittable ratio: measures Collection complexity/sparsity
        self.objective_function_3 = objective_function_3 / len(unique_orders)

        # Average by_region metrics using per-region order counts
        self.objective_function_1_by_region = {}
        self.objective_function_2_by_region = {}
        self.objective_function_3_by_region = {}
        self.orders_by_region = orders_by_region
        self.box_usage_by_region = box_usage_by_region

        for region, n_total in orders_by_region.items():
            n_unfittable = objective_function_3_by_region.get(region, 0)
            n_fittable = n_total - n_unfittable
            self.objective_function_1_by_region[region] = objective_function_1_by_region.get(region, 0) / n_fittable if n_fittable > 0 else float('inf')
            self.objective_function_2_by_region[region] = objective_function_2_by_region.get(region, 0) / n_fittable if n_fittable > 0 else float('inf')
            self.objective_function_3_by_region[region] = n_unfittable / n_total

        n_fittable_total = len(unique_orders) - int(objective_function_3 * len(unique_orders)) if self.objective_function_3 < 1 else 0
        print(f"[Valuation] Done — f1={self.objective_function_1:.2f}, f2={self.objective_function_2:.6f}, f3={self.objective_function_3:.4f} ({len(unique_orders)} orders, {len(orders_by_region)} regions)", flush=True)

    def results(self, w1: float = 0.6, w2: float = 0.35, w3: float = 0.05, k: float = 3.0):
        """Combined objective function with exponential unfittable penalty.
        
        F = w1 * (f1/b1) + w2 * (f2/b2) + w3 * g(f3)
        where g(f3) = exp(k * (f3/b3 - 1)) - 1
        
        - w1 + w2 + w3 = 1.0 (w1=0.6 cost, w2=0.35 utilization, w3=0.05 unfittable penalty)
        - f1/f2 are computed only over fittable orders (clean separation)
        - g(f3) is centered at b3: neutral when f3=b3, harsh penalty when f3>>b3, reward when f3<<b3
        - k controls sensitivity (k=1 mild, k=3-5 aggressive)
        """
        g_f3 = math.exp(k * (self.objective_function_3 / self.objective_function_3_baseline - 1)) - 1
        return (w1 * (self.objective_function_1 / self.objective_function_1_baseline) 
                + w2 * (self.objective_function_2 / self.objective_function_2_baseline) 
                + w3 * g_f3)

    def results_by_region(self, w1: float = 0.6, w2: float = 0.35, w3: float = 0.05, k: float = 3.0):
        # Calculate combined fitness per region using exponential penalty for f3
        results_by_region = {}
        for region in self.orders_by_region.keys():
            f1 = self.objective_function_1_by_region.get(region, float('inf'))
            f2 = self.objective_function_2_by_region.get(region, float('inf'))
            f3 = self.objective_function_3_by_region.get(region, 1.0)  # default to worst case if region missing
            b3 = self.objective_function_3_by_region_baseline.get(region, self.objective_function_3_baseline)
            g_f3 = math.exp(k * (f3 / b3 - 1)) - 1 if b3 > 0 else (0.0 if f3 == 0 else float('inf'))
            results_by_region[region] = (w1 * (f1 / self.objective_function_1_baseline) 
                                        + w2 * (f2 / self.objective_function_2_baseline) 
                                        + w3 * g_f3)
        return results_by_region


class Sample:
    ''' This class is used to create a small sample from the whole dataset for testing purposes via stratified sampling, to speed up the development process.
    It would be tested by applying:
    - KS test: To check if the distribution of the sample is similar to the distribution of the whole dataset.
    - Valuation metrics: To check if the valuation metrics calculated on the sample are similar to the valuation metrics calculated on the whole dataset. Would use the current default Collection.

    If the sample passes these tests, we can be more confident that the sample is representative of the whole dataset and can be used for testing and development purposes.
    '''
    def __init__(
            self,
            data_order: pd.DataFrame,
            sample_size: int = 10000,
            number_of_buckets_per_aspect: int = 5
    ) -> None:
        temp_data_order = copy.deepcopy(data_order)
        n_orders = temp_data_order['order_code'].nunique()
        print(f"[Sample] Building stratified sample: {sample_size} from {n_orders} orders, {number_of_buckets_per_aspect} bins/aspect...", flush=True)
        
        # Build per-order summary for stratification (vectorized — avoids slow per-group .apply())
        temp_data_order['_item_volume'] = temp_data_order['length'] * temp_data_order['width'] * temp_data_order['height'] * temp_data_order['unit']
        grouped = temp_data_order.groupby('order_code')
        order_summary = pd.DataFrame({
            'total_volume': grouped['_item_volume'].sum(),
            'max_length': temp_data_order[['order_code', 'length', 'width', 'height']].melt(id_vars='order_code', value_name='dim').groupby('order_code')['dim'].max(),
            'total_unit': grouped['unit'].sum()
        }).reset_index()
        temp_data_order.drop(columns=['_item_volume'], inplace=True)

        # Create bins for stratification (use duplicates='drop' to handle non-unique values)
        order_summary['volume_bin'] = pd.qcut(order_summary['total_volume'], q=number_of_buckets_per_aspect, labels=False, duplicates='drop')
        order_summary['length_bin'] = pd.qcut(order_summary['max_length'], q=number_of_buckets_per_aspect, labels=False, duplicates='drop')
        order_summary['unit_bin'] = pd.qcut(order_summary['total_unit'], q=number_of_buckets_per_aspect, labels=False, duplicates='drop')

        # Stratified proportional sampling at ORDER level (not row level)
        sample_frac = sample_size / len(order_summary)
        
        sampled_parts = []
        for _, group in order_summary.groupby(['volume_bin', 'length_bin', 'unit_bin']):
            n = max(1, int(len(group) * sample_frac))
            if len(group) > 1:
                sampled_parts.append(group.sample(n, replace=False))
            else:
                sampled_parts.append(group)
        sampled_orders = pd.concat(sampled_parts, ignore_index=True)

        # Store order summary with features for KS test
        self.order_summary_full = order_summary
        self.order_summary_sample = sampled_orders

        # Filter original data to only include sampled orders (keeps all rows per order)
        sampled_order_codes = sampled_orders['order_code'].unique()
        self.sample = temp_data_order[temp_data_order['order_code'].isin(sampled_order_codes)].copy()
        self.data_order = temp_data_order

        # Merge bin columns into data so Valuation can use them for region-based metrics
        bin_cols = order_summary[['order_code', 'volume_bin', 'length_bin', 'unit_bin']]
        self.sample = self.sample.merge(bin_cols, on='order_code', how='left')
        self.data_order = self.data_order.merge(bin_cols, on='order_code', how='left')
        print(f"[Sample] Done — sampled {self.order_summary_sample['order_code'].nunique()} orders across {sampled_orders.groupby(['volume_bin','length_bin','unit_bin']).ngroups} strata", flush=True)
    
    def ks_test(self, p_value_threshold: float = 0.05, ks_stat_threshold: float = 0.1):
        # Perform KS test between the sample and the whole dataset for key aspects
        result = {}
        for aspect in ['total_volume', 'max_length', 'total_unit']:
            sample_values = self.order_summary_sample[aspect]
            data_values = self.order_summary_full[aspect]
            ks_stat, p_value = ks_2samp(sample_values, data_values)
            print(f"KS test for {aspect}: KS statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")
            result[aspect] = (ks_stat, p_value)

        # return "Passed" if all p-values > 0.05 and ks_stat < 0.1 (assuming these thresholds for similarity), otherwise return "Failed"
        if all(p > p_value_threshold and ks < ks_stat_threshold for ks, p in result.values()):
            return "KS Test Passed"
        else:
            return "KS Test Failed"
        
    def valuation_test(self, collection: Collection, filename: str, w1: float = 0.6, w2: float = 0.35, w3: float = 0.05, tolerance: float = 0.1):
        # This function can be implemented to calculate the valuation metrics on the sample and compare them with the valuation metrics calculated on the whole dataset using the same Collection and filename for results.
        print(f"[Valuation_test] Evaluating sample...", flush=True)
        valuation_sample = Valuation(self.sample, number_of_buckets=5, collection=collection, filename=filename)
        print(f"[Valuation_test] Evaluating full dataset...", flush=True)
        valuation_full = Valuation(self.data_order, number_of_buckets=5, collection=collection, filename=filename)

        print(f"Valuation results for sample: {valuation_sample.results(w1, w2, w3)}")
        print(f"Valuation results for full dataset: {valuation_full.results(w1, w2, w3)}")
        print(f"Unfittable ratio — sample: {valuation_sample.objective_function_3:.4f}, full: {valuation_full.objective_function_3:.4f}")

        # return "Passed" if the valuation results are within 10% of each other, otherwise return "Failed"
        if abs(valuation_sample.results(w1, w2, w3) - valuation_full.results(w1, w2, w3)) / valuation_full.results(w1, w2, w3) < tolerance:
            return "Valuation Test Passed"
        else:
            return "Valuation Test Failed"
    
class GeneticAlgorithm:
    def __init__(
            self,
            data_order: pd.DataFrame,
            number_of_boxes: int,
            number_of_buckets: int = 5,
            population_size: int = 50,
            generations: int = 100,
            mutation_rate: float = 0.1,
            crossover_rate: float = 0.8,
            tournament_size: int = 3,
            elitism_count: int = 2,
            immigrant_count: int = 3,
            region_tournament_size: int = 20,
            region_tournament_rounds: int = 3,
            top_boxes_per_region: int = 3,
            pool_ratio: float = 0.5,
            dim_min: int = 5,
            dim_max: int = 60,
            mutation_sigma: float = 3.0,
            valuation_kwargs: dict = None,  # extra kwargs for Valuation (baselines, thicknesses, etc.)
            results_kwargs: dict = None,    # extra kwargs for results() (w1, w2, w3, k)
            filename: str = '/tmp/ga_temp_result.csv'  # /tmp/ is local-only, bypasses iCloud sync
    ) -> None:
        self.data_order = data_order
        self.number_of_boxes = number_of_boxes
        self.number_of_buckets = number_of_buckets
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.immigrant_count = immigrant_count
        self.region_tournament_size = region_tournament_size
        self.region_tournament_rounds = region_tournament_rounds
        self.top_boxes_per_region = top_boxes_per_region
        self.pool_ratio = pool_ratio
        self.dim_min = dim_min
        self.dim_max = dim_max
        self.mutation_sigma = mutation_sigma
        self.valuation_kwargs = valuation_kwargs or {}
        self.results_kwargs = results_kwargs or {}
        self.filename = filename

    def _random_individual(self):
        """Generate random box dimensions: list of [L, W, H], sorted by volume."""
        boxes = []
        for _ in range(self.number_of_boxes):
            dims = sorted([random.randint(self.dim_min, self.dim_max) for _ in range(3)], reverse=True)
            boxes.append(dims)
        # Sort boxes by volume (ascending) for consistency
        boxes.sort(key=lambda b: b[0] * b[1] * b[2])
        return boxes

    def _evaluate(self, individual):
        """Evaluate fitness of an individual (lower is better)."""
        # Remove stale CSV so results don't bleed across evaluations
        if os.path.exists(self.filename):
            os.remove(self.filename)
        collection = Collection(self.number_of_boxes, individual)
        valuation = Valuation(
            data_order=self.data_order,
            number_of_buckets=self.number_of_buckets,
            collection=collection,
            filename=self.filename,
            **self.valuation_kwargs
        )
        return (
            valuation.results(**self.results_kwargs),
            valuation.results_by_region(**self.results_kwargs),
            valuation.box_usage_by_region
        )

    def _tournament_select(self, population, fitnesses):
        """Select one individual via tournament selection (lower fitness wins)."""
        indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = min(indices, key=lambda i: fitnesses[i])
        return copy.deepcopy(population[best_idx])

    def _crossover(self, parent1, parent2):
        """Single-point crossover: swap boxes after a random cut point."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        point = random.randint(1, self.number_of_boxes - 1)
        child1 = copy.deepcopy(parent1[:point]) + copy.deepcopy(parent2[point:])
        child2 = copy.deepcopy(parent2[:point]) + copy.deepcopy(parent1[point:])

        # Re-sort by volume for consistency
        child1.sort(key=lambda b: b[0] * b[1] * b[2])
        child2.sort(key=lambda b: b[0] * b[1] * b[2])
        return child1, child2

    def _mutate(self, individual):
        """Gaussian mutation: perturb random dimensions, clamp to bounds, keep L >= W >= H."""
        mutated = copy.deepcopy(individual)
        for i in range(len(mutated)):
            for j in range(3):
                if random.random() < self.mutation_rate:
                    mutated[i][j] = int(round(mutated[i][j] + random.gauss(0, self.mutation_sigma)))
                    mutated[i][j] = max(self.dim_min, min(self.dim_max, mutated[i][j]))
            # Keep L >= W >= H within each box
            mutated[i] = sorted(mutated[i], reverse=True)
        # Re-sort boxes by volume (ascending)
        mutated.sort(key=lambda b: b[0] * b[1] * b[2])
        return mutated

    def _region_tournament_select(self, region_fitnesses, region):
        """Tournament selection for a specific region (lower fitness wins).

        Args:
            region_fitnesses: list of dicts, one per individual, {region: fitness_value}
            region: the region key to select on
        Returns:
            Index of the winning individual.
        """
        indices = random.sample(range(len(region_fitnesses)), min(self.region_tournament_size, len(region_fitnesses)))
        best_idx = min(indices, key=lambda i: region_fitnesses[i].get(region, float('inf')))
        return best_idx

    def _build_selection_pool(self, population, region_fitnesses, box_usages):
        """Build selection pool via per-region tournaments.

        For each region, run multiple tournaments.  For each winner, take
        its top contributing boxes for that region.  Merge results across all
        regions into: {box_position (0-indexed): set of collection indices}.
        """
        pool = {}  # {box_position: set of collection_indices}

        # Collect all regions present across individuals
        all_regions = set()
        for rf in region_fitnesses:
            all_regions.update(rf.keys())

        for region in all_regions:
            for _ in range(self.region_tournament_rounds):
                winner_idx = self._region_tournament_select(region_fitnesses, region)

                # Top-K contributing boxes for this region from the winner
                usage = box_usages[winner_idx].get(region, {})
                if not usage:
                    continue
                top_boxes = sorted(usage.items(), key=lambda x: -x[1])[:self.top_boxes_per_region]

                for box_name, _ in top_boxes:
                    # "Box1" -> position 0, "Box2" -> position 1, …
                    box_pos = int(box_name.replace("Box", "")) - 1
                    if box_pos not in pool:
                        pool[box_pos] = set()
                    pool[box_pos].add(winner_idx)

        return pool

    def _construct_child_from_pool(self, pool, population):
        """Construct one child by picking, for each box position, a random
        collection from the pool and taking that collection's box at that position.

        Positions not covered by the pool fall back to a random box.
        """
        child = []
        for pos in range(self.number_of_boxes):
            candidates = pool.get(pos)
            if candidates:
                chosen_idx = random.choice(list(candidates))
                child.append(copy.deepcopy(population[chosen_idx][pos]))
            else:
                # Fallback: random box dimensions
                dims = sorted([random.randint(self.dim_min, self.dim_max) for _ in range(3)], reverse=True)
                child.append(dims)
        # Re-sort by volume (ascending) for consistency
        child.sort(key=lambda b: b[0] * b[1] * b[2])
        return child

    def run(self):
        """Run the GA. Returns (best_collection, best_fitness, history)."""
        # Initialize population
        population = [self._random_individual() for _ in range(self.population_size)]

        # Evaluate initial population
        fitnesses = [self._evaluate(ind)[0] for ind in population]

        # Track best
        best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_individual = copy.deepcopy(population[best_idx])
        best_fitness = fitnesses[best_idx]
        history = [best_fitness]               # best fitness per generation
        fitness_log = [list(fitnesses)]         # all fitnesses per generation (list of lists)

        print(f"Gen 0: best fitness = {best_fitness:.6f}", flush=True)

        for gen in range(1, self.generations + 1):
            # Elitism: carry forward top N
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:self.elitism_count]
            new_population = [copy.deepcopy(population[i]) for i in elite_indices]

            # Random immigrants: inject fresh random individuals to maintain diversity
            for _ in range(self.immigrant_count):
                if len(new_population) < self.population_size:
                    new_population.append(self._random_individual())

            # Fill rest via selection, crossover, mutation
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population
            fitnesses = [self._evaluate(ind)[0] for ind in population]

            # Update best
            gen_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] < best_fitness:
                best_individual = copy.deepcopy(population[gen_best_idx])
                best_fitness = fitnesses[gen_best_idx]

            history.append(best_fitness)
            fitness_log.append(list(fitnesses))

            if gen % 10 == 0 or gen == self.generations:
                print(f"Gen {gen}: best fitness = {best_fitness:.6f}", flush=True)

        # Clean up temp file
        if os.path.exists(self.filename):
            os.remove(self.filename)

        best_collection = Collection(self.number_of_boxes, best_individual)
        return best_collection, best_fitness, history, fitness_log

    # run() is naive. run_by_region() uses per-region tournament selection and
    # box-level crossover informed by region performance — selecting the best-contributing
    # boxes from the best-performing Collections in each region, then assembling new
    # children from that pool.  Later we will compare the two approaches to see if
    # the added complexity is justified by improved results or convergence speed.
    def run_by_region(self):
        """Region-informed GA: per-region tournaments → box-level crossover from selection pool.

        Returns (best_collection, best_fitness, history, fitness_log).
        """
        # Initialize population
        population = [self._random_individual() for _ in range(self.population_size)]

        # Evaluate initial population — each returns (overall, region_dict, box_usage_dict)
        eval_results = [self._evaluate(ind) for ind in population]
        overall_fitnesses = [e[0] for e in eval_results]
        region_fitnesses = [e[1] for e in eval_results]
        box_usages = [e[2] for e in eval_results]

        # Track best (by overall fitness)
        best_idx = min(range(len(overall_fitnesses)), key=lambda i: overall_fitnesses[i])
        best_individual = copy.deepcopy(population[best_idx])
        best_fitness = overall_fitnesses[best_idx]
        history = [best_fitness]
        fitness_log = [list(overall_fitnesses)]

        print(f"Gen 0: best fitness = {best_fitness:.6f}", flush=True)

        for gen in range(1, self.generations + 1):
            # Elitism: carry forward top N by overall fitness
            elite_indices = sorted(range(len(overall_fitnesses)), key=lambda i: overall_fitnesses[i])[:self.elitism_count]
            new_population = [copy.deepcopy(population[i]) for i in elite_indices]

            # Random immigrants
            for _ in range(self.immigrant_count):
                if len(new_population) < self.population_size:
                    new_population.append(self._random_individual())

            # Build selection pool from per-region tournaments
            pool = self._build_selection_pool(population, region_fitnesses, box_usages)

            # Determine how many children come from pool vs traditional crossover
            remaining = self.population_size - len(new_population)
            n_pool = int(remaining * self.pool_ratio)
            n_crossover = remaining - n_pool

            # Pool-based children
            for _ in range(n_pool):
                child = self._construct_child_from_pool(pool, population)
                child = self._mutate(child)
                new_population.append(child)

            # Traditional crossover children (same as run())
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, overall_fitnesses)
                parent2 = self._tournament_select(population, overall_fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population
            eval_results = [self._evaluate(ind) for ind in population]
            overall_fitnesses = [e[0] for e in eval_results]
            region_fitnesses = [e[1] for e in eval_results]
            box_usages = [e[2] for e in eval_results]

            # Update best
            gen_best_idx = min(range(len(overall_fitnesses)), key=lambda i: overall_fitnesses[i])
            if overall_fitnesses[gen_best_idx] < best_fitness:
                best_individual = copy.deepcopy(population[gen_best_idx])
                best_fitness = overall_fitnesses[gen_best_idx]

            history.append(best_fitness)
            fitness_log.append(list(overall_fitnesses))

            if gen % 10 == 0 or gen == self.generations:
                print(f"Gen {gen}: best fitness = {best_fitness:.6f}", flush=True)

        # Clean up temp file
        if os.path.exists(self.filename):
            os.remove(self.filename)

        best_collection = Collection(self.number_of_boxes, best_individual)
        return best_collection, best_fitness, history, fitness_log