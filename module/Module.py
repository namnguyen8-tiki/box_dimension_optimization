# pylint: disable=C0103
# pylint: disable=E0102

import sys
import math
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
            utilization_optimal: float = 0.9,
            reinforcement_thickness: float = 0.5,
            bubble_thickness: float = 1.0,
            bubble_filling_rate: float = 0.5,
            objective_function_1_baseline: float = 1000000.0,  # baseline cost for normalization
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
        for p in processes:
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()

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

        for order in unique_orders:
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

        for region, n_total in orders_by_region.items():
            n_unfittable = objective_function_3_by_region.get(region, 0)
            n_fittable = n_total - n_unfittable
            self.objective_function_1_by_region[region] = objective_function_1_by_region.get(region, 0) / n_fittable if n_fittable > 0 else float('inf')
            self.objective_function_2_by_region[region] = objective_function_2_by_region.get(region, 0) / n_fittable if n_fittable > 0 else float('inf')
            self.objective_function_3_by_region[region] = n_unfittable / n_total

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
        
        # Build per-order summary for stratification
        order_summary = temp_data_order.groupby('order_code').apply(
            lambda df: pd.Series({
                'total_volume': (df['length'] * df['width'] * df['height'] * df['unit']).sum(),
                'max_length': df[['length', 'width', 'height']].max().max(),
                'total_unit': df['unit'].sum()
            })
        ).reset_index()

        # Create bins for stratification (use duplicates='drop' to handle non-unique values)
        order_summary['volume_bin'] = pd.qcut(order_summary['total_volume'], q=number_of_buckets_per_aspect, labels=False, duplicates='drop')
        order_summary['length_bin'] = pd.qcut(order_summary['max_length'], q=number_of_buckets_per_aspect, labels=False, duplicates='drop')
        order_summary['unit_bin'] = pd.qcut(order_summary['total_unit'], q=number_of_buckets_per_aspect, labels=False, duplicates='drop')

        # Stratified proportional sampling at ORDER level (not row level)
        sample_frac = sample_size / len(order_summary)
        
        sampled_orders = order_summary.groupby(['volume_bin', 'length_bin', 'unit_bin']).apply(
            lambda x: x.sample(max(1, int(len(x) * sample_frac)), replace=False)
                if len(x) > 1 else x
        ).reset_index(drop=True)

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
            return "Passed"
        else:
            return "Failed"
        
    def valuation_test(self, collection: Collection, filename: str, w1: float = 0.6, w2: float = 0.35, w3: float = 0.05, tolerance: float = 0.1):
        # This function can be implemented to calculate the valuation metrics on the sample and compare them with the valuation metrics calculated on the whole dataset using the same Collection and filename for results.
        valuation_sample = Valuation(self.sample, number_of_buckets=5, collection=collection, filename=filename)
        valuation_full = Valuation(self.data_order, number_of_buckets=5, collection=collection, filename=filename)

        print(f"Valuation results for sample: {valuation_sample.results(w1, w2, w3)}")
        print(f"Valuation results for full dataset: {valuation_full.results(w1, w2, w3)}")
        print(f"Unfittable ratio — sample: {valuation_sample.objective_function_3:.4f}, full: {valuation_full.objective_function_3:.4f}")

        # return "Passed" if the valuation results are within 10% of each other, otherwise return "Failed"
        if abs(valuation_sample.results(w1, w2, w3) - valuation_full.results(w1, w2, w3)) / valuation_full.results(w1, w2, w3) < tolerance:
            return "Passed"
        else:
            return "Failed"
    
class GeneticAlgorithm:
    def __init__(
            self,
            valuation: Valuation,
            population_size: int = 50,
            generations: int = 100,
            mutation_rate: float = 0.01
    ) -> None:
        self.valuation = valuation
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def run(self):
        # Implement the genetic algorithm to optimize the box assignment and bubble usage based on the valuation metrics
        