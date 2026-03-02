# pylint: disable=C0103
# pylint: disable=E0102

import sys
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
        bubble_thickness,
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
            objective_function_2_baseline: float = 0.1  # baseline utilization deviation for normalization
    ) -> None:
        self.objective_function_1_baseline = objective_function_1_baseline
        self.objective_function_2_baseline = objective_function_2_baseline

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
        n_fittable = 0
        n_unfittable = 0

        for order in unique_orders:
            df_order_temp = data_order_mod[data_order_mod['order_code'] == order]
            item_volume = (df_order_temp['length'] * df_order_temp['width'] * df_order_temp['height'] * df_order_temp['unit']).sum()

            df_result_temp = df_result[df_result['order'] == order]

            # Check if order is unfittable (no result row, or box_final is a known unfittable marker)
            if len(df_result_temp) == 0 or df_result_temp['box_final'].iloc[0] in unfittable_markers:
                n_unfittable += 1
                continue

            # Fittable order — look up the chosen box by name
            chosen_box_name = df_result_temp['box_final'].iloc[0]
            df_box_temp = df_box[df_box['box'] == chosen_box_name]

            if len(df_box_temp) == 0:
                # box name not found in collection (shouldn't happen, but guard against it)
                n_unfittable += 1
                continue

            chosen_box_volume = df_box_temp['volume'].iloc[0]
            chosen_box_cost = df_box_temp['cost'].iloc[0]

            filling_bubble_usage_volume = (chosen_box_volume - item_volume) * bubble_filling_rate
            bubble_usage = Bubble(bubble_thickness, filling_bubble_usage_volume)

            # Objective function 1: minimize total cost (box cost + bubble cost)
            objective_function_1 += chosen_box_cost + bubble_usage.bubble_cost
            # Objective function 2: optimize total utilization (item volume / box volume)
            objective_function_2 += (item_volume / chosen_box_volume - utilization_optimal)**2
            n_fittable += 1

        # Average over fittable orders only (avoid division by zero)
        self.objective_function_1 = objective_function_1 / n_fittable if n_fittable > 0 else float('inf')
        self.objective_function_2 = objective_function_2 / n_fittable if n_fittable > 0 else float('inf')

        # Unfittable ratio: measures Collection complexity/sparsity
        self.r_unfittable = n_unfittable / len(unique_orders)
        self.n_fittable = n_fittable
        self.n_unfittable = n_unfittable

    def results(self, w1: float = 0.6, w2: float = 0.35, w3: float = 0.05):
        """Combined objective function with unfittable penalty.
        
        F = w1 * (f1/b1) + w2 * (f2/b2) + w3 * r_unfittable
        
        - w1 + w2 + w3 = 1.0 (w1=0.6 cost, w2=0.35 utilization, w3=0.05 unfittable penalty)
        - f1/f2 are computed only over fittable orders (clean separation)
        - r_unfittable is naturally in [0, 1], no baseline needed
        - w3 reflects that unfittable orders are expected but indicate Collection complexity/sparsity
        """
        return (w1 * (self.objective_function_1 / self.objective_function_1_baseline) 
                + w2 * (self.objective_function_2 / self.objective_function_2_baseline) 
                + w3 * self.r_unfittable)

    def results_by_region(self, number_of_regions: int):
        # This function can be implemented to calculate the valuation metrics by region if needed
        

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
        print(f"Unfittable ratio — sample: {valuation_sample.r_unfittable:.4f}, full: {valuation_full.r_unfittable:.4f}")

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
        