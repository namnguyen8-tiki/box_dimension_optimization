from module.Module import *
# pyright: ignore[reportUndefinedVariable]

# ── Collection ──
NUMBER_OF_BOXES = 10  # how many box types in a Collection

# ── Valuation — physical constants ──
UTILIZATION_OPTIMAL = 0.8  # ideal utilization ratio (80%)
REINFORCEMENT_THICKNESS = 0.5  # cm per side
BUBBLE_THICKNESS = 1            # cm
BUBBLE_FILLING_RATE = 0.5

# ── Valuation — baselines for normalization (calibrate from current Collection) ──
OBJECTIVE_FUNCTION_1_BASELINE = 5000       # cost baseline
OBJECTIVE_FUNCTION_2_BASELINE = 0.1        # utilization deviation baseline
OBJECTIVE_FUNCTION_3_BASELINE = 0.05       # unfittable ratio baseline

# ── Valuation — objective function weights & penalty ──
W1 = 0.60   # cost weight
W2 = 0.35   # utilization weight
W3 = 0.05   # unfittable ratio weight
K  = 3.0    # exponential penalty sensitivity for f3

# ── Sample — stratified sampling ──
SAMPLE_SIZE = 100000
NUMBER_OF_BUCKETS_PER_ASPECT = 5  # quantile bins per stratification feature

# ── Sample — KS test thresholds ──
P_VALUE_THRESHOLD = 0.05
KS_STAT_THRESHOLD = 0.1

# ── Sample — valuation test ──
VALUATION_TEST_TOLERANCE = 0.1  # max relative difference between sample and full valuation

# ── Parallel processing ──
NUMBER_OF_BUCKETS = 5  # multiprocessing buckets for Valuation

# ── GA — core ──
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2
IMMIGRANT_COUNT = 3
DIM_MIN = 5    # cm
DIM_MAX = 60   # cm
MUTATION_SIGMA = 3.0  # Gaussian perturbation std dev (cm)

# ── GA — region-informed (run_by_region only) ──
REGION_TOURNAMENT_SIZE = 20
REGION_TOURNAMENT_ROUNDS = 3
TOP_BOXES_PER_REGION = 3
POOL_RATIO = 0.5  # fraction of children from region pool vs. traditional crossover

# Process seperately for External and TikiCorp
if __name__ == '__main__':
    data_order_external_1 = pd.read_csv('data/data_order_external_1.csv')
    data_order_external_2 = pd.read_csv('data/data_order_external_2.csv')
    # join the two external datasets for a comprehensive view
    data_order_external = pd.concat([data_order_external_1, data_order_external_2], ignore_index=True)
    print(f"External dataset shape: {data_order_external.shape}")

    data_box_external = pd.read_csv('data/data_box_external.csv')

    current_collection_external = Collection(
        number_of_boxes=data_box_external.nunique()['box_code'],
        boxes_dimensions=data_box_external[['length', 'width', 'height']].values.tolist()
    )

    sample = Sample(
        sample_size=SAMPLE_SIZE,
        number_of_buckets_per_aspect=NUMBER_OF_BUCKETS_PER_ASPECT,
        data_order=data_order_external
    )

    print(sample.ks_test(P_VALUE_THRESHOLD, KS_STAT_THRESHOLD))
    print(sample.valuation_test(filename= 'sample_test.csv', tolerance=VALUATION_TEST_TOLERANCE, collection=current_collection_external))