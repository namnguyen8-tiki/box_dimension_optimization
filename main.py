from src.Module import *
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
SAMPLE_SIZE = 10000
NUMBER_OF_REGIONS_PER_ASPECT = 5  # quantile bins per stratification feature

# ── Sample — KS test thresholds ──
P_VALUE_THRESHOLD = 0.05
KS_STAT_THRESHOLD = 0.1

# ── Sample — valuation test ──
VALUATION_TEST_TOLERANCE = 0.1  # max relative difference between sample and full valuation

# ── Parallel processing ──
NUMBER_OF_BUCKETS = 7  # multiprocessing buckets for Valuation

# ── GA — core ──
POPULATION_SIZE = 10
GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2
IMMIGRANT_COUNT = 3
DIM_MIN = 4    # cm
DIM_MAX = 60   # cm
MUTATION_SIGMA = 3.0  # Gaussian perturbation std dev (cm)

# ── GA — region-informed (run_by_region only) ──
REGION_TOURNAMENT_SIZE = 20
REGION_TOURNAMENT_ROUNDS = 3
TOP_BOXES_PER_REGION = 3
POOL_RATIO = 0.3  # fraction of children from region pool vs. traditional crossover

NUMBER_OF_RUNS = 5  # how many times to repeat the GA process for robustness

# Process seperately for External and TikiCorp
if __name__ == '__main__':
    # data_order_external_1 = pd.read_csv('data/data_order_external_1.csv')
    # data_order_external_2 = pd.read_csv('data/data_order_external_2.csv')
    # # join the two external datasets for a comprehensive view
    # data_order_external = pd.concat([data_order_external_1, data_order_external_2], ignore_index=True)

    # data_order_external = pd.read_csv('data/data_order_external_test.csv')
    # print(f"External dataset shape: {data_order_external.shape}")

    # data_box_external = pd.read_csv('data/data_box_external.csv')

    # current_collection_external = Collection(
    #     number_of_boxes=data_box_external.nunique()['box_code'],
    #     boxes_dimensions=data_box_external[['length', 'width', 'height']].values.tolist()
    # )

    # sample = Sample(
    #     sample_size=SAMPLE_SIZE,
    #     number_of_regions_per_aspect=NUMBER_OF_REGIONS_PER_ASPECT,
    #     data_order=data_order_external
    # )

    # ks_test = sample.ks_test(P_VALUE_THRESHOLD, KS_STAT_THRESHOLD)
    # valuation_test = sample.valuation_test(filename= 'sample_temp.csv', tolerance=VALUATION_TEST_TOLERANCE, collection=current_collection_external,log_filename='external_valuation_test_log.csv')
    # print(ks_test)
    # print(valuation_test)
    # if ks_test == "KS Test Passed" and valuation_test == "Valuation Test Passed":
    #     sample.record_sample('sample_final.csv')
    # else:
    #     print("Sample did not pass tests. Skipping GA.")
    #     sys.exit(1)
    
    # sample_test là file tạm thời để Valuation lưu data vào
    # sample_final là file chính thức lưu data sample sau khi đã qua KS test và Valuation test, dùng để chạy GA

    # read the recorded sample and run the GA
    data_order = pd.read_csv('sample_final.csv')
    valuation_log = pd.read_csv("external_valuation_test_log.csv")
    f1_baseline = valuation_log[valuation_log['dataset'] == 'full']['objective_function_1'].values[0]
    f2_baseline = valuation_log[valuation_log['dataset'] == 'full']['objective_function_2'].values[0]
    f3_baseline = valuation_log[valuation_log['dataset'] == 'full']['objective_function_3'].values[0]

    for run_id in range(1, NUMBER_OF_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"  RUN {run_id}/{NUMBER_OF_RUNS}")
        print(f"{'='*60}\n")

        ga = GeneticAlgorithm(
            data_order=data_order,
            number_of_boxes=NUMBER_OF_BOXES,
            number_of_buckets=NUMBER_OF_BUCKETS,
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            tournament_size=TOURNAMENT_SIZE,
            elitism_count=ELITISM_COUNT,
            immigrant_count=IMMIGRANT_COUNT,
            region_tournament_size=REGION_TOURNAMENT_SIZE,
            region_tournament_rounds=REGION_TOURNAMENT_ROUNDS,
            top_boxes_per_region=TOP_BOXES_PER_REGION,
            pool_ratio=POOL_RATIO,
            dim_min=DIM_MIN,
            dim_max=DIM_MAX,
            mutation_sigma=MUTATION_SIGMA,
            valuation_kwargs={
                'objective_function_1_baseline': f1_baseline,
                'objective_function_2_baseline': f2_baseline,
                'objective_function_3_baseline': f3_baseline,
                'utilization_optimal': UTILIZATION_OPTIMAL,
                'reinforcement_thickness': REINFORCEMENT_THICKNESS,
                'bubble_thickness': BUBBLE_THICKNESS,
                'bubble_filling_rate': BUBBLE_FILLING_RATE,
            },
            results_kwargs={
                'w1': W1,
                'w2': W2,
                'w3': W3,
                'k': K,
            },
        )

        # ── Run naive GA ──
        best_collection, best_fitness, fitness_log = ga.run(run_id=run_id)
        print(f"\n[GA run() #{run_id}] Best fitness: {best_fitness:.6f}")
        print(f"[GA run() #{run_id}] Best box dimensions: {best_collection.boxes_dimensions}")

        # ── Run region-informed GA ──
        best_collection_r, best_fitness_r, fitness_log_r = ga.run_by_region(run_id=run_id)
        print(f"\n[GA run_by_region() #{run_id}] Best fitness: {best_fitness_r:.6f}")
        print(f"[GA run_by_region() #{run_id}] Best box dimensions: {best_collection_r.boxes_dimensions}")