module MonteCarloHealth

using MLJ, DataFrames, Random, Distributed, Statistics, MLJDecisionTreeInterface

include("preprocess_data.jl")
include("stat_des.jl")
include("simulation.jl")
include("visualisation.jl")
export load_and_clean_data, basic_stats, simulate, plot_simulation_results, compare_simulations, 
BasicSimulation, ParallelSimulation, Simulation,add_noise, predict_proba, _simulate_iteration, add_noise, ThreadedSimulation
end