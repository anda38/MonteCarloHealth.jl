module MonteCarloHealth

using MLJ, CSV, DataFrames, Random, Distributed, Statistics, MLJDecisionTreeInterface, Base.Threads
using MLJBase, MLJDecisionTreeInterface, MLJLinearModels, CategoricalArrays


export load_and_clean_data, basic_stats, simulate, plot_simulation_results, compare_simulations, 
BasicSimulation, ParallelSimulation, Simulation, add_noise, predict_proba, _simulate_iteration, add_noise, ThreadedSimulation
include("preprocess_data.jl")
include("stat_des.jl")
include("simulation.jl")

end