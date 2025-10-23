module MonteCarloHealth

using MLJ, DataFrames, Random, Distributed, Statistics

include("preprocess_data.jl")
include("stat_des.jl")
include("simulation.jl")

export train_model, BasicSimulation, ParallelSimulation, simulate

end