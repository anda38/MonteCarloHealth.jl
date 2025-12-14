module MonteCarloHealth

using CSV, DataFrames, Random, Statistics, MLJ, MLJBase
using MLJDecisionTreeInterface, MLJLinearModels, CategoricalArrays
using Base.Threads

include("preprocess_data.jl")
include("stat_des.jl")
include("simulation.jl")

export load_and_clean_data, description, correlation_table, train
export Simulation, BasicSimulation, ThreadedSimulation
export add_noise, predict_proba, simulate

end