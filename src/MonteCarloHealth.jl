module MonteCarloHealth

using CSV, DataFrames, Random, Statistics, MLJ, MLJBase
using MLJDecisionTreeInterface, MLJLinearModels, CategoricalArrays
using Base.Threads

include("preprocess_data.jl")
include("simulation.jl")

export load_and_clean_data
export Simulation, BasicSimulation, ThreadedSimulation
export add_noise, predict_proba, simulate

end