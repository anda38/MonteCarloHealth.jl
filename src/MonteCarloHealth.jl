module MonteCarloHealth

using CSV, DataFrames, CategoricalArrays

include("preprocess_data.jl")
using .PreprocessData: load_and_clean_data  

export load_and_clean_data

end
