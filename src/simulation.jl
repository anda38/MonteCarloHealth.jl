using DataFrames, Statistics, Random, MLJ, Distributed

# ----------------------------------------------------------
# 🧱 Base Simulation Types
# ----------------------------------------------------------

abstract type Simulation end

struct BasicSimulation <: Simulation
    model::Machine
    n_iter::Int
end

struct ParallelSimulation <: Simulation
    model::Machine
    n_iter::Int
end

# ----------------------------------------------------------
# ⚙️ Distributed Setup
# ----------------------------------------------------------

if nprocs() == 1
    addprocs(4)
end

@everywhere using MLJ, DataFrames, Random

# ----------------------------------------------------------
# 🧠 Serial Simulation
# ----------------------------------------------------------

function simulate(sim::BasicSimulation, data::DataFrame)
    preds = zeros(Float64, nrow(data))
    for _ in 1:sim.n_iter
        noisy_data = deepcopy(data)
        noisy_data.age .= noisy_data.age .+ randn(nrow(data)) .* 0.5
        ŷ = predict_mode(sim.model, noisy_data)
        preds .+= Float64.(ŷ)
    end
    preds ./= sim.n_iter
end

# ----------------------------------------------------------
# ⚙️ Parallel Simulation
# ----------------------------------------------------------

function simulate(sim::ParallelSimulation, data::DataFrame)
    n = nrow(data)
    results = pmap(1:sim.n_iter) do _
        noisy_data = deepcopy(data)
        noisy_data.age .= noisy_data.age .+ randn(n) .* 0.5
        Float64.(predict_mode(sim.model, noisy_data))
    end
    reduce(+, results) ./ sim.n_iter
end

# ----------------------------------------------------------
# 👩‍⚕️ Patient Representation
# ----------------------------------------------------------

struct PatientProfile
    age::Float64
    bmi::Float64
    hypertension::Bool
    diabetes::Bool
end

function to_df(p::PatientProfile)
    DataFrame(age=[p.age],
              bmi=[p.bmi],
              hypertension=[p.hypertension ? 1 : 0],
              diabetes=[p.diabetes ? 1 : 0])
end
