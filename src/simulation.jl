using DataFrames, Random, MLJ, Distributed, Statistics
using Base.Threads

#miaou
function add_noise(df::DataFrame, num_cols::Vector{Symbol}, σ::Float64)
    df_noisy = deepcopy(df)
    for c in num_cols
        df_noisy[!, c] .+= σ .* randn(nrow(df))
    end
    return df_noisy
end

function predict_proba(mach::Machine, X::DataFrame)
    yhat = MLJ.predict(mach, X)
    if yhat isa AbstractVector{<:UnivariateFinite}
        pos_class = last(levels(first(yhat)))
        return [pdf(y, pos_class) for y in yhat]
    else
        return Float64.(yhat)
    end
end

abstract type Simulation end

struct BasicSimulation <: Simulation
    model::Machine
    n_iter::Int
    noise_level::Float64
end

struct ThreadedSimulation <: Simulation
    model::Machine
    n_iter::Int
    noise_level::Float64
end

function simulate(sim::BasicSimulation, X::DataFrame)
    num_cols = Symbol.(names(X)[findall(c -> eltype(X[!, c]) <: Real, names(X))])
    preds = zeros(nrow(X))
    for _ in 1:sim.n_iter
        X_noisy = add_noise(X, num_cols, sim.noise_level)
        preds .+= predict_proba(sim.model, X_noisy)
    end
    return preds ./ sim.n_iter
end

function simulate(sim::ThreadedSimulation, X::DataFrame)
    nthreads_active = Threads.nthreads()
    if nthreads_active == 1
        @warn "1 seul thread actif détecté. Utilisation de BasicSimulation à la place."
        return simulate(BasicSimulation(sim.model, sim.n_iter, sim.noise_level), X)
    end

    @info "Running ThreadedSimulation on $nthreads_active threads..."

    num_cols = Symbol.(filter(c -> eltype(X[!, c]) <: Real, names(X)))

    preds_per_thread = [zeros(nrow(X)) for _ in 1:nthreads_active]

    Threads.@threads for rep in 1:sim.n_iter
        tid = mod1(threadid(), nthreads_active)
        X_noisy = add_noise(X, num_cols, sim.noise_level)
        preds = predict_proba(sim.model, X_noisy)
        preds_per_thread[tid] .+= preds
    end

    preds_total = reduce(+, preds_per_thread)
    return preds_total ./ sim.n_iter
end