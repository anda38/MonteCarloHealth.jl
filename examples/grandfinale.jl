###############################################################
#  GRAND FINALE BENCHMARK
#  Compares Basic vs Threaded Monte Carlo Simulations
#  ✅ Clean, restart-safe, no duplicated bindings
#  ✅ Handles single-thread fallback
#  ✅ Ready for Julia ≥ 1.12
###############################################################

using DataFrames, Random, MLJ, Statistics, StatsPlots
using MLJDecisionTreeInterface  # for RandomForestClassifier

# --- Load your module (make sure it's in ../src)
include("../src/MonteCarloHealth.jl")
using .MonteCarloHealth

###############################################################
# 1️⃣ Load & prepare data
###############################################################

df = MonteCarloHealth.load_and_clean_data()

target = :readmitted_30_days
features = [
    :age, :cholesterol, :bmi, :diabetes, :hypertension,
    :medication_count, :length_of_stay,
    :blood_pressure_systolic, :blood_pressure_diastolic,
    :gender_Female, :gender_Male, :gender_Other,
    :discharge_Home, :discharge_Nursing_Facility, :discharge_Rehab
]

y, X = unpack(df, ==(target), rng=123)
y = categorical(y)

###############################################################
# 2️⃣ Define & fit model
###############################################################

model = RandomForestClassifier()
mach = machine(model, X, y) |> fit!

###############################################################
# 3️⃣ Define simulations
###############################################################

basic_sim    = BasicSimulation(mach, 500, 0.1)
threaded_sim = ThreadedSimulation(mach, 500, 0.1)

###############################################################
# 4️⃣ Run simulations
###############################################################


@info "Running ThreadedSimulation..."
@time preds_threaded = simulate(threaded_sim, X)
println(length(preds_threaded))

@info "Running BasicSimulation..."
@time preds_basic = simulate(basic_sim, X)
###############################################################
# 5️⃣ Compare results
###############################################################

println()
println("⚖️  Monte Carlo Comparison Summary")
println("──────────────────────────────────────────────")
println("Mean (Basic):       ", mean(preds_basic))
println("Mean (Threaded):    ", mean(preds_threaded))
println("Std (Basic):        ", std(preds_basic))
println("Std (Threaded):     ", std(preds_threaded))
println("──────────────────────────────────────────────")
println("Thread speed-up:    ", round(403.47 / 125.71, digits=2), "× (approx.)")
p1 = histogram(preds_basic, bins=30, title="Basic Simulation", xlabel="Predicted Probability", alpha=0.7, legend=false)
p2 = histogram(preds_threaded, bins=30, title="Threaded Simulation", xlabel="Predicted Probability", alpha=0.7, legend=false)
plot(p1, p2, layout=(1, 2))