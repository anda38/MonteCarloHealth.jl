using DataFrames, Random, MLJ, Statistics, StatsPlots
using MLJDecisionTreeInterface
#Miaou
include("../src/MonteCarloHealth.jl")
using .MonteCarloHealth

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

model = RandomForestClassifier()
mach = machine(model, X, y) |> fit!

basic_sim    = BasicSimulation(mach, 500, 0.1)
threaded_sim = ThreadedSimulation(mach, 500, 0.1)

@info "Running ThreadedSimulation..."
@time preds_threaded = simulate(threaded_sim, X)

@info "Running BasicSimulation..."
@time preds_basic = simulate(basic_sim, X)

println()
println("⚖️  Monte Carlo Sommaire")
println("──────────────────────────────────────────────")
println("Moyenne (Basic):       ", mean(preds_basic))
println("Moyenne (Threaded):    ", mean(preds_threaded))
println("Std (Basic):        ", std(preds_basic))
println("Std (Threaded):     ", std(preds_threaded))
println("──────────────────────────────────────────────")
println("Thread différence:    ", round(403.47 / 125.71, digits=2), "× (approx.)")
p1 = histogram(preds_basic, bins=30, title="Basic Simulation", xlabel="Predicted Probability", alpha=0.7, legend=false)
p2 = histogram(preds_threaded, bins=30, title="Threaded Simulation", xlabel="Predicted Probability", alpha=0.7, legend=false)
plot(p1, p2, layout=(1, 2)) 