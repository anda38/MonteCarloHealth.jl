using MonteCarloHealth
using MLJ, DataFrames, CategoricalArrays, Statistics, StatsPlots, Random, MLJDecisionTreeInterface


# Charge le jeu de données nettoyé et encodé
df = load_and_clean_data()

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

# Sépare 20 % pour l'évaluation
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)

model = RandomForestClassifier()
# Entraîne uniquement sur l'échantillon d'apprentissage
mach = machine(model, X[train, :], y[train]) |> fit!

# Paramètre les simulations : basique vs multithread
basic_sim    = BasicSimulation(mach, 500, 0.1)
threaded_sim = ThreadedSimulation(mach, 5000, 0.1)

@info "ThreadedSimulation 5000 iter..."
@time preds_threaded = simulate(threaded_sim, X[test, :])

@info "BasicSimulation 500 iter..."
@time preds_basic = simulate(basic_sim, X[test, :])

println()

# Compare les statistiques clés entre les approches
println("Résumé")
println("──────────────────────────────────────────────")

println("Moyenne (Séquentiel):       ", mean(preds_basic))

println("Moyenne (Threaded):    ", mean(preds_threaded))
println("Ecart-type (Séquentiel):        ", std(preds_basic))
println("Ecart-type (Threaded):     ", std(preds_threaded))


println("──────────────────────────────────────────────")



# Visualise la distribution des probabilités prédites
p1 = histogram(preds_basic, bins=30, title="Simulation séquentielle", xlabel="Prédicition", alpha=0.7, legend=false)

p2 = histogram(preds_threaded, bins=30, title="Simulation parallèle", xlabel="Prédiction", alpha=0.7, legend=false)
StatsPlots.plot(p1, p2, layout=(1, 2)) 
