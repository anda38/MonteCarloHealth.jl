using DataFrames, MLJ, MLJBase, MLJDecisionTreeInterface, MLJLinearModels, CategoricalArrays, Statistics

# === Charger et nettoyer les données ===
df = MonteCarloHealth.load_and_clean_data()

# Supprimer les colonnes non utiles
dropcols = [:patient_id, :blood_pressure]
df = DataFrames.select(df, Not(dropcols))

# Définir la variable cible
target = :readmitted_30_days

# Liste complète des variables explicatives
features = [
    :age, :cholesterol, :bmi,
    :diabetes, :hypertension, :medication_count,
    :length_of_stay,
    :blood_pressure_systolic, :blood_pressure_diastolic,
    :gender_Female, :gender_Male, :gender_Other,
    :discharge_Home, :discharge_Nursing_Facility, :discharge_Rehab
]

# Créer un DataFrame contenant uniquement les variables utiles
df_model = DataFrames.select(df, [features; target])

# Séparer la cible et les variables explicatives
y, X = unpack(df_model, ==(:readmitted_30_days), rng=123)
y = categorical(y)

# === Test des combinaisons de variables ===
results = Dict()

feature_sets = Dict(
    "all" => features,
    "clinical_only" => [:age, :diabetes, :hypertension, :length_of_stay],
    "no_blood_pressure" => [:age, :cholesterol, :bmi, :diabetes, :hypertension, :length_of_stay],
)

println("\n=== TEST DES COMBINAISONS DE VARIABLES ===")

for (label, feats) in feature_sets
    println("\n----- Jeu de variables : $label -----")
    X_subset = DataFrames.select(X, feats)
    model = RandomForestClassifier()  # modèle de base pour comparer les subsets

    mach = machine(model, X_subset, y)
    res = evaluate!(mach, resampling=CV(nfolds=5, shuffle=true), measure=accuracy, verbosity=0)

    results[label] = res.measurement[1]
    println("Précision moyenne : $(round(results[label], digits=3))")
end

# === Sélection du meilleur subset ===
best_score, best_key = findmax(results)
best_features = feature_sets[best_key]
println("\n✅ Meilleur subset : $best_key avec précision $(round(best_score, digits=3))")

# === Comparaison finale des modèles ===
X_best = DataFrames.select(X, best_features)

@load LogisticClassifier pkg=MLJLinearModels
@load DecisionTreeClassifier pkg=DecisionTree
@load RandomForestClassifier pkg=DecisionTree

models = Dict(
    "logistic" => LogisticClassifier(),
    "tree"     => DecisionTreeClassifier(),
    "forest"   => RandomForestClassifier()
)

println("\n=== COMPARAISON DES MODÈLES ===")

for (name, model) in models
    println("\n==============================")
    println("Test du modèle : $name")
    println("==============================")

    mach = machine(model, X_best, y)
    res = evaluate!(mach,
        resampling=CV(nfolds=5, shuffle=true),
        measure=accuracy,
        verbosity=0
    )

    println("Précision moyenne : $(round(res.measurement[1], digits=3))")
end

best_model = RandomForestClassifier()
mach_final = machine(best_model, X, y)
fit!(mach_final)
yhat = predict_mode(mach_final, X)
acc = mean(yhat .== y)
println("Accuracy (train set): ", round(acc, digits=3))