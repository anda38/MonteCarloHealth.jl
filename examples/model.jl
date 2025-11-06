using MLJ, MLJBase, MLJDecisionTreeInterface, MLJLinearModels, DataFrames, Random, Statistics, CategoricalArrays

df = MonteCarloHealth.load_and_clean_data()

dropcols = [:patient_id, :blood_pressure]
df = DataFrames.select(df, Not(dropcols))

target = :readmitted_30_days

features = [
    :age, :cholesterol, :bmi,
    :diabetes, :hypertension, :medication_count,
    :length_of_stay,
    :blood_pressure_systolic, :blood_pressure_diastolic,
    :gender_Female, :gender_Male, :gender_Other,
    :discharge_Home, :discharge_Nursing_Facility, :discharge_Rehab
]

df_model = DataFrames.select(df, [features; target])

y, X = unpack(df_model, ==(:readmitted_30_days), rng=123)
y = categorical(y)

feature_sets = Dict(
    "tout" => features,
    "cliniques" => [:age, :diabetes, :hypertension, :length_of_stay],
    "pas_blood_pressure" => [:age, :cholesterol, :bmi, :diabetes, :hypertension, :length_of_stay],
)

subset_results = DataFrame(SetName=String[], Accuracy=Float64[])

println("Test subsets")

for (label, feats) in feature_sets
    X_subset = DataFrames.select(X, feats)
    model = RandomForestClassifier()
    mach = machine(model, X_subset, y)
    res = evaluate!(mach, resampling=CV(nfolds=5, shuffle=true), measure=accuracy, verbosity=0)
    acc = res.measurement[1]
    push!(subset_results, (label, acc))
    println("Subset: $(label) → accuracy = $(round(acc, digits=3))")
end

best_row = subset_results[argmax(subset_results.Accuracy), :]
best_features = feature_sets[best_row.SetName]
println("Meilleur subset: $(best_row.SetName) avec fiabilité = $(round(best_row.Accuracy, digits=3))")


@load LogisticClassifier pkg=MLJLinearModels
@load DecisionTreeClassifier pkg=DecisionTree
@load RandomForestClassifier pkg=DecisionTree

models = Dict(
    "Logistic Regression" => LogisticClassifier(),
    "Decision Tree"       => DecisionTreeClassifier(),
    "Random Forest"       => RandomForestClassifier()
)

model_results = DataFrame(Model=String[], Accuracy=Float64[])

X_best = DataFrames.select(X, best_features)
println("Comparaison modèles")

for (name, model) in models
    mach = machine(model, X_best, y)
    res = evaluate!(mach, resampling=CV(nfolds=5, shuffle=true), measure=accuracy, verbosity=0)
    acc = res.measurement[1]
    push!(model_results, (name, acc))
    println("Modèle: $(name) → fiabilité = $(round(acc, digits=3))")
end

best_model_row = model_results[argmax(model_results.Accuracy), :]
println("Meilleur modèle: $(best_model_row.Model) avec fiabilitié = $(round(best_model_row.Accuracy, digits=3))")
