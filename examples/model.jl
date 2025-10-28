using Distributed

# === 0️⃣ WORKER SETUP ===
# add worker processes if we're running single-threaded
if nprocs() == 1
    addprocs(4)
end

# load the necessary packages on every worker
@everywhere using MLJ, MLJBase, MLJDecisionTreeInterface, MLJLinearModels, DataFrames, Random, Statistics, CategoricalArrays
# === 1️⃣ LOAD & PREPROCESS ===
df = MonteCarloHealth.load_and_clean_data()

# Drop irrelevant columns
dropcols = [:patient_id, :blood_pressure]
df = DataFrames.select(df, Not(dropcols))

# Target and features
target = :readmitted_30_days

features = [
    :age, :cholesterol, :bmi,
    :diabetes, :hypertension, :medication_count,
    :length_of_stay,
    :blood_pressure_systolic, :blood_pressure_diastolic,
    :gender_Female, :gender_Male, :gender_Other,
    :discharge_Home, :discharge_Nursing_Facility, :discharge_Rehab
]

# Data subset for modeling
df_model = DataFrames.select(df, [features; target])

# Split X, y
y, X = unpack(df_model, ==(:readmitted_30_days), rng=123)
y = categorical(y)

# === 2️⃣ FEATURE SUBSET TEST (RandomForest only) ===
feature_sets = Dict(
    "all" => features,
    "clinical_only" => [:age, :diabetes, :hypertension, :length_of_stay],
    "no_blood_pressure" => [:age, :cholesterol, :bmi, :diabetes, :hypertension, :length_of_stay],
)

subset_results = DataFrame(SetName=String[], Accuracy=Float64[])

println("\n=== PHASE 1: FEATURE SUBSET TEST ===")

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
println("\n✅ Best subset: $(best_row.SetName) with accuracy = $(round(best_row.Accuracy, digits=3))")

# === 3️⃣ MODEL COMPARISON (on best subset only) ===

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
println("\n=== PHASE 2: MODEL COMPARISON ===")

for (name, model) in models
    mach = machine(model, X_best, y)
    res = evaluate!(mach, resampling=CV(nfolds=5, shuffle=true), measure=accuracy, verbosity=0)
    acc = res.measurement[1]
    push!(model_results, (name, acc))
    println("Model: $(name) → accuracy = $(round(acc, digits=3))")
end

best_model_row = model_results[argmax(model_results.Accuracy), :]
println("\n🏆 Best model: $(best_model_row.Model) with accuracy = $(round(best_model_row.Accuracy, digits=3))")

# === 4️⃣ SUMMARY TABLES ===

println("\n📊 Feature Subset Results:")
println(subset_results)

println("\n🤖 Model Comparison Results:")
println(model_results)
