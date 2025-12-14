function train(df::AbstractDataFrame)
    target = :readmitted_30_days
    features = [
    :age, :cholesterol, :bmi, :diabetes, :hypertension,
    :medication_count, :length_of_stay,
    :blood_pressure_systolic, :blood_pressure_diastolic,
    :gender_Female, :gender_Male, :gender_Other,
    :discharge_Home, :discharge_Nursing_Facility, :discharge_Rehab
]
    # MLJ unpack
    y, X = MLJ.unpack(df, ==(target), rng=123)
    y = categorical(y)

    # MLJ model
    model = MLJDecisionTreeInterface.RandomForestClassifier()

    mach = machine(model, X, y)
    fit!(mach)

    return mach
end
