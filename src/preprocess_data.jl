module PreprocessData

using CSV, DataFrames, CategoricalArrays

function load_and_clean_data()
    chemin = joinpath(@__DIR__, "..", "data", "données.csv")

    df = CSV.read(chemin, DataFrame)

    # Séparer la pression artérielle
    df.blood_pressure_systolic = parse.(Int, first.(split.(df.blood_pressure, "/")))
    df.blood_pressure_diastolic = parse.(Int, last.(split.(df.blood_pressure, "/")))

    # Colonnes booléennes
    for col in [:diabetes, :hypertension, :readmitted_30_days]
        df[!, col] = ifelse.(df[!, col] .== "Yes", 1, 0)
    end

    # Colonnes catégorielles
    for col in [:gender, :discharge_destination]
        df[!, col] = categorical(df[!, col])
    end

    # Sauvegarde du CSV nettoyé
    cheminpropre = joinpath(@__DIR__, "..", "data", "données_clean.csv")
    CSV.write(cheminpropre, df)

    println("✅ données sauvegardées: $cheminpropre")
    return df
end

end # module
