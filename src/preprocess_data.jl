using CSV, DataFrames, Statistics

function load_and_clean_data()
    # Charger le fichier brut
    chemin = joinpath(@__DIR__, "..", "data", "données.csv")
    df = CSV.read(chemin, DataFrame)

    # Séparer la pression artérielle en deux colonnes numériques
    df.blood_pressure_systolic = parse.(Int, first.(split.(df.blood_pressure, "/")))
    df.blood_pressure_diastolic = parse.(Int, last.(split.(df.blood_pressure, "/")))

    # Convertir les colonnes Yes/No en 1/0
    for col in [:diabetes, :hypertension, :readmitted_30_days]
        df[!, col] = ifelse.(df[!, col] .== "Yes", 1, 0)
    end 

    # Encodage one-hot manuel pour la variable "gender"
    genders = unique(df.gender)
    for g in genders
        df[!, Symbol("gender_", g)] = ifelse.(df.gender .== g, 1, 0)
    end
    # Encodage one-hot manuel pour la variable "discharge_destination"
    discharges = unique(df.discharge_destination)
    for d in discharges
        df[!, Symbol("discharge_", d)] = ifelse.(df.discharge_destination .== d, 1, 0)
    end

    # Supprimer les colonnes d’origine catégorielles
    select!(df, Not([:gender, :discharge_destination]))

    # Sauvegarder le fichier nettoyé
    cheminpropre = joinpath(@__DIR__, "..", "data", "données_clean.csv")
    CSV.write(cheminpropre, df)
    println("✅ Données nettoyées et encodées sauvegardées : $cheminpropre")

    return df
end
