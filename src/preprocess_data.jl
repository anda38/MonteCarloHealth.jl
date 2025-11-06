using CSV, DataFrames

function load_and_clean_data()
    
    path = joinpath(@__DIR__, "..", "data", "données.csv")
    df = CSV.read(path, DataFrame; normalizenames=true)

    
    parts = split.(df.blood_pressure, "/")
    df.blood_pressure_systolic  = parse.(Int, getindex.(parts, 1))
    df.blood_pressure_diastolic = parse.(Int, getindex.(parts, 2))

    
    select!(df, Not([:patient_id, :blood_pressure]))

    
    df.diabetes     = ifelse.(df.diabetes .== "Yes", 1, 0)
    df.hypertension = ifelse.(df.hypertension .== "Yes", 1, 0)
    df.readmitted_30_days = ifelse.(df.readmitted_30_days .== "Yes", 1, 0)

    
    for g in unique(df.gender)
        df[!, Symbol("gender_", g)] = ifelse.(df.gender .== g, 1, 0)
    end
    select!(df, Not(:gender))


    for d in unique(df.discharge_destination)
        df[!, Symbol("discharge_", d)] = ifelse.(df.discharge_destination .== d, 1, 0)
    end
    select!(df, Not(:discharge_destination))

    println("Données propres: $(nrow(df)) lignes × $(ncol(df)) colonnes")
    return df
end
