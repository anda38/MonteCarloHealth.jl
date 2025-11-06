using Statistics, DataFrames

function description(df::DataFrame)
    println("Dimensions : $(nrow(df)) lignes × $(ncol(df)) colonnes")
    println("----------------------------------------------------")

    println(" Types de colonnes :")
    for (name, col) in zip(names(df), eachcol(df))
        println("• $(name): $(eltype(col))")
    end

    println(" Valeurs manquantes :")
    for (name, col) in zip(names(df), eachcol(df))
        nmiss = count(ismissing, col)
        if nmiss > 0
            println("• $(name): $(nmiss)")
        end
    end

    println("Statistiques descriptives (variables numériques) :")
    num_cols = [c for c in names(df) if eltype(df[!, c]) <: Number]
    if isempty(num_cols)
        println("Aucune variable numérique trouvée.")
        return
    end

    for c in num_cols
        x = skipmissing(df[!, c])
        println("\n→ $(c):")
        println("   Moyenne = $(round(mean(x), digits=2))")
        println("   Médiane = $(round(median(x), digits=2))")
        println("   Écart-type = $(round(std(x), digits=2))")
        println("   Min = $(minimum(x)), Max = $(maximum(x))")
    end

    return nothing
end


function correlation_table(df::DataFrame)
    num = select(df, names(df, Number))
    if ncol(num) > 1
        println("\n🔗 Matrice de corrélation numérique :")
        corr_mat = cor(Matrix(num))
        corr_df = DataFrame(corr_mat, Symbol.(names(num)))
        println(corr_df)
    else
        println("Pas assez de variables numériques pour une matrice de corrélation.")
    end
end