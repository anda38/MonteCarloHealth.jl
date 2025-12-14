using MonteCarloHealth
using DataFrames, MLJ, MLJDecisionTreeInterface, CategoricalArrays
using Statistics
using WGLMakie, BonitoBook
using BonitoBook.Bonito
using ColorSchemes, KernelDensity


df = load_and_clean_data()
target = :readmitted_30_days
y, X = unpack(df, ==(target), rng=123)
y = categorical(y)

model = RandomForestClassifier()
mach = machine(model, X, y) |> fit!

noise        = Observable(0.05)
n_iter       = Observable(100)
threaded     = Observable(true)
subset_size  = Observable(500)
preds        = Observable(Float64[])
preds_matrix = Observable(Matrix{Float64}(undef, 0, 0))


function run_all(; n_iter, noise, threaded, subset_size=500)
    Xsub = X[rand(1:nrow(X), min(subset_size, nrow(X))), :]
    num_cols = Symbol.(filter(c -> eltype(Xsub[!, c]) <: Real, names(Xsub)))
    n_rows = nrow(Xsub)
    preds_mat = zeros(n_iter, n_rows)

    if threaded
        @info "Simulation multi-thread sur $(Threads.nthreads()) threads..."
        Threads.@threads for i in 1:n_iter
            X_noisy = MonteCarloHealth.add_noise(Xsub, num_cols, noise)
            preds_mat[i, :] = MonteCarloHealth.predict_proba(mach, X_noisy)
        end
    else
        for i in 1:n_iter
            X_noisy = MonteCarloHealth.add_noise(Xsub, num_cols, noise)
            preds_mat[i, :] = MonteCarloHealth.predict_proba(mach, X_noisy)
        end
    end
    preds_mean = mean(preds_mat, dims=1)[:]
    return preds_mean, preds_mat
end


fig = Figure(size=(900, 400))
ax = Axis(fig[1, 1], title="Distribution des itérations Monte Carlo",
    xlabel="Probabilité prédite (p(réadmission))", ylabel="Densité")

import KernelDensity: kde

function redraw_curves!(M)
    empty!(ax)
    n_iter = size(M, 1)
    cs = ColorSchemes.viridis.colors
    colors = [cs[round(Int, i * (length(cs)-1) / n_iter) + 1] for i in 0:n_iter-1]

    kdes = [KernelDensity.kde(M[i, :]) for i in 1:n_iter]

    for (i, c) in enumerate(colors)
        lines!(ax, kdes[i].x, kdes[i].density, color=(c, 0.35), linewidth=2.0)
    end

    ax.title = "Distribution des itérations Monte Carlo"
end


bouton_run      = Components.Button("Lancer la simulation")
curseur_bruit   = Components.Slider(0.0:0.01:0.5; value=noise[])
curseur_iter    = Components.Slider(5:5:500; value=n_iter[])
curseur_subset  = Components.Slider(100:100:5000; value=subset_size[])
case_thread     = Components.Checkbox(threaded[])

on(curseur_bruit.value)  do v; noise[] = v; end
on(curseur_iter.value)   do v; n_iter[] = Int(v); end
on(curseur_subset.value) do v; subset_size[] = Int(v); end
on(case_thread.value)    do v; threaded[] = v; end

on(bouton_run.value) do _
    @info "Recalcul..." itérations=n_iter[] bruit=noise[] threads=threaded[] sous_échantillon=subset_size[]
    p_mean, p_mat = run_all(n_iter=n_iter[], noise=noise[], threaded=threaded[], subset_size=subset_size[])
    preds[] = p_mean
    preds_matrix[] = p_mat
    redraw_curves!(p_mat)
    @info "Simulation terminée"
end


stats_df = lift(preds) do p
    if isempty(p)
        return DataFrame(mesure=["Moyenne","Écart-type","Minimum","Maximum"], valeur=[NaN,NaN,NaN,NaN])
    end
    DataFrame(
        mesure=["Moyenne", "Écart-type", "Minimum", "Maximum"],
        valeur=[mean(p), std(p), minimum(p), maximum(p)]
    )
end

stats_html = lift(stats_df) do df
    DOM.table(
        DOM.tr(
            DOM.th("Mesure", style=Styles("padding" => "12px 20px", "border-bottom" => "2px solid #B4E0C9")),
            DOM.th("Valeur", style=Styles("padding" => "12px 20px", "border-bottom" => "2px solid #B4E0C9"))
        ),
        [DOM.tr(
            DOM.td(string(row.mesure), style=Styles("padding" => "10px 20px", "font-weight" => "500", "color" => "#333")),
            DOM.td(round(row.valeur, digits=4), style=Styles("padding" => "10px 20px", "text-align" => "right", "color" => "#2F4F4F"))
        ) for row in eachrow(df)]...,
        style=Styles(
            "width" => "100%",
            "border-collapse" => "collapse",
            "border-radius" => "12px",
            "overflow" => "hidden",
            "font-size" => "16px",
            "background" => "#FFFFFF",
            "box-shadow" => "0 4px 10px rgba(0,0,0,0.05)",
            "margin-top" => "10px"
        )
    )
end

app_style = Styles(
    "font-family" => "'Inter', system-ui, sans-serif",
    "font-weight" => "500",
    "background-color" => "#FBE8ED", 
    "padding" => "40px",
    "max-width" => "950px",
    "margin" => "auto"
)

carte_style = Styles(
    "background" => "rgba(255,255,255,0.75)",
    "backdrop-filter" => "blur(14px)",
    "border-radius" => "18px",
    "padding" => "25px 30px",
    "box-shadow" => "0 10px 30px rgba(0,0,0,0.08)",
    "margin-bottom" => "30px"
)

curseur_style = Styles(
    "margin-bottom" => "12px",
    "accent-color" => "#B4E0C9"
)


ui = DOM.div(
    DOM.h1("Simulateur de réadmission à l’hôpital",
        style=Styles("font-size" => "34px", "color" => "#1a1a1a", "font-weight" => "700", "margin-bottom" => "10px")
    ),
    DOM.p("Simulation de Monte Carlo interactive pour estimer le risque de réadmission des patients.",
        style=Styles("font-size" => "17px", "color" => "#444", "margin-bottom" => "25px", "font-weight" => "500")
    ),

    DOM.div(
        DOM.h3("Paramètres de simulation", style=Styles("color" => "#1a1a1a", "font-weight" => "600")),
        DOM.div(DOM.p(lift(noise) do v "Bruit σ = $(round(v, digits=2))" end), curseur_bruit, style=curseur_style),
        DOM.div(DOM.p(lift(n_iter) do v "Itérations = $(v)" end), curseur_iter, style=curseur_style),
        DOM.div(DOM.p(lift(subset_size) do v "Sous-échantillon = $(v)" end), curseur_subset, style=curseur_style),
        DOM.div("Multi-thread :", case_thread),
        DOM.div(bouton_run, style=Styles("margin-top" => "15px")),
        style=merge(carte_style, Styles("background-color" => "#B4E0C9"))
    ),

    DOM.div(
        DOM.h3("Statistiques du modèle", style=Styles("color" => "#1a1a1a", "font-weight" => "600")),
        stats_html,
        style=merge(carte_style, Styles("background-color" => "#EFD1D4"))
    ),

    DOM.div(
        DOM.h3("Distribution des prédictions", style=Styles("color" => "#1a1a1a", "font-weight" => "600")),
        fig,
        style=merge(carte_style, Styles("background-color" => "#C5BD96"))
    ),

    style=app_style
)

display(App(() -> ui))
