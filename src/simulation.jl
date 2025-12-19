# Ajoute du bruit gaussien sur les colonnes numériques en réutilisant la table
# (copie superficielle) pour limiter les allocations. L'argument RNG explicite
# évite la contention sur le RNG global en mode multi-thread.
function add_noise(df::DataFrame, num_cols::Vector{Symbol}, σ::Float64, rng::AbstractRNG=Random.default_rng())
    df_noisy = copy(df)                       # copie superficielle pour réutiliser la structure
    for c in num_cols                         # boucle sur chaque colonne numérique ciblée
        col = df[!, c]                        # extrait la colonne d'origine
        df_noisy[!, c] = col .+ σ .* randn(rng, length(col))  # applique le bruit gaussien avec RNG passé
    end
    return df_noisy                           # renvoie le DataFrame bruité (même schéma)
end

# Convertit une prédiction probabiliste MLJ en probabilité de la classe positive.
function predict_proba(mach::Machine, X::DataFrame)
    yhat = MLJ.predict(mach, X)               # appel MLJ générique (peut renvoyer UnivariateFinite)
    if yhat isa AbstractVector{<:UnivariateFinite}  # cas classification probabiliste
        pos_class = last(levels(first(yhat)))       # choisit la dernière classe comme positive
        return [pdf(y, pos_class) for y in yhat]    # extrait la proba de la classe positive
    else
        return Float64.(yhat)                        # convertit directement en Float64 si déjà numérique
    end
end

abstract type Simulation end

struct BasicSimulation <: Simulation
    model::Machine
    n_iter::Int
    noise_level::Float64
end

struct ThreadedSimulation <: Simulation
    model::Machine
    n_iter::Int
    noise_level::Float64
end

# Simulation séquentielle avec préallocation pour minimiser les allocations.
function simulate(sim::BasicSimulation, X::DataFrame)
    num_cols = Symbol.(names(X)[findall(c -> eltype(X[!, c]) <: Real, names(X))])  # isole les colonnes numériques
    n_rows = nrow(X)                                                                # taille des vecteurs de sortie

    # On extrait une fois les colonnes numériques (Float64) pour les réinjecter
    # dans un DataFrame réutilisé à chaque itération. Cela évite de créer un
    # DataFrame complet à chaque boucle (goulot observé en profilage).
    base_cols = [Float64.(X[!, c]) for c in num_cols]   # conversion unique en Float64
    X_noisy = DataFrame(X)                              # DataFrame réutilisé pour injecter le bruit
    for (idx, c) in enumerate(num_cols)                 # initialise les colonnes numériques avec les bases
        X_noisy[!, c] = copy(base_cols[idx])            # copie pour pouvoir les modifier sans toucher X
    end

    # Tampon pour générer le bruit sans réallouer.
    preds = zeros(n_rows)                               # accumulateur des probabilités moyennes
    noise_buf = zeros(n_rows)                           # tampon pour randn! afin d'éviter de nouvelles allocations

    for _ in 1:sim.n_iter                               # boucle principale des simulations
        for (idx, c) in enumerate(num_cols)             # ajoute du bruit sur chaque colonne numérique
            randn!(noise_buf)                           # remplit le tampon de bruit gaussien
            noisy_col = X_noisy[!, c]                   # référence à la colonne modifiable
            base_col = base_cols[idx]                   # colonne de base (sans bruit)
            @inbounds @. noisy_col = base_col + sim.noise_level * noise_buf  # applique le bruit in-place
        end
        preds .+= predict_proba(sim.model, X_noisy)     # ajoute les probabilités prédites à l'accumulateur
    end

    return preds ./ sim.n_iter                          # moyenne des probabilités sur toutes les itérations
end

# Simulation multi-thread avec découpage manuel en blocs pour réduire le
# surcoût du scheduler, RNG par thread pour éviter la contention, et
# préallocation locale pour diminuer la pression GC (issues vues lors des tests).
function simulate(sim::ThreadedSimulation, X::DataFrame)
    nthreads_active = Threads.nthreads()                                             # nombre de threads disponibles
    if nthreads_active == 1                                                          # garde-fou : pas de parallélisme
        @warn "1 seul thread actif détecté. Utilisation de BasicSimulation à la place."
        return simulate(BasicSimulation(sim.model, sim.n_iter, sim.noise_level), X)  # bascule en mode séquentiel
    end

    @info "Simulation avec $nthreads_active threads"

    num_cols = Symbol.(filter(c -> eltype(X[!, c]) <: Real, names(X)))               # colonnes numériques ciblées
    base_cols = [Float64.(X[!, c]) for c in num_cols]                                # stockage base en Float64

    n_rows = nrow(X)                                                                 # taille des vecteurs par thread
    preds_per_thread = Vector{Vector{Float64}}(undef, nthreads_active)               # accumulations locales
    chunk = cld(sim.n_iter, nthreads_active)                                         # nombre d'itérations par thread
    rngs = [MersenneTwister(0x1234 + tid) for tid in 1:nthreads_active]              # RNG indépendants par thread

    Threads.@threads for tid in 1:nthreads_active                                    # parallélise sur les threads actifs
        rng = rngs[tid]                                                              # RNG spécifique au thread courant
        local_preds = zeros(n_rows)                                                  # accumulation locale des probabilités
        # DataFrame réutilisé par thread pour limiter les allocations partagées.
        X_noisy = DataFrame(X)                                                       # DataFrame dédié au thread
        for (idx, c) in enumerate(num_cols)                                          # initialise avec les colonnes de base
            X_noisy[!, c] = copy(base_cols[idx])                                     # copie pour modifier indépendamment
        end
        # Tampon local pour le bruit afin de rester thread-safe.
        noise_buf = zeros(n_rows)                                                    # tampon local pour randn!
        start_iter = (tid - 1) * chunk + 1                                           # début du bloc géré par ce thread
        end_iter = min(tid * chunk, sim.n_iter)                                      # fin du bloc (borne supérieure)
        for _ in start_iter:end_iter                                                 # boucle des itérations assignées
            for (idx, c) in enumerate(num_cols)                                      # applique le bruit colonne par colonne
                randn!(rng, noise_buf)                                               # remplit le tampon via RNG du thread
                noisy_col = X_noisy[!, c]                                            # référence colonne modifiable
                base_col = base_cols[idx]                                            # colonne de base correspondante
                @inbounds @. noisy_col = base_col + sim.noise_level * noise_buf      # met à jour in-place avec bruit
            end
            local_preds .+= predict_proba(sim.model, X_noisy)                        # accumulateur local
        end
        preds_per_thread[tid] = local_preds                                          # enregistre l'accumulation du thread
    end

    preds_total = reduce(+, preds_per_thread)                                        # agrège les contributions de tous les threads
    return preds_total ./ sim.n_iter                                                 # moyenne globale sur toutes les itérations
end
