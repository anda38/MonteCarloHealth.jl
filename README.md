lancer julia avec threads avant de lancer le code : julia --project=. -t 4 ou t 8 

src/ : code principal du projet

examples/ : scripts exécutables

grandfinale.jl : simulations Monte Carlo (version basique et multi-threadée, sans interface)

bonito_dashboard.jl : tableau de bord interactif (Bonito + Makie)

notebook/ : rapport Quarto expliquant le projet, le processus de développement et le débogage
