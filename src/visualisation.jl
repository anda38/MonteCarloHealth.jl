using StatsPlots, DataFrames

"""
    plot_simulation_results(preds::Vector{Float64}; title="Monte Carlo Prediction Distribution")

Visualize the distribution of simulated predictions as a histogram.
"""
function plot_simulation_results(preds::Vector{Float64}; title="Monte Carlo Prediction Distribution")
    histogram(preds,
              bins=30,
              alpha=0.7,
              xlabel="Predicted Probability",
              ylabel="Frequency",
              title=title,
              legend=false)
end


"""
    plot_correlation_heatmap(df::DataFrame)

Displays a heatmap of correlations among numeric features.
"""
function plot_correlation_heatmap(df::DataFrame)
    num = select(df, names(df, Number))
    cor_matrix = cor(Matrix(num))
    heatmap(names(num), names(num), cor_matrix,
            c=:viridis, aspect_ratio=1, title="Correlation Heatmap")
end
