
using DataFrames

df = MonteCarloHealth.load_and_clean_data()
MonteCarloHealth.description(df)

MonteCarloHealth.describe(df)
MonteCarloHealth.correlation_table(df)