# New Book

```julia (editor=true, logging=false, output=true)
using MLJ
using MLJDecisionTreeInterface
using DecisionTree
using DataFrames
using Statistics
using MonteCarloHealth
using StatsPlots

@load RandomForestClassifier pkg=DecisionTree

df = MonteCarloHealth.load_and_clean_data()
target = :readmitted_30_days

y, X = MLJ.unpack(df, ==(target), rng=123)
y = categorical(y)

model = RandomForestClassifier()
mach = machine(model, X, y) |> fit!

df = MonteCarloHealth.load_and_clean_data()
target = :readmitted_30_days

y, X = MLJ.unpack(df, ==(target), rng=123)
y = categorical(y)

model = RandomForestClassifier()
mach = machine(model, X, y) |> fit!

```
