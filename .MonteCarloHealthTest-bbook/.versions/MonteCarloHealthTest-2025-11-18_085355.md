# New Book

```julia (editor=true, logging=true, output=true)
using Pkg;Pkg.activate(".")
using MonteCarloHealth
```
```julia (editor=true, logging=false, output=true)

df = MonteCarloHealth.load_and_clean_data()

target = :readmitted_30_days
features = [
    :age, :cholesterol, :bmi,
    :diabetes, :hypertension,
    :medication_count, :length_of_stay,
    :blood_pressure_systolic, :blood_pressure_diastolic,
    :gender_Female, :gender_Male, :gender_Other,
    :discharge_Home, :discharge_Nursing_Facility, :discharge_Rehab
]

y, X = MLJ.unpack(df, ==(target), rng=123)
y = categorical(y)

model = RandomForestClassifier()
mach = machine(model, X, y) |> fit!



age = @bind Slider(20:1:100, default=50)
bmi = @bind Slider(15:1:45, default=28)
chol = @bind Slider(100:1:300, default=190)

bp_sys = @bind Slider(90:1:200, default=130)
bp_dia = @bind Slider(50:1:130, default=80)

meds = @bind Slider(0:1:15, default=3)
los  = @bind Slider(1:1:30, default=5)

diab = @bind Checkbox(false)
hyper = @bind Checkbox(false)

gender = @bind Select(["Female","Male","Other"], default="Female")
discharge = @bind Select(["Home","Nursing_Facility","Rehab"], default="Home")

noise_level = @bind Slider(0.0:0.01:0.3, default=0.05)
n_iter = @bind Slider(10:10:1000, default=200)
```
