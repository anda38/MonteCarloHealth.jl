# New Book

```julia (editor=true, logging=false, output=true)
using Bonito
using Pkg;Pkg.activate(".")
using MonteCarloHealth
```
```julia (editor=true, logging=false, output=true)
using Bonito 
df = load_and_clean_data()
mach = train(df)   # your clean wrapper
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
