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

y, X = unpack(df, ==(target), rng=123)
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

function build_patient_row()
    df = DataFrame(
        age = age[],
        cholesterol = chol[],
        bmi = bmi[],
        diabetes = diab[] ? 1 : 0,
        hypertension = hyper[] ? 1 : 0,
        medication_count = meds[],
        length_of_stay = los[],
        blood_pressure_systolic = bp_sys[],
        blood_pressure_diastolic = bp_dia[],
        gender_Female = gender[] == "Female",
        gender_Male   = gender[] == "Male",
        gender_Other  = gender[] == "Other",
        discharge_Home = discharge[] == "Home",
        discharge_Nursing_Facility = discharge[] == "Nursing_Facility",
        discharge_Rehab = discharge[] == "Rehab",
    )
    return df
end


###
# Reactive prediction + simulation
###

function predict_once()
    Xnew = build_patient_row()
    prob = MonteCarloHealth.predict_proba(mach, Xnew)[1]
    return prob
end

function run_simulation()
    Xnew = build_patient_row()
    sim = ThreadedSimulation(mach, n_iter[], noise_level[])
    preds = MonteCarloHealth.simulate(sim, Xnew)
    return preds
end


###
# UI Layout
###

md"""
# 🧬 Patient Profile Simulator
Adjust the patient characteristics below and view the predicted readmission risk.

---

## Patient Inputs

### 🧑 Demographics
- **Age**: $(age)
- **BMI**: $(bmi)
- **Gender**: $(gender)

### 🩺 Clinical
- **Cholesterol**: $(chol)
- **Blood Pressure Systolic**: $(bp_sys)
- **Blood Pressure Diastolic**: $(bp_dia)
- **Diabetes**: $(diab)
- **Hypertension**: $(hyper)

### 💊 Treatment
- **Medication Count**: $(meds)
- **Length of Stay**: $(los)
- **Discharge Destination**: $(discharge)

---

## 🔮 Simulation Controls  
- **Noise level**: $(noise_level)
- **Monte Carlo Iterations**: $(n_iter)
"""

probability = @react predict_once()
sim_preds   = @react run_simulation()


###
# Display results
###

md"""
## 📊 Base model prediction  
**Probability of 30-day readmission:**  
# **$(round(probability[] * 100, digits=2)) %**

---

## 🎲 Monte Carlo Simulation Results  

Mean: **$(round(mean(sim_preds[]), digits=3))**  
Std: **$(round(std(sim_preds[]), digits=3))**  
Min/Max: **$(round(minimum(sim_preds[]), digits=3)) — $(round(maximum(sim_preds[]), digits=3))**

"""

@react begin
    histogram(sim_preds[], bins=30,
        xlabel="Predicted probability",
        title="Monte Carlo distribution",
        legend=false)
end
```
