# New Book

```julia (editor=true, logging=false, output=true)
using Pkg;Pkg.activate(".")
using MonteCarloHealth
```
```julia (editor=true, logging=true, output=true)
df = load_and_clean_data()
mach = train(df) 
```
