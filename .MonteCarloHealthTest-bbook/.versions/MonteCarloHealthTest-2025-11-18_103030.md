# New Book

```julia (editor=true, logging=true, output=true)
using Bonito
using Pkg;Pkg.activate(".")
using MonteCarloHealth
```
```julia (editor=true, logging=false, output=true)
using Bonito
# Create a reactive counter app
app = App() do session
    count = Observable(0)
    button = Button("Click me!")
    on(click-> (count[] += 1), button)
    return DOM.div(button, DOM.h1("Count: ", count))
end
display(app) # display it in browser or plotpane
export_static("app.html", app) # generated self contained HTML file from App
export_static("folder", Routes("/" => app)) # Export static site (without Julia connection)
# Or serve it on a server
server = Server(app, "127.0.0.1", 8888)
# add it as different route
# regex, and even custom matchers are possible for routes, read more in the docs!
route!(server, "/my-route" => app)
```
