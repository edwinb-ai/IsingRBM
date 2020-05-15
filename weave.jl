# using Pkg
# Pkg.activate(".")
using Weave

weave("plots_ising.jmd", doctype = "md2html", out_path=:pwd)
