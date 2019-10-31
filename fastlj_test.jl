include("Experiment.jl")

using JuLIP, BenchmarkTools
using Experiment

A = [1.0 sin(π/6); 0.0 cos(π/6)]
# n = Int64(10.0 / 1.0 + 1.0)
n = 50
t = 1:n
o = ones(n)
xref = A * [(t * o')[:]'; (o * t')[:]']
xvecs = [xref; zeros(n^2)'] |> vecs
at = Atoms(:X, xvecs)
# set_pbc!(at, (true, true, true))
set_constraint!(at, FixedCell2D(at))
set_calculator!(at, lennardjones())
V = lennardjones()

# info("Profiling the standard LJ calculator:")
# print("Neighbourlist: "); @btime neighbourlist($at, $(cutoff(V)))
# print("energy: "); @btime energy($V, $at);
# print("forces: "); @btime forces($V, $at);

info("Profile the Experimental calculator")
ljf = Experiment.FastLJ()
energy(ljf, at) # warmup
at = Atoms(:X, xvecs) # new at to clear the data
print("First energy: "); @time energy(ljf, at) # first assembly requires nlist
print("Second energy: "); @btime energy(ljf, at) # second assembly uses stored nlist

# allocate storage for the forces
F = zeros(JVecF, length(at))

info("Profiling energy, forces, forces! without assembly:")
print("energy: "); @btime energy($ljf, $at)
print("forces: "); @btime forces($ljf, $at)
print("forces!: "); @btime Experiment.forces!($F, $ljf, $at);

# pseudo-evolution: should observe occasional nlist assembly
# for n = 1:50
#     rattle!(at, 0.1)
#     @time (energy(ljf, at), forces(ljf, at));
# end