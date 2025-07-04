# --- Imports and Config (unchanged) ---
include("../../config/case_config.jl")
include("ReadData.jl")
include("ReadWind.jl")
include("NodeY.jl")
using .TSSNETConfigModule
using JuMP, Gurobi, Logging
using MathOptInterface
using JSON
using Statistics
using Random

# --- Select case and load config ---
case = length(ARGS) > 0 ? ARGS[1] : "A"
conf = get_case_config(case)
data_file     = conf.data_file
wind_file     = conf.wind_file
output_file   = conf.output_file_opt    # output file for generated dataset
iteration_num = conf.iteration_num      # how many Neur2SP data points to generate
K0 = 10                                 # number of scenarios per data point, set as needed

# --- Read data ---
data = ReadData(data_file)
Wind_data = ReadWind(data, wind_file)

# --- Parameters setup ---
T = length(data.LT.PD)
G = isnothing(conf.G) ? data.BP.unitN : conf.G
N = data.BP.busN
W = data.WD.W
S_w = length(Wind_data[1][1,:])    # wind power scenarios per wind farm
Gbus = [unit.busG for unit in data.UP]
Wbus = data.WD.WN
Pramp = [unit.ramp for unit in data.UP]
Pstart = [unit.PG_up * 0.5 for unit in data.UP]
Pshut = [unit.PG_up for unit in data.UP]
Pup = [unit.PG_up for unit in data.UP]
Plow = [unit.PG_low for unit in data.UP]
Ton = [unit.T_on for unit in data.UP]
Toff = [unit.T_off for unit in data.UP]
U0 = vec(zeros(G,1))
L0 = vec(zeros(G,1))
u0 = vec(zeros(G,1))

# --- Wind scenarios to 3D array (12, T, S_w) ---
Wdata_3D = Array{Float64,3}(undef, 12, T, S_w)
for i in 1:12
    Wdata_3D[i,:,:] = Wind_data[i]
end

# --- Function to generate random feasible unit commitment (respects min up/down times) ---
function sample_random_feasible_u(G, T, Ton, Toff, U0, L0, u0)
    u = zeros(Int, G, T)
    for i in 1:G
        # Set initial
        for t in 1:(U0[i] + L0[i])
            u[i, t] = u0[i]
        end
        t = Int(U0[i] + L0[i] + 1)
        while t <= T
            if rand() < 0.5
                # ON for at least Ton[i]
                for s in t:min(T, t+Ton[i]-1)
                    u[i, s] = 1
                end
                t += Ton[i]
            else
                # OFF for at least Toff[i]
                for s in t:min(T, t+Toff[i]-1)
                    u[i, s] = 0
                end
                t += Toff[i]
            end
        end
    end
    return u
end

# --- Simple second stage cost calculation ---
# You may want to replace this with a more realistic recourse/dispatch model!
function solve_second_stage(u, wind, data, conf)
    # This example simply penalizes load loss and generation cost.
    # Replace with your actual dispatch model if available!
    # You can also implement a JuMP dispatch subproblem here.
    gen_cost = sum(u)             # Placeholder (replace!)
    wind_power = sum(wind)
    load = sum([bus.PD for bus in data.LN])
    penalty = max(0, load - wind_power) * 1000.0  # Penalty for unmet demand (if any)
    return gen_cost + penalty
end

# --- Main Neur2SP data generation loop ---
dataset = []
for it in 1:iteration_num
    u = sample_random_feasible_u(G, T, Ton, Toff, U0, L0, u0)
    wind_scenarios = [Array{Float64}(undef, W, T) for _ in 1:K0]
    for k in 1:K0
        for i in 1:W
            index = rand(1:12)
            index2 = rand(1:S_w)
            wind_scenarios[k][i, :] = Wdata_3D[index, :, index2]
        end
    end
    # Recourse costs for all scenarios
    costs = [solve_second_stage(u, ws, data, conf) for ws in wind_scenarios]
    expected_cost = mean(costs)
    push!(dataset, Dict(
        "u" => u,
        "wind_scenarios" => wind_scenarios,
        "expected_cost" => expected_cost
    ))
    if it % 10 == 0
        println("Sample $it / $iteration_num: mean cost = $expected_cost")
    end
end

# --- Save dataset ---
open(output_file, "w") do f
    JSON.print(f, dataset)
end

println("Neur2SP-style dataset generated: $(length(dataset)) samples in $output_file")
