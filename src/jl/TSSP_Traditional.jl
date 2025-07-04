# --- Imports and Config ---
include("../../config/case_config.jl")
include("ReadData.jl")
include("ReadWind.jl")
include("NodeY.jl")
using .TSSNETConfigModule
using JuMP, Gurobi, Logging
using MathOptInterface
using JSON
using DataFrames
using Statistics
using Random
using Flux
using BSON: @save, @load

# --- Select case and load config ---
case = length(ARGS) > 0 ? ARGS[1] : "A"
conf = get_case_config(case)

autoencoder_file  = conf.autoencoder_file
data_file     = conf.data_file
wind_file     = conf.wind_file
output_file   = conf.output_file_opt
iteration_num = conf.iteration_num
Lup           = conf.Lup
G             = conf.G
debug         = conf.debug

if debug
    @info "Case: $case | data: $data_file | wind: $wind_file | output: $output_file | iters: $iteration_num | Lup: $Lup | G: $G | debug: $debug"
end

# --- Read data ---
data = ReadData(data_file, debug=debug)
Wind_data = ReadWind(data, wind_file, debug=debug)

# --- Parameters setup ---
T = length(data.LT.PD)
Gval = isnothing(G) ? data.BP.unitN : G
G = Gval
N = data.BP.busN
W = data.WD.W
S_w = length(Wind_data[1][1,:])    # wind power scenarios per wind farm
S = 50                             # Number of uncertainty scenarios, adjust as needed or add to config
Gbus = [unit.busG for unit in data.UP]
Dbus = [bus.N for bus in data.LN]
Wbus = data.WD.WN
Pramp = [unit.ramp for unit in data.UP]
Pstart = [unit.PG_up * 0.5 for unit in data.UP]
Pshut = [unit.PG_up for unit in data.UP]
Pup = [unit.PG_up for unit in data.UP]
Plow = [unit.PG_low for unit in data.UP]
Ton = [unit.T_on for unit in data.UP]
Toff = [unit.T_off for unit in data.UP]
L = 4
fa = [unit.Calpha for unit in data.UP]
fb = [unit.Cbeta for unit in data.UP]
fc = [unit.Cgamma for unit in data.UP]
Ccold = [unit.StartCost * 2 for unit in data.UP]
Chot = [unit.StartCost for unit in data.UP]
Cll = 250
Tcold = min.(max.(Toff,ones(G)),T)
Lupv = Lup * ones(N)
Llow = 0 * ones(N)
LN_ori = data.LN

type_of_pf = "DC"
GM, BM = NodeY(data, type_of_pf)

u0 = vec(zeros(G,1)) # the initial status of each unit, assume all units are off at the beginning
T0 = vec(zeros(G,1)) # number of periods units have been on or off 
P0 = vec(zeros(G,1)) # the initial generation of each unit
U0 = vec(zeros(G,1)) # number of periods that units should be on from the first period of current time framework
L0 = vec(zeros(G,1)) # number of periods that units should be off from the first period of current time framework
tao0 = Toff .+ Tcold .+1
for i = 1:G
    mid_u = min(T,u0[i] * (Ton[i] - T0[i]))
    mid_l = min(T,(1-u0[i]) * (Toff[i] + T0[i]))
    U0[i] = max(0,mid_u)
    L0[i] = max(0,mid_l)
end

# --- Branch and transformer data ---
allbranch_start = [[branch.fromBus for branch in data.BR]; [transformer.fromBus for transformer in data.TP]]
allbranch_end = [[branch.toBus for branch in data.BR]; [transformer.toBus for transformer in data.TP]]
allbranch_P = [[branch.P_max for branch in data.BR]; [transformer.P_max for transformer in data.TP]]
B = length(allbranch_start)

# --- Wind scenarios to 3D array (12, T, S_w) ---
Wdata_3D = Array{Float64,3}(undef, 12, T, S_w)
for i in 1:12
    Wdata_3D[i,:,:] = Wind_data[i]
end

# --- Load encoder for wind scenario encoding ---
@load autoencoder_file encoder

# --- Setup output file ---
open(output_file, "w") do f
    JSON.print(f, Dict("IN" => iteration_num))
end
#=
# --- Forbidden set for previous solutions ---
forb_u = []

# --- Add forbidden constraints function ---
function add_forb(model, u, forbidden, a)
    for i in 1:G, j in 1:T
        if forbidden[i, j] == 0
            @constraint(model, (u[i, j] - forbidden[i, j]) >= a[i, j])
        else
            @constraint(model, (forbidden[i, j] - u[i, j]) >= a[i, j])
        end
    end
    @constraint(model, (sum(a) >= 1))
end
=#

# --- Main optimization loop ---
for it in 1:iteration_num
    # --- Fluctuate load ---
    LN = Vector{Vector{Float64}}(undef, length(LN_ori))
    ln = Array{Float64}(undef, T, 1)
    for i = 1:length(Dbus)
        fluctuations = (rand(length(LN_ori[i].PD)) .- 0.5) .* 2 .* 0.05
        fluctuated_vector = LN_ori[i].PD .+ fluctuations
        LN[i] = fluctuated_vector .+ (LN_ori[i].PD .- mean(fluctuated_vector))
        ln += LN[i]
    end

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    # --- Random wind scenarios ---
    Pw = []
    Pw_encoder = []
    for i in 1:W
        tmp = Array{Float64,2}(undef, T, S)
        for j in 1:S
            index = rand(1:12)
            index2 = rand(1:S_w)
            tmp[:,j] = Wdata_3D[index,:,index2]
        end
        tmp_encoder = encoder(tmp)
        append!(Pw_encoder, [tmp_encoder])
        append!(Pw, [tmp])
    end

    # --- Variables ---
    @variable(model, Pg[1:G,1:T,1:S])
    @variable(model, theta[1:N,1:T,1:S])
    @variable(model, l[1:N,1:T,1:S])
    @variable(model, z[1:G,1:T,1:S])
    @variable(model, Ccl[1:N,1:T,1:S])
    @variable(model, Chs[1:G,1:T])
    @variable(model, Ces[1:G,1:T] >= 0)
    @constraint(model, theta[1,:,:] .== 0)
    @variable(model, u[1:G,1:T], Bin)
    @variable(model, s[1:G,1:T], Bin)
    @variable(model, d[1:G,1:T], Bin)
    @variable(model, a[1:G,1:T,1:it], Bin)

    #=
    # --- Forbid previous optimal solutions ---
    it_forb = 1
    for forb_i in forb_u
        add_forb(model, u, forb_i, a[:,:,it_forb:it_forb])
        it_forb += 1
    end
    =#

    # --- Thermal unit constraints (as in your original script) ---
    for i in 1:G
        for t in 1:(Int(U0[i]+L0[i]))
            @constraint(model, u[i,t] == u0[i])
        end
        for t in 1:T
            @constraint(model, Pg[i,t,:] .<= Pup[i] * u[i,t])
            @constraint(model, Pg[i,t,:] .>= Plow[i] * u[i,t])
            if t == 1
                @constraint(model, Pg[i,t,:] .- P0[i] .<= u0[i] * Pramp[i] + s[i,t] * Pstart[i])
                @constraint(model, P0[i] .- Pg[i,t,:] .<= u[i,t] * Pramp[i] + d[i,t] * Pshut[i])
                @constraint(model, s[i,t] - d[i,t] == u[i,t] - u0[i])
            else
                @constraint(model, Pg[i,t,:] - Pg[i,t-1,:] .<= u[i,t-1] * Pramp[i] + s[i,t] * Pstart[i])
                @constraint(model, Pg[i,t-1,:] - Pg[i,t,:] .<= u[i,t] * Pramp[i] + d[i,t] * Pshut[i])
                @constraint(model, s[i,t] - d[i,t] == u[i,t] - u[i,t-1])
            end
            omiga_on = max(0,t - Ton[i]) + 1
            if omiga_on <= t && t >= 1+U0[i]
                @constraint(model, sum(s[i,omiga_on:t]) <= u[i,t])
            end
            omiga_off = max(0,t - Toff[i]) + 1
            if omiga_off <= t && t >= 1+L0[i]
                @constraint(model, sum(d[i,omiga_off:t]) <= 1 - u[i,t])
            end 
        end
    end

    # --- Branch constraints ---
    for b in 1:B
        i = allbranch_start[b]
        j = allbranch_end[b]
        x_cb = abs(1/BM[i,j])
        @constraint(model, theta[i,:,:] - theta[j,:,:] .<= x_cb * allbranch_P[b])
        @constraint(model, -allbranch_P[b] * x_cb .<= theta[i,:,:] - theta[j,:,:])
    end

    # --- System constraints ---
    for t in 1:T
        right = Vector{Any}(undef, S)
        right[:] .= 0
        left = Vector{Any}(undef, S)
        left[:] .= 0
        for i in 1:N
            @constraint(model, Llow[i] .<= l[i,t,:])
            @constraint(model, l[i,t,:] .<= Lupv[i])
            for j in 1:N
                right .+= BM[i,j] * theta[j,t,:]
            end
            if i in Gbus
                left .+= Pg[findfirst(isequal(i), Gbus),t,:]
            end
            if i in Wbus
                left .+= Pw[findfirst(isequal(i), Wbus)][t,1:S]
            end
            left .+= l[i,t,:]
        end
        left .-= ln[t]
        @constraint(model, left .== right)
    end

    # --- Generation cost ---
    for i in 1:N
        if i in Gbus
            index = findfirst(isequal(i), Gbus)
            for t in 1:T
                @constraint(model, Chs[index,t] == Chot[index] * s[index,t]) 
                for l_idx in 0:(L-1)
                    pil = Plow[index] + l_idx * (Pup[index] - Plow[index]) / L
                    @constraint(model, [s=1:S], z[index,t,s] >= (2*fc[index]*pil + fb[index]) * Pg[index,t,s] + (fa[index] - fc[index]*(pil^2)) * u[index,t])
                end
                taoi = Int(max(1, t-Toff[index]-Tcold[index]))
                fit = ((t-taoi <= 0) && (max(0, -T0[index]) < abs(t-taoi) + 1)) ? 1 : 0
                @constraint(model, Ces[index,t] >= (Ccold[index] - Chot[index]) * (s[index,t] - sum(d[index,taoi:t-1]) - fit))
            end
        end
        @constraint(model, Ccl[i,:,:] .== Cll * l[i,:,:])
    end

    # --- Objective ---
    @objective(model, Min, sum(sum(Chs)) + sum(sum(Ces)) + (1/S) * sum(sum(sum(z))) + (1/S) * sum(sum(sum(Ccl))))

    optimize!(model)

    # --- Results and logging ---


    scd_obj = (1/S) * sum(sum(sum(value.(z)))) + (1/S) * sum(sum(sum(value.(Ccl))))
    first_obj = sum(sum(value.(Chs)) + sum(value.(Ces)))
    @info "$it iteration's Second Objective value: $scd_obj"
    @info "$it iteration's First Objective value: $first_obj"
    opt_u = round.(Int, value.(u))
    ### push!(forb_u, opt_u)

    # --- Save iteration results to JSON ---
    my_data = JSON.parsefile(output_file, use_mmap = false)
    my_data["load$it"] = ln
    my_data["scd_obj$it"] = scd_obj
    my_data["opt_u$it"] = opt_u
    my_data["Pw$it"] = Pw_encoder
    open(output_file, "w") do f
        JSON.print(f, my_data)
    end

    model = nothing # free up memory
end

if debug
    @info "TSSP Traditional finished for case $case"
end
