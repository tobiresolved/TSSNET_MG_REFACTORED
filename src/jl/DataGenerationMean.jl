# Unified Data Generation Script for TSSNet-MG (Cases A, B, C)
# ------------------------------------------------------------

# 1. ---- Load necessary packages and modules ----
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
using BSON    


# 2. ---- Load config ----

function data_generation_main()
    num_infeasible = 0
    num_feasible = 0
    case = length(ARGS) > 0 ? ARGS[1] : "A"
    conf = get_case_config(case)

    autoencoder_file = conf.autoencoder_file
    data_file     = conf.data_file
    wind_file     = conf.wind_file
    output_file = conf.output_file_ram * "_mean.json"
    iteration_num = conf.iteration_num
    Lup           = conf.Lup
    Gparam        = conf.G
    debug         = conf.debug

    if debug
        @info "Data Generation for Case $case"
        @info "Using config: $conf"
    end

    data = ReadData(data_file, debug=debug)
    Wind_data = ReadWind(data, wind_file, debug=debug)


    # 4. ---- System parameters ----
    T = length(data.LT.PD)
    G = isnothing(Gparam) ? data.BP.unitN : Gparam
    N = data.BP.busN
    W = data.WD.W
    print("Number of wind farms: $W, Number of buses: $N, Number of units: $G, Time steps: $T\n")
    S_w = length(Wind_data[1][1,:])
    S = 50             # Number of wind scenarios per sample (hardcoded, or add to config if needed)
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
    Tcold = min.(max.(Toff, ones(G)), T)
    Lupv = Lup * ones(N)
    Llow = 0 * ones(N)
    LN_ori = data.LN

    type_of_pf = "DC"
    GM, BM = NodeY(data, type_of_pf)

    u0 = vec(zeros(G,1)) # the initial status of each unit, assume all units are off at the beginning
    T0 = vec(zeros(G,1)) # number of periods units have been on or off 
    P0 = vec(zeros(G,1)) # the initial generation of each unit
    U0 = vec(zeros(G,1)) # number of periods that units should be on from the first period of current time framework
    L0 = vec(zeros(G,1))
    tao0 = Toff .+ Tcold .+ 1
    for i = 1:G
        mid_u = min(T, u0[i] * (Ton[i] - T0[i]))
        mid_l = min(T, (1-u0[i]) * (Toff[i] + T0[i]))
        U0[i] = max(0, mid_u)
        L0[i] = max(0, mid_l)
    end

    # ---- Branch/transformer data ----
    allbranch_start = [[branch.fromBus for branch in data.BR]; [transformer.fromBus for transformer in data.TP]]
    allbranch_end   = [[branch.toBus for branch in data.BR]; [transformer.toBus for transformer in data.TP]]
    allbranch_P     = [[branch.P_max for branch in data.BR]; [transformer.P_max for transformer in data.TP]]
    B = length(allbranch_start)

    # ---- Wind data to 3D array ----
    Wdata_3D = Array{Float64, 3}(undef, 12, T, S_w)
    for i in 1:12
        Wdata_3D[i, :, :] = Wind_data[i][1:T, :]
    end

    # ---- Load the encoder ----
    d = BSON.load("model_weights.bson")
    println(keys(d))  # Should show (:encoder_weights,)
    encoder = Chain(
        x -> reshape(x, 50 * 24),
        Dense(50 * 24, 256, relu),
        Dense(256, 128, relu),
        Dense(128, 24, relu)
    )
    @load "model_weights.bson" encoder_weights
    Flux.loadmodel!(encoder, encoder_weights) 

    # ---- Write initial JSON ----
    open(output_file, "w") do f
        JSON.print(f, Dict("IN" => iteration_num))
    end

    # ---- Data generation loop ----


    for it in 1:iteration_num
        if debug && it % 10 == 1
            @info "Iteration $it / $iteration_num"
        end

        # --- Model ---
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)

           # --- Random wind scenarios, but aggregate via mean instead of encoding ---
            Pw = []
            Pw_mean = []
            for i in 1:W
                tmp = Array{Float64,2}(undef, T, S)
                for j in 1:S
                    index = rand(1:12)
                    index2 = rand(1:S_w)
                    tmp[:,j] = Wdata_3D[index,:,index2]
                end
                tmp_mean = mean(tmp, dims=2) |> vec
                append!(Pw_mean, [tmp_mean])
                append!(Pw, [tmp])
            end

        # --- Randomly perturb load demand ---
        LN = Vector{Vector{Float64}}(undef, length(LN_ori))
        ln = zeros(Float64, T)
        for i = 1:length(Dbus)
            fluctuations = (rand(length(LN_ori[i].PD)) .- 0.5) .* 2 .* 0.05
            fluctuated_vector = LN_ori[i].PD .+ fluctuations
            LN[i] = fluctuated_vector .+ (LN_ori[i].PD .- mean(fluctuated_vector))
            ln += LN[i]
        end

        # --- Variables (add, change, or adjust as needed) ---
        @variable(model, Pg[1:G,1:T,1:S])
        @variable(model, theta[1:N,1:T,1:S])
        @variable(model, l[1:N,1:T,1:S])
        @variable(model, z[1:G,1:T,1:S])
        @variable(model, Ccl[1:N,1:T,1:S])
        @variable(model, Chs[1:G,1:T])
        @variable(model, Ces[1:G,1:T] >= 0)
        @constraint(model, theta[1,:,:] .== 0)
        @variable(model, a[1:G,1:T,1:it], Bin) # the forbidden status of each unit
        @variable(model, u[1:G,1:T], Bin) # the status of each unit
        if it == 1
            u = zeros(G,T)
        else
            u = rand(0:1,G,T) # randomly generate the status of each unit
            println("Randomly generated unit status for iteration $it: ", u)
            for i in 1:G # set the feasible status of each unit
                for t in 1:(Int(U0[i]+L0[i]))
                    u[i,t] == u0[i] 
                end
        end
        end
        @variable(model, s[1:G,1:T], Bin) # the start status of each unit
        @variable(model, d[1:G,1:T], Bin) # the shut status of each unit

        # --- Initial/fixed status (if needed) ---
        for i in 1:G
            for t in 1:(Int(U0[i]+L0[i]))
                @constraint(model, u[i,t] == u0[i])
            end
        end

        # thermal unit constraints
        for i in 1:G
            for t in 1:(Int(U0[i]+L0[i]))
                @constraint(model, u[i,t] == u0[i]) # the initial status constraint
            end
            for t in 1:T
                @constraint(model, Pg[i,t,:] .<= Pup[i] * u[i,t]) # the upper bound of active generation
                @constraint(model, Pg[i,t,:] .>= Plow[i] * u[i,t]) # the lower bound of active generation
                if t == 1
                    @constraint(model, Pg[i,t,:] .- P0[i] .<= u0[i] * Pramp[i] + s[i,t] * Pstart[i]) # the start and up ramp constraint for t1
                    @constraint(model, P0[i] .- Pg[i,t,:] .<= u[i,t] * Pramp[i] + d[i,t] * Pshut[i]) # the shut and down ramp constraint for t1
                    @constraint(model, s[i,t] - d[i,t] == u[i,t] - u0[i]) # the start and shut status constraint for t1
                else
                    @constraint(model, Pg[i,t,:] - Pg[i,t-1,:] .<= u[i,t-1] * Pramp[i] + s[i,t] * Pstart[i]) # the start and up ramp constraint
                    @constraint(model, Pg[i,t-1,:] - Pg[i,t,:] .<= u[i,t] * Pramp[i] + d[i,t] * Pshut[i]) # the shut and down ramp constraint
                    @constraint(model, s[i,t] - d[i,t] == u[i,t] - u[i,t-1]) # the start and shut status constraint
                end

                omiga_on = max(0,t - Ton[i]) + 1
                if omiga_on <= t && t >= 1+U0[i]
                    @constraint(model, sum(s[i,omiga_on:t]) <= u[i,t]) # the start status constraint
                end

                omiga_off = max(0,t - Toff[i]) + 1
                if omiga_off <= t && t >= 1+L0[i]
                    @constraint(model, sum(d[i,omiga_off:t]) <= 1 - u[i,t]) # the shut status constraint
                end 
            end
        end

        for b in 1:B
            i = allbranch_start[b]
            j = allbranch_end[b]
            x_cb = abs(1 / BM[i, j])
            @constraint(model, theta[i,:,:] - theta[j,:,:] .<= x_cb * allbranch_P[b])
            @constraint(model, -allbranch_P[b] * x_cb .<= theta[i,:,:] - theta[j,:,:])
        end

        # system constraints
        for t in 1:T
            # for each period, the power balance equation should be satisfied
            right = Vector{Any}(undef, S)
            right[:] .= 0
            left = Vector{Any}(undef, S)
            left[:] .= 0
            for i in 1:N
                @constraint(model, Llow[i] .<= l[i,t,:]) # the lower bound of load loss

                #@info "Load loss at bus $i, time $t: $(l[i,t,:])"
                @constraint(model, l[i,t,:] .<= Lupv[i]) # the upper bound of load loss

                # the power balance equation
                # construct rightside item
                for j in 1:N
                    right[:] .+= BM[i,j] * theta[j,t,:]
                end

                # construct leftside item
                if i in Gbus # if bus i is a generation bus
                    # findfirst(isequal(i), Gbus) find the index of the unit at bus i
                    left[:] .+= Pg[findfirst(isequal(i), Gbus),t,:]
                end
                if i in Wbus # if bus i is a wind farm
                    left[:] .+= Pw[findfirst(isequal(i), Wbus)][t,1:S]
                end
                left[:] .+= l[i,t,:]
            end
            left[:] .-= ln[t]
            @constraint(model, left[:] .== right[:]) # the power balance equation
        end

        # generation cost
        for i in 1:N
            if i in Gbus
                index = findfirst(isequal(i), Gbus) # the index of the unit at bus i
                for t in 1:T
                    # the hot start cost of each unit
                    @constraint(model, Chs[index,t] == Chot[index] * s[index,t]) 

                    # linearization of generation cost
                    for l in 0:(L-1)
                        pil = Plow[index] + l * (Pup[index] - Plow[index]) / L
                        @constraint(model, [s=1:S], z[index,t,s] >= (2*fc[index]*pil + fb[index]) * Pg[index,t,s] + (fa[index] - fc[index]*(pil^2)) * u[index,t])
                    end

                    # the excess of hot start cost
                    taoi = Int(max(1, t-Toff[index]-Tcold[index]))
                    if (t-taoi <= 0) && (max(0, -T0[index]) < abs(t-taoi) + 1)
                        fit = 1
                    else
                        fit = 0
                    end
                    @constraint(model, Ces[index,t] >= (Ccold[index] - Chot[index]) * (s[index,t] - sum(d[index,taoi:t-1]) - fit))
                end
            end
            @constraint(model, Ccl[i,:,:] .== Cll * l[i,:,:]) # the load loss cost of each bus
        end

        @objective(model, Min, sum(sum(Chs)) + sum(sum(Ces)) + (1/S) * sum(sum(sum(z))) + (1/S) * sum(sum(sum(Ccl))))

        optimize!(model)

        obj_val = 0
        scd_obj = 0
        termstat = termination_status(model)

        if termstat in [MathOptInterface.INFEASIBLE, MathOptInterface.INFEASIBLE_OR_UNBOUNDED]
            compute_conflict!(model)
            if get_attribute(model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
                iis_model, _ = copy_conflict(model)
                print(iis_model)
            end
            obj_val = 9999
            scd_obj = 9999
            num_infeasible += 1
            if debug
                @info "Iteration $it: INFEASIBLE or INFEASIBLE_OR_UNBOUNDED objective value: $obj_val"
            end
        

        elseif termstat == MathOptInterface.OPTIMAL
            obj_val = objective_value(model)
            scd_obj = (1/S) * sum(sum(sum(value.(z)))) + (1/S) * sum(sum(sum(value.(Ccl))))
            num_feasible += 1
            if debug
                @info "Iteration $it: OPTIMAL objective value: $obj_val"
            end

        else
            if debug
                @warn "Iteration $it: Solver returned unexpected termination status: $termstat"
            end
        end

        opt_u = value.(u)

        # ---- Save iteration data ----
        my_data = JSON.parsefile(output_file, use_mmap = false)
        my_data["load$it"] = ln
        my_data["scd_obj$it"] = scd_obj
        my_data["opt_u$it"] = opt_u
        my_data["Pw$it"] = Pw_mean
        open(output_file, "w") do f
            JSON.print(f, my_data)
        end
        if debug
            @info "Data generation finished for case $case"

        end
        model = nothing
    end

    if debug
        @info "Data generation finished for case $case"

        @info "Total feasible iterations: $num_feasible"
        @info "Total infeasible iterations: $num_infeasible"
    end
end

data_generation_main()
