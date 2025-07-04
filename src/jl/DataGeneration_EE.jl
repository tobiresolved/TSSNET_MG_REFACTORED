# randomly generate unit commitment firstly
# solving tranditional two-stage stochastic optimization problem to get the optimal second stage objective value
# all uncertainty scenarios across a mean-aggregated action to get a represented scenario
# by ying yang, done: 2024-06-20, add forbidden constraints by 2024-06-25


# 1. Load necessary packages and modules
include("ReadData.jl")
include("ReadWind.jl")
include("NodeY.jl")

if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end
using JuMP, MosekTools, Gurobi
using MathOptInterface
using JSON
using DataFrames
using Statistics
using Flux
using Flux: train!, ADAM, mse, params
using BSON: @save, @load

FileName = "../../data/SCUC6.txt"
script_dir = @__DIR__  # directory where this script lives
FileName = joinpath(script_dir, "..", "..", "data", "SCUC6.txt")
FileName_wind = "../../data/Wind_power.xlsx"
script_dir = @__DIR__  # directory where this script lives
FileName_wind = joinpath(script_dir, "..", "..", "data", "Wind_power_flc.CSV")

data = ReadData(FileName) # read data from the file
Wind_data = ReadWind(data,FileName_wind) # read wind data from the file

# define the parameters of units and buses
T = length(data.LT.PD) # the number of periods
# T = 12 # try 12 periods firstly
G = data.BP.unitN # the number of units
N = data.BP.busN # the number of buses
W = data.WD.W # the number of wind farms
S_w = length(Wind_data[1][1,:]) # the number of wind power at each wind farm
S = 50 # try 10 scenarios firstly
Gbus = [unit.busG for unit in data.UP] # the bus number of each unit
Dbus = [bus.N for bus in data.LN] # the bus number of each bus
Wbus = data.WD.WN # the bus number of each wind farm
Pramp = [unit.ramp for unit in data.UP] # the ramping constraint of each unit, assume up and down ramping constraints are the same
Pstart = [unit.PG_up * 0.5 for unit in data.UP] # the start ramp constraint of each unit, set to 50% of the upper bound of active generation
Pshut = [unit.PG_up for unit in data.UP] # the shut ramp constraint of each unit, set to the upper bound of active generation
Pup = [unit.PG_up for unit in data.UP] # the upper bound of active generation of each unit
Plow = [unit.PG_low for unit in data.UP] # the lower bound of active generation of each unit
Ton = [unit.T_on for unit in data.UP] # the minimum on time of each unit
Toff = [unit.T_off for unit in data.UP] # the minimum off time of each unit
L = 4 # constant coefficient of linearization of generation cost
fa = [unit.Calpha for unit in data.UP] # the coefficient of the constant term of the cost function of each unit
fb = [unit.Cbeta for unit in data.UP] # the coefficient of the linear term of the cost function of each unit
fc = [unit.Cgamma for unit in data.UP] # the coefficient of the quadratic term of the cost function of each unit
Ccold = [unit.StartCost * 2 for unit in data.UP] # the start cost of each unit, set to 50 times of the original start cost
Chot = [unit.StartCost for unit in data.UP] # the hot start cost of each unit
Cll = 250 # the load loss cost
Tcold = min.(max.(Toff,ones(G)),T) # the cold start time of each unit
Lup = 0.26 * ones(N) # the upper bound of load loss
Llow = 0 * ones(N) # the lower bound of load loss
LN_ori = data.LN # the load data of each bus

# construct direct current (DC) power flow coefficient matrix
type_of_pf = "DC"
GM,BM = NodeY(data, type_of_pf) # the conductance matrix and susceptance matrix of the DC power flow model

u0 = vec(zeros(G,1)) # the initial status of each unit, assume all units are off at the beginning
T0 = vec(zeros(G,1)) # number of periods units have been on or off 
P0 = vec(zeros(G,1)) # the initial generation of each unit
U0 = vec(zeros(G,1)) # number of periods that units should be on from the first period of current time framework
L0 = vec(zeros(G,1)) # number of periods that units should be off from the first period of current time framework
tao0 = Toff .+ Tcold .+1 # the initial cold start time of each unit
for i = 1:G
    mid_u = min(T,u0[i] * (Ton[i] - T0[i]))
    mid_l = min(T,(1-u0[i]) * (Toff[i] + T0[i]))
    U0[i] = max(0,mid_u)
    L0[i] = max(0,mid_l)
end

# construct the branch and transformer data
allbranch_start = [[branch.fromBus for branch in data.BR]; [transformer.fromBus for transformer in data.TP]] # the start point of all branches and transformers
allbranch_end = [[branch.toBus for branch in data.BR]; [transformer.toBus for transformer in data.TP]] # the end point of all branches and transformers
allbranch_P = [[branch.P_max for branch in data.BR]; [transformer.P_max for transformer in data.TP]] # the maximum transmission capacity of all branches and transformers
B = size(allbranch_start,1) # the number of branches and transformers

# set wind generation scenarios
Wdata_3D = Array{Float64,3}(undef, 12, T, S_w) # convert wind data to 3D array
for i in 1:12
    Wdata_3D[i,:,:] = Wind_data[i][1:T,:]
end

iteration_num = 5 # define the number of iterations

# save some parameters to a json file, for transmitting to python
script_dir = @__DIR__  # directory where this script lives
rel_filename = joinpath(script_dir, "..", "..", "results", "ram_case_A.json")

open(rel_filename, "w") do f
    JSON.print(f, Dict(
        "IN" => iteration_num,
        )
    )
end

# define the function for adding forbidden constraints
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

# load bson file
script_dir = @__DIR__  # directory where this script lives
autoencoder = joinpath(script_dir, "..", "..", "models", "auto_encoder.bson")
@load autoencoder encoder

# looply solve iteration_num's optimization problem
# save the results to a json file, for transmitting to python
for it in 1:iteration_num 
    # construct the model
    # model = Model(Mosek.Optimizer)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 1) # suppress the output of the solver

    # randomly select wind power scenarios
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

    LN = Vector{Vector{Float64}}(undef, length(LN_ori)) # Initialize LN
    ln = Array{Float64}(undef, T, 1)
    # fluctuating the load requirement
    # Generate random fluctuations based on normal distribution
    for i = 1:length(Dbus)
        fluctuations = (rand(length(LN_ori[i].PD)) .- 0.5) .* 2 .* 0.05
        fluctuated_vector = LN_ori[i].PD .+ fluctuations
        LN[i] = fluctuated_vector .+ (LN_ori[i].PD .- mean(fluctuated_vector))
        ln = ln + LN[i]
    end

    # define continuous variables
    @variable(model, Pg[1:G,1:T,1:S]) # generation of each unit G*T*S
    @variable(model, theta[1:N,1:T,1:S]) # phase angle of each bus
    @variable(model, l[1:N,1:T,1:S]) # load loss of each bus
    # @variable(model, Pw[1:W,1:T]) # wind power of each wind farm, in a certain scenario, we assume the wind power is known
    @variable(model, z[1:G,1:T,1:S]) # auxiliary variable for linearization of generation cost
    @variable(model, Ccl[1:N,1:T,1:S]) # the load loss cost of each bus

    # hot start cost and excess of hot start cost are not related to the scenarios
    @variable(model, Chs[1:G,1:T]) # the hot start cost of each unit
    @variable(model, Ces[1:G,1:T] >= 0) # the excess of hot start cost of each unit

    @constraint(model, theta[1,:,:] .== 0) # set first bus as reference bus, with phase angle 0

    # define binary variables
    @variable(model, a[1:G,1:T,1:it], Bin) # the forbidden status of each unit
    @variable(model, u[1:G,1:T], Bin) # the status of each unit
    if it == 1
        u = zeros(G,T)
    else
        u = rand(0:1,G,T) # randomly generate the status of each unit
        for i in 1:G # set the feasible status of each unit
            for t in 1:(Int(U0[i]+L0[i]))
                u[i,t] == u0[i] 
            end
    end
    end
    @variable(model, s[1:G,1:T], Bin) # the start status of each unit
    @variable(model, d[1:G,1:T], Bin) # the shut status of each unit

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

    # branch constraints
    for b in 1:B
        i = allbranch_start[b]
        j = allbranch_end[b]
        x_cb = abs(1/BM[i,j])
        @constraint(model, theta[i,:,:] - theta[j,:,:] .<= x_cb * allbranch_P[b]) # the upper bound of branch flow
        @constraint(model, -allbranch_P[b] * x_cb .<= theta[i,:,:] - theta[j,:,:]) # the lower bound of branch flow
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
            @constraint(model, l[i,t,:] .<= Lup[i]) # the upper bound of load loss

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

    @objective(model, Min, sum(sum(Chs)) + sum(sum(Ces)) + (1/S) * sum(sum(sum(z))) + (1/S) * sum(sum(sum(Ccl)))) # the objective function

    # JuMP.write_to_file(model, "TS_SP_Tranditional.lp")
    optimize!(model)

    obj_val = 0
    scd_obj = 0
    if termination_status(model) == MathOptInterface.INFEASIBLE_OR_UNBOUNDED
        # println("$it iteration is infeasible.")
        obj_val = 9999
        scd_obj = 9999
        println(termination_status(model))
    else
        obj_val = objective_value(model)
        scd_obj = (1/S) * sum(sum(sum(value.(z)))) + (1/S) * sum(sum(sum(value.(Ccl))))
        println("Optimal objective value: ", obj_val)
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("$it iteration's Objective value: ", obj_val)
    end

    # println("$it iteration's Objective value: ", obj_val)
    # println("Unit status: ", value.(u))

    # record the results for nnt training
    # println("scd_obj: ", scd_obj)
    # wind_his = [Pw[i] for i in 1:W] # in this work, we dont focus on wind power -> cost
    opt_u = value.(u)
    # save current iteration's results to the json file
    my_data = JSON.parsefile(rel_filename, use_mmap = false)
    my_data["load$it"] = ln
    my_data["scd_obj$it"] = scd_obj
    my_data["opt_u$it"] = opt_u
    # my_data["Pw_mean$it"] = Pw_mean
    my_data["Pw$it"] = Pw_encoder
    open(rel_filename, "w") do f
        JSON.print(f, my_data)
    end

    # delete the model, so next iteration can create a new model
    model = nothing
end

# continuous variables are not important when solve the first stage problem
# they are decided when uncertainty scenarios are known
# println("Generation of each unit: ", value.(Pg))
# println("phase angle of each bus: ", value.(theta))
# println("load loss of each bus: ", value.(l))

# # Retrieve the list of all constraints
# constraints_list = all_constraints(model, include_variable_in_set_constraints = true)
# # Print each constraint
# for (index, constr) in enumerate(constraints_list)
#     println("Constraint $index: ", constr)
# end