# read the data from the file and construct the UCdata
# modular & debug version, for TSSNET
# by ying yang, 2024-06-18

include("MgParameters.jl")

using .MgParameters
if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end
using .MgParameters
using StructIO
using Logging    # <--- Logging eingebunden

"""
    check_zero!(data::Array{String,1})

Deletes lines at the beginning of `data` which start with "0".
"""
function check_zero!(data::Array{String,1})
    while !isempty(data) && startswith(data[1], "0")
        deleteat!(data, 1)
    end
end

"""
    read_float_or_int(num::String) -> Real

Parses a number as Float64 if it contains a decimal, otherwise as Int.
"""
function read_float_or_int(s::AbstractString)
    if contains(s, ".")
        return parse(Float64, s)
    else
        return parse(Int, s)
    end
end

"""
    parse_group!(data, n, parse_fun)

Reads and parses `n` lines from `data` using `parse_fun`, deleting lines as they're read.
Returns a vector of parsed values.
"""
function parse_group!(data, n, parse_fun)
    result = []
    for i in 1:n
        push!(result, parse_fun(data[1]))
        deleteat!(data, 1)
    end
    check_zero!(data)
    return result
end

"""
    ReadData(FileName::String; debug::Bool=false) -> UCdata

Reads system data from a file and returns a UCdata struct.
If `debug=true`, prints shapes and key parameters for diagnostics.
"""
function ReadData(FileName::String; debug::Bool=false)
    file = open(FileName, "r")
    data = readlines(file)

    # 1. Base parameters
    FirstLine = [read_float_or_int(num) for num in split(data[1])]
    busN, branchN, balanceBus, standardP, iterationMax, centerParameter, unitN =
        FirstLine[1], FirstLine[2], FirstLine[3], FirstLine[4], FirstLine[5], FirstLine[6], FirstLine[7]
    bp = MgParameters.baseparameters(busN, branchN, balanceBus, standardP, iterationMax, centerParameter, unitN)
    deleteat!(data, 1)

    # 2. Units
    check_zero!(data)
    ups = []
    for i in 1:bp.unitN
        unitData = [read_float_or_int(num) for num in split(data[1])]
        unitID, busG, Calpha, Cbeta, Cgamma, PG_up, PG_low, QG_up, QG_low, T_off, T_on, U_init, ramp, StartCost =
            unitData[1], unitData[2], unitData[3] / bp.standardP, unitData[4], unitData[5],
            unitData[6] / bp.standardP, unitData[7] / bp.standardP,
            unitData[8] / bp.standardP, unitData[9] / bp.standardP,
            unitData[10], unitData[11], unitData[12], unitData[13] / bp.standardP, unitData[14]
        push!(ups, MgParameters.unitparameters(unitID, busG, Calpha, Cbeta, Cgamma, PG_up, PG_low, QG_up, QG_low, T_off, T_on, U_init, ramp, StartCost))
        deleteat!(data, 1)
    end
    check_zero!(data)

    # 3. Buses
    bvs = []
    for i in 1:bp.busN
        busData = [read_float_or_int(num) for num in split(data[1])]
        busID, busV_up, busV_low = busData[1], busData[2], busData[3]
        push!(bvs, MgParameters.busvaltage(busID, busV_up, busV_low))
        deleteat!(data, 1)
    end
    check_zero!(data)

    # 4. Branches
    brs = []
    while !startswith(data[1], "0")
        branchData = [read_float_or_int(num) for num in split(data[1])]
        branchID, fromBus, toBus, R, X, B, P_max =
            branchData[1], branchData[2], branchData[3], branchData[5], branchData[6], branchData[7], branchData[8] / bp.standardP
        push!(brs, MgParameters.branchparameters(branchID, fromBus, toBus, R, X, B, P_max))
        deleteat!(data, 1)
    end
    check_zero!(data)

    # 5. Transformers
    tps = []
    while !startswith(data[1], "0")
        transformerData = [read_float_or_int(num) for num in split(data[1])]
        transformerID, fromBus, toBus, R, X, P_max, K =
            transformerData[1], transformerData[2], transformerData[3], transformerData[5], transformerData[6],
            transformerData[8] / bp.standardP, transformerData[9]
        push!(tps, MgParameters.transformerparameters(transformerID, fromBus, toBus, R, X, K, P_max))
        deleteat!(data, 1)
    end
    check_zero!(data)

    # 6. Total load per period (LT)
    PD, QD = Float64[], Float64[]
    while !startswith(data[1], "0")
        loadData = [read_float_or_int(num) for num in split(data[1])]
        push!(PD, loadData[2] / bp.standardP)
        push!(QD, loadData[3] / bp.standardP)
        deleteat!(data, 1)
    end
    lt = MgParameters.loadT(PD, QD)
    check_zero!(data)

    # 7. Loads at each bus (LN)
    ln, PD, QD, lnN = [], Float64[], Float64[], []
    while !startswith(data[1], "0")
        loadNData = [read_float_or_int(num) for num in split(data[1])]
        push!(lnN, loadNData[1])
        push!(PD, loadNData[2] / bp.standardP)
        push!(QD, loadNData[3] / bp.standardP)
        deleteat!(data, 1)
    end
    blN, sum_P, sum_Q = length(lnN), sum(PD), sum(QD)
    P_factor, Q_factor = PD / sum_P, QD / sum_Q
    for i in 1:blN
        PDA = lt.PD * P_factor[i]
        QDA = lt.QD * Q_factor[i]
        push!(ln, MgParameters.loadN(lnN[i], PDA, QDA))
    end

    # 8. Wind nodes (case-dependent)
    if occursin("SCUC6", FileName)
        W, WN = 2, [1,3]
    elseif occursin("SCUC30", FileName)
        W, WN = 3, [3,8,10]
    elseif occursin("SCUC118", FileName)
        W, WN = 6, [3,7,60,72,80,91]
    else
        error("Unknown system for wind nodes")
    end
    wd = MgParameters.wind(W, WN)

    # -- Debug Logging statt Print
    if debug
        @info "System: $busN buses, $branchN branches, $unitN units, wind: $W at $WN"
        @info "Branches: $(length(brs)), Transformers: $(length(tps)), Loads: $(length(ln))"
        @info "Periods: $(length(lt.PD)), Load samples: $(length(ln[1].PD)) per bus"
    end

    # 9. Assemble everything
    ucdata = MgParameters.UCdata(bp, ups, bvs, brs, tps, lt, ln, wd)
    return ucdata
end
