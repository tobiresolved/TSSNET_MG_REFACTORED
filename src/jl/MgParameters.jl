# define the data structure
# by ying yang 2024-05-07
module MgParameters
using StructIO
using Logging

export baseparameters, unitparameters, busvaltage, branchparameters, transformerparameters, loadT, loadN, wind, UCdata

@io struct baseparameters
    busN::Int # number of buses
    branchN::Int # number of branches
    balanceBus::Int # the bus number of the balance bus
    standardP::Float64 # the standard coefficient of the system
    iterationMax::Int # the maximum number of iterations
    centerParameter::Float64 # the center parameter
    unitN::Int # the number of units
end

@io struct unitparameters
# @io mutable struct unitparameters # in julia, struct is immutable by default
    unitID::Int # the ID of the unit
    busG::Int # the bus number of the unit
    Calpha::Float64 # the coefficient of the constant term of the cost function
    Cbeta::Float64 # the coefficient of the linear term of the cost function
    Cgamma::Float64 # the coefficient of the quadratic term of the cost function
    PG_up::Float64 # the upper bound of active generation
    PG_low::Float64 # the lower bound of active generation
    QG_up::Float64 # the upper bound of reactive generation
    QG_low::Float64 # the lower bound of reactive generation
    T_off::Int # the minimum off time
    T_on::Int # the minimum on time
    U_init::Int # the mimimum time of keeping initial status
    ramp::Float64 # the constraint of ramping generation
    StartCost::Float64 # the start cost
    # unitparameters() = new() # default constructor
end

@io struct busvaltage
    busID::Int # the ID of the bus
    busV_up::Float64 # the upper bound of bus voltage
    busV_low::Float64 # the lower bound of bus voltage
end

@io mutable struct branchparameters
    branchID::Int # the ID of the branch
    fromBus::Int # the bus number of the start point of the branch
    toBus::Int # the bus number of the end point of the branch
    R::Float64 # the resistance of the branch
    X::Float64 # the reactance of the branch
    B::Float64 # the susceptance of the branch
    P_max::Float64 # the maximum transmission capacity of the branch
end

@io mutable struct transformerparameters
    transformerID::Int # the ID of the transformer
    fromBus::Int # the bus number of the start point of the transformer
    toBus::Int # the bus number of the end point of the transformer
    R::Float64 # the resistance of the transformer
    X::Float64 # the reactance of the transformer
    K::Float64 # the rating of the transformer
    P_max::Float64 # the maximum transmission capacity of the transformer
end

@io struct loadT 
    PD::Array{Float64,1} # the active power demand
    QD::Array{Float64,1} # the reactive power demand
end

@io struct loadN
    N::Int # the number of load buses
    PD::Array{Float64,1} # the active power demand of each load bus
    QD::Array{Float64,1} # the reactive power demand of each load bus
end

@io struct wind
    W::Int # the number of wind buses
    WN::Vector{Int} # the bus number of each wind bus
end

@io struct UCdata
    BP::baseparameters # the base parameters of the system
    UP::Vector{unitparameters} # the parameters of units
    BV::Vector{busvaltage} # the parameters of buses
    BR::Vector{branchparameters} # the parameters of branches
    TP::Vector{transformerparameters} # the parameters of transformers
    LT::loadT # the total load
    LN::Vector{loadN} # the load of each bus
    WD::wind # the wind data
end


end #module