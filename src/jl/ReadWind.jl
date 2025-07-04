using CSV, DataFrames, Logging

function ReadWind(data, FileName; debug=false)
    windN = data.WD.W
    standardW = 0.0003

    # Read the CSV file (assumes semicolon separator and comma as decimal mark)
    wind_raw = CSV.read(FileName, DataFrame; delim=';', decimal=',', ignorerepeated=true, silencewarnings=true)
    # Remove the index column (usually the first one)
    wind = select(wind_raw, Not(1))
    # Replace missing values with 1.0
    wind = coalesce.(wind, 1.0)
    # Strip whitespace from column names
    rename!(wind, names(wind) .=> [strip(col) for col in names(wind)])

    # Ensure all values are Float64
    for col in names(wind)
        if !(eltype(wind[!, col]) <: Float64)
            wind[!, col] = Float64.(wind[!, col])
        end
    end

    # Standardize and threshold small values
    wind = round.(standardW .* wind, sigdigits=3)
    wind = ifelse.(wind .< 0.001, 0, wind)

    # Prepare output: for each month, reshape data to (24, n_days)
    hw = []
    for i = 1:12
        push!(hw, reshape(wind[:,i], 24, div(length(wind[:,i]), 24)))
    end

    if debug
        @info "WindData (CSV): $(size(wind,1)) rows (hours), $(size(wind,2)) months"
        # @info "Example for January: $(hw[1])"
    end

    return hw  # Returns Vector{Matrix{Float64}}: one (24, n_days) matrix per month
end
