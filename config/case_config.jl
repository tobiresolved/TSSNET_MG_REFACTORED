module TSSNETConfigModule

export TSSNETConfig, get_case_config

struct TSSNETConfig
    autoencoder_file::String
    data_file::String
    wind_file::String
    output_file_opt::String
    output_file_ram::String
    iteration_num::Int
    Lup::Float64
    G::Union{Int, Nothing}
    debug::Union{Bool, Nothing}
end

const DEFAULT_AUTOENCODER_FILE = "models/auto_encoder.bson"


const CASE_CONFIGS = Dict(
    "A" => TSSNETConfig(DEFAULT_AUTOENCODER_FILE, "data/SCUC6.txt",    "data/Wind_power_flc.csv", "results/opt_case_A.json","results/ram_case_A.json", 50,  0.26, nothing, true),
    "B" => TSSNETConfig(DEFAULT_AUTOENCODER_FILE, "data/SCUC30.txt",   "data/Wind_power_flc.csv", "results/opt_case_B.json","results/ram_case_B.json", 10,  0.26,  nothing, true),
    "C" => TSSNETConfig(DEFAULT_AUTOENCODER_FILE, "data/SCUC118.txt",  "data/Wind_power_flc.csv", "results/opt_case_C.json","results/ram_case_C.json", 20,  0.65,     20, true)
)

function get_case_config(case::AbstractString)
    haskey(CASE_CONFIGS, case) || error("Case '$case' not found in TSSNET_Config.")
    return CASE_CONFIGS[case]
end

end # module
