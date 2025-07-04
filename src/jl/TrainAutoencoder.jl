using Flux
using Flux: mse, train!
using BSON: @save
using Statistics
using MLUtils: DataLoader
using Random
using Logging
using Dates

include("../../config/case_config.jl")
using .TSSNETConfigModule

include("ReadWind.jl")
include("ReadData.jl")

function main()
    # --- Logger setup ---
    global_logger(ConsoleLogger(stderr, Logging.Info))

    # --- Select case and load config ---
    case = length(ARGS) > 0 ? ARGS[1] : "B"
    conf = get_case_config(case)

    @info "Loaded configuration for case: $case"

    # --- File paths ---
    autoencoder_file = conf.autoencoder_file
    data_file = conf.data_file
    wind_file = conf.wind_file

    # --- Load data ---
    data = ReadData(data_file)
    Wind_data = ReadWind(data, wind_file)

    T = length(data.LT.PD)            # Time steps (e.g., 24)
    W = data.WD.W                     # Number of wind farms
    S_w = size(Wind_data[1], 2)       # Number of wind scenarios per source
    S = 50                            # Number of scenarios per sample
    num_samples = 1500                # Training sample size

    @info "Preparing training data tensor with $num_samples samples"

    # --- Build wind scenario tensor: shape (12, T, S_w) ---
    Wdata_3D = Array{Float64,3}(undef, 12, T, S_w)
    for i in 1:12
        Wdata_3D[i, :, :] = Wind_data[i][1:T, :]
    end

    data_tensor = Array{Float32,3}(undef, num_samples, S, T)
    for yy in 1:num_samples
        Pw = []
        for _ in 1:W
            tmp = Array{Float32}(undef, T, S)
            for j in 1:S
                index = rand(1:12)
                index2 = rand(1:S_w)
                tmp[:, j] = Float32.(Wdata_3D[index, :, index2])
            end
            push!(Pw, tmp')
        end
        data_tensor[yy, :, :] = Pw[1]  # Only use first wind farm
    end

    samples = [data_tensor[i, :, :] for i in 1:num_samples]
    train_loader = DataLoader(samples, batchsize=32, shuffle=true, collate=identity)

    @info "DataLoader initialized with batch size = 32"

    # --- Define encoder and decoder ---
    encoder = Chain(
        x -> reshape(x, 50 * 24),
        Dense(50 * 24, 256, relu),
        Dense(256, 128, relu),
        Dense(128, 24, relu)
    )

    decoder = Chain(
        Dense(24, 128, relu),
        Dense(128, 256, relu),
        Dense(256, 50 * 24),
        x -> reshape(x, 50, 24)
    )

    autoencoder = Chain(encoder, decoder)
    opt = Flux.setup(Adam(), autoencoder)

    loss_fn(model, batch) = mean(mse.(model.(batch), batch))

    # --- Training loop ---
    epochs = 20
    for epoch in 1:epochs
        start_time = now()
        train!(loss_fn, autoencoder, train_loader, opt)
        elapsed = now() - start_time
        example_latent = size(encoder(samples[1]))
        @info "Epoch $epoch complete"
    end

    # --- Save encoder ---

    @save "model_weights.bson" encoder_weights=Flux.trainable(encoder)

    # --- Test encoding example ---
    test_input = rand(Float32, 50, 24)
    latent = encoder(test_input)
    @info "Test encoding shape: $(size(latent))"
end

main()
