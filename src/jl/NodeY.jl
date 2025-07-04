# calculate the nodal admittance matrix Y
# by ying yang 2024-05-09
using SparseArrays

function NodeY(data, type)
    if type == "DC"
        for i in data.BR
            i.R = 0 
        end
        for i in data.TP
            i.R = 0
        end
    end

    # read data and stored as vector or matrix
    n = data.BP.busN
    # branch data
    BfromBus = [branch.fromBus for branch in data.BR]
    BtoBus = [branch.toBus for branch in data.BR]
    B_R = [branch.R for branch in data.BR]
    B_X = [branch.X for branch in data.BR]
    B_B = [branch.B for branch in data.BR]
    # transformer data 
    TPfromBus = [transformer.fromBus for transformer in data.TP]
    TPtoBus = [transformer.toBus for transformer in data.TP]
    TP_R = [transformer.R for transformer in data.TP]
    TP_X = [transformer.X for transformer in data.TP]
    TP_K = [transformer.K for transformer in data.TP]

    # construct the admittance matrix for branches
    Y1 = 1 ./ (B_R+ B_X * 1im)
    Y1 = sparse(Y1) # sparsing the matrix
    Y11 = sparse(BfromBus, BtoBus, Y1, n, n)
    branchYij = sparse(-Y11 - permutedims(Y11)) # the off-diagonal elements of the branch admittance matrix

    Ya = sparse(BfromBus, BtoBus, 1im * B_B, n, n)
    Yc = sparse(Ya+Ya')
    branchYii = spdiagm(0 => vec(sum(-branchYij,dims=1)+sum(Yc,dims=1))) # the diagonal elements of the branch admittance matrix
    branchY = sparse(branchYij + branchYii) # the branch admittance matrix

    # # construct the admittance matrix for transformers
    Y2 = sparse(1 ./ (TP_R + TP_X * 1im))
    Y22 = sparse(TPfromBus, TPtoBus, -TP_K .* Y2, n, n)
    transformerYij = sparse(Y22 + permutedims(Y22)) # the off-diagonal elements of the transformer admittance matrix
    transformerYii = sparse(TPfromBus, TPfromBus, (TP_K.^2).*(Y2), n, n)
    transformerYjj = sparse(TPtoBus, TPtoBus, Y2, n, n)
    transformerY = sparse(transformerYij + transformerYii + transformerYjj) # the transformer admittance matrix
    YY = sparse(branchY + transformerY) # the nodal admittance matrix
    
    G = real(YY) # the conductance matrix
    B = imag(YY) # the susceptance matrix
    return G, -B
end