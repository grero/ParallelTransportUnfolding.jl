function cknn(NN::BruteForce{T}, X::AbstractVecOrMat{T}, k::Integer,
             δ::T; weights=false, kwargs...) where T<:Real

    n = size(X,2)
    D = pairwise((x,y)->norm(x-y), eachcol(NN.fitted), eachcol(X))
    Q = fill(0.0, n)
    QI = fill(0, n)
    #get the k-th nearest neighbour distance for each point
    @inbounds for (j, ds) in enumerate(eachcol(D))
        ii = sortperm(ds)[k]
        Q[j] = ds[ii]
        QI[j] = ii
    end
    A = Vector{Vector{Int}}(undef, n)
    W = Vector{Vector{T}}(undef, (weights ? n : 0))
    @inbounds for (j, ds) in enumerate(eachcol(D))
        A[j] = findall(ds .< δ*sqrt.(Q[j].*Q))
        if weights
            W[j] = D[A[j], j]
        end
    end
    A,W
end

function ManifoldLearning.adjacency_list(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer, δ::Real;weights::Bool=false, kwargs...) where T <: Real
    A,W = cknn(NN, X, k, δ;weights=weights, kwargs...)
end

function ManifoldLearning.adjacency_matrix(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T},
                           k::Integer, δ::Real; symmetric::Bool=true, kwargs...) where T<:Real
    n = size(NN)[2]
    m = length(eachcol(X))
    @assert n >=m "Cannot construc matrix for more then $n fitted points"
    E, W = cknn(NN, X, k,δ; weights=true, kwargs...)
    return ManifoldLearning.sparse(E, W, n, symmetric=symmetric)
end

function get_path(dj::DijkstraState, v::Integer)
    u = v
    path = [u]
    while dj.parents[u] != 0
        u = dj.parents[u]
        pushfirst!(path, u)
    end
    return path
end

function torus(n::Int=1000, noise::Real=0.03; segments=1, rng::AbstractRNG=default_rng())
    θ = 2π*rand(rng, n)
    ϕ = 2π*rand(rng, n)
    rs = 0.1
    rb = 0.4
    X = fill(0.0, 3, n)
    R = (rb .+ rs.*cos.(ϕ))
    X[1,:] = R.*cos.(θ)
    X[2,:] = R.*sin.(θ)
    X[3,:] = rs.*sin.(ϕ)
    X .+ noise*randn(3,n)
    mn,mx = extrema(R)
    labels = segments == 0 ? R : round.(Int, (R.-mn)./(mx-mn).*(segments-1))
    X,labels
end
