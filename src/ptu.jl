# Parallel Transport Unfolding (PTU)
# ---------------------------
# Budninskiy, Max, Glorian Yin, Leman Feng, Yiying Tong, and Mathieu Desbrun.
# “Parallel Transport Unfolding: A Connection-Based Manifold Learning Approach.” arXiv, November 2, 2018.
# http://arxiv.org/abs/1806.09039.


"""
    PTU{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction

The `PTU type represents a local tangent space alignment model constructed for `T` type data with a help of the `NN` nearest neighbor algorithm.
"""
struct PTU{NN <: AbstractNearestNeighbors, T<:Real} <: NonlinearDimensionalityReduction
    d::Int
    k::Real
    gauges::Array{T,3}
    adjacency_matrix::SparseMatrixCSC{T,Int64}
    proximity_graph::SimpleGraph{Int64}
    model::MDS{T}
    nearestneighbors::NN
    degree::Vector{Int64}
    component::AbstractVector{Int}
end

## properties
size(R::PTU) = (R.d, size(R.gauges, 2))
eigvals(R::PTU) = R.model.λ
neighbors(R::PTU) = R.k
vertices(R::PTU) = R.component

## show
function summary(io::IO, R::PTU)
    id, od = size(R)
    msg = isinteger(R.k) ? "neighbors" : "epsilon"
    print(io, "PTU{$(R.nearestneighbors)}(indim = $id, outdim = $od, neighbors = $(R.k))")
end

"""
    get_connection(i::Int, j::Int, Θ::AbstractArray{T,3}) where T <: Real

Get the connection from point `i` to point `j` associated with bases `Θ[:,:,i]` and
Θ[:,:,j]
"""
function get_connection(i::Int, j::Int, Θ::AbstractArray{T,3}) where T <: Real
    d,p,n = size(Θ)
    R = zeros(T, p,p)
    get_connection!(R, i,j,Θ)
    R
end

function get_connection!(R::AbstractMatrix{T}, i::Int, j::Int, Θ::AbstractArray{T,3}) where T <: Real
    d,p,n = size(Θ)
    if i == j
        return I(p)
    end
    θ0 = view(Θ,:,:,i)
    θ1 = view(Θ,:,:,j)
    get_connection!(R, θ0,θ1)
end

function get_connection!(R::AbstractMatrix{T}, θi::AbstractMatrix{T}, θj::AbstractMatrix{T}) where T <: Real
    ss = svd(θi'*θj)
    mul!(R, ss.U, ss.Vt)
end

function get_basis!(B::AbstractMatrix{T}, X::AbstractMatrix{T}, i::T2, NI::Vector{T2}) where T <: Real where T2 <: Integer
    d,p = size(B)
    VX = view(X, :, NI)
    δ_x = VX .- view(X, :, i:i)

    ss = svd(δ_x;full=true)
    Up = standardize_basis(ss.U)
    B .= Up[:,1:p]
end


## interface functions
"""
    fit(PTU, data; k=12, maxoutdim=2, nntype=BruteForce)

Fit a local tangent space alignment model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `k`: a number of nearest neighbors for construction of local subspace representation
* `maxoutdim`: a dimension of the reduced space.
* `nntype`: a nearest neighbor construction class (derived from `AbstractNearestNeighbors`)

# Examples
```julia
M = fit(PTU, rand(3,100)) # construct PTU model
R = transform(M)           # perform dimensionality reduction
```
"""
function fit(::Type{PTU}, X::AbstractMatrix{T};
        k::Real=12, K=k, maxoutdim::Int=2, tangentdim=min(size(X,1)-1,k), nntype=BruteForce,debug=false) where {T<:Real}
    # Construct NN graph
    d, n = size(X)
    # construct orthognoal basis
    B = zeros(T, d,tangentdim,n)

    NN = fit(nntype, X)
    E, _ = adjacency_list(NN, X, K)

    A = adjacency_matrix(NN, X, k)
    G, C2 = largest_component(SimpleGraph(A))
    Ac2 = A[C2,C2]

    ΔTsb = 0.0
    prog1 = Progress(n, 1.0, "Constructiong bases...") 
    for i in 1:n
        NI = E[i] # neighbor's indexes

        l = length(NI)
        l == 0 && continue # skip
        t0 = time()
        get_basis!(view(B, :,:,i), X, i, NI)
        ΔTsb += time() - t0
        # re-center points in neighborhood
        #VX = view(X, :, NI)
        #δ_x = VX .- view(X, :, i:i)

        # Compute orthogonal basis H of θ'
        #ss = svd(δ_x;full=true)
        #t0 = time()
        #Up = standardize_basis(ss.U)
        #ΔTsb += time() - t0
        #B[:,:,i] = Up[:,1:maxoutdim]
        next!(prog1)
    end
    n = length(C2)
    # compute shortest path for every point
    DD = zeros(T,n,n)
    R = diagm(ones(T, tangentdim))
    R2 = similar(R)
    R3 = similar(R)
    #θ = zeros(T, d,maxoutdim)
    if debug
        V = zeros(T, d)
    else
        V = zeros(T, tangentdim)
    end
    # debug timing info
    ΔTdj = 0.0
    ΔTp = 0.0
    ΔT0 = 0.0
    ΔT1 = 0.0
    ΔT2 = 0.0
    ΔT3 = 0.0
    nt = 0

    # temporary variables
    Rq = fill(0.0, tangentdim, tangentdim, n,n)
    qq = fill(false, n,n)
    θp = fill(0.0, tangentdim, d)
    ΔX = fill(0.0, d)
    v = fill(0.0, tangentdim)

    prog2 = Progress(n; dt=1.0, desc="Computing geodesics...")
    for i in 1:n
        t0 = time()
        dj = dijkstra_shortest_paths(G,i,Ac2;trackvertices=true)
        ΔTdj += time() - t0
        # modified geodesic distance
        # TODO: modify the dijkstra algorithm directly)
        for k in 2:n
            # get the path from i to closest_vertices[k]
            kn = dj.closest_vertices[k]
            fill!(V,zero(T))
            R .= I(tangentdim)
            jp = i
            θ0 = view(B,:,:,C2[jp])
            t0 = time()
            path = get_path(dj, kn)
            ΔTp += time() - t0
            for l in 2:length(path)
                j1 = path[l-1]
                j2 = path[l]
                #ss = svd(θ'*Θ[:,:,j1])
                if debug
                    # no projection
                    V .+= X[:,C2[j2]]-X[:,C2[j1]]
                    DD[i,kn] += norm(X[:,C2[j2]]-X[:,C2[j1]])
                else
                    θ1 = view(B,:,:,C2[j1])
                    if sum(θ1 .!= 0.0) == 0
                        @show i,l,j1,j2,kn
                        error("All zero gauge")
                    end
                    # check if we have already computed this alignment
                    if qq[jp,j1] == false
                        t0 = time()
                        #ss = svd(θ0'*θ1)
                        #R .= R*(ss.V*ss.U')'
                        nt += 1
                        #Rq[:,:, jp,j1] = ss.U*ss.Vt
                        get_connection!(view(Rq, :,:,jp,j1), θ0,θ1)
                        ΔT1 += time() - t0
                        qq[jp,j1] = true
                    end
                    t0 = time()
                    #mul!(R2, R, view(Rq, :,:,jp,j1))
                    #R2 .= Rq[:,:,jp,j1]
                    copy!(R2, view(Rq,:,:,jp,j1))
                    #R2 = view(Rq,:,:,jp,j1)
                    ΔT0 += time() - t0
                    t0 = time()
                    #R .= R*view(Rq,:,:, jp,j1)
                    #R .= R*R2
                    mul!(R3, R, R2)
                    copy!(R,R3)
                    ΔT2 += time() - t0
                    t0 = time()
                    copy!(ΔX, view(X, :, C2[j2]))
                    ΔX .-= view(X, :, C2[j1])
                    mul!(θp, R, θ1')
                    mul!(v, θp, ΔX)
                    #v = R*θ1'*ΔX
                    ΔT3 += time() - t0
                    V .+= v
                    θ0 = θ1
                    jp = j1
                end
            end
            #DD[i,kn] = sum(abs2, V)
            if debug == false
                DD[i,kn] = norm(V)
            else
                #DD[i,kn] = sqrt(DD[i,kn])
            end
        end
        next!(prog2)
    end
    # broadcast!(x->-x*x/2, DD, DD)
    #symmetrize!(DD) # error in MvStats
    @show ΔT0, ΔT1, ΔT2, ΔT3, ΔTdj, ΔTp, ΔTsb, nt
    if debug
        broadcast!(x->-x*x/2, DD, DD)
        DD = (DD+DD')/2
        M = fit(KernelPCA, DD, kernel=nothing, maxoutdim=maxoutdim)
        return Isomap{nntype}(d, k, M, NN, C)
    else
        DD = (DD+DD')/2
        M = fit(MDS, DD, distances=true, maxoutdim=maxoutdim)
        return PTU{nntype,T}(d, k,B, A,G,M, NN, degree(G), C2)
    end
end

"""
    predict(R::PTU)

Transforms the data fitted to the local tangent space alignment model `R` into a reduced space representation.
"""
predict(R::PTU) = predict(R.model)

