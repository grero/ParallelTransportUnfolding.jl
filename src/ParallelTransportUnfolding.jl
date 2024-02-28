module ParallelTransportUnfolding
import Base: size, show, summary
import StatsAPI: fit, predict, pairwise, pairwise!
import ManifoldLearning
using ManifoldLearning: adjacency_list, adjacency_matrix, largest_component, AbstractNearestNeighbors, NonlinearDimensionalityReduction,BruteForce 
using Graphs: nv, add_edge!, connected_components, dijkstra_shortest_paths,
                  induced_subgraph, SimpleGraph, DijkstraState
using MultivariateStats: NonlinearDimensionalityReduction, KernelPCA,
                             dmat2gram, gram2dmat, transform!, projection,
                             symmetrize!, PCA, MDS

using Random: AbstractRNG, default_rng
using StandardBasisVectors 
using LinearAlgebra

export PTU, fit, predict
include("utils.jl")
include("ptu.jl")
end # module ParallelTransportUnfolding
