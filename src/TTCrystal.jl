module TTCrystal

using FileIO, Distributed, DistributedArrays, LinearAlgebra, SharedArrays, StaticArrays, Printf, OffsetArrays, Elliptic, Roots, NLSolversBase, Optim, Plots, Quadmath, GenericSchur, QuadGK, JLD2, FFTW

using CustomQuad

import InteractiveUtils: subtypes
using LineSearches
#import LineSearches: MoreThuente
import Base.Iterators: product
import Base: ==, <, >, <=, >=


include("parameters.jl")
include("repulsion.jl")
#include("supplementary_functions.jl")
#include("initial_conditions.jl")
include("matrices.jl")
include("structure_generation.jl")
include("obj_functions.jl")
include("optimizers.jl")
include("ctime_gfuncs.jl")
#include("interface.jl")

export ParamsB1, ParamsB1_pinned, ParamsNoB1, ParamsNoB1_pinned, ReducedDispersion, Dispersion, Saver
export get_u₀, get_psamples, widen
export generate_shared_cash, generate_psamples, get_psamples, get_psamples_old, get_psamples_new, separate_psamples
export construct_objective, process_bs
export G_Cash, G_Cash2
export compute_single_period!, compute_full_span!
export precompute_step!, precompute_step, compute_grad_components!
export isB1, output, save_data, log_data, finalize!, finalize
export B, B′₀, B′₁, B′′₀₀, B′′₁₁, B′′₀₁
export bfgsObjFunc
export @unpack

    
end # module
