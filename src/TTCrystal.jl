module TTCrystal

using Distributed, DistributedArrays, LinearAlgebra, SharedArrays, StaticArrays, Printf, OffsetArrays, Elliptic, Roots, NLSolversBase, Optim, Plots, Quadmath, GenericSchur, QuadGK, JLD2

import InteractiveUtils: subtypes
import LineSearches: MoreThuente
import Base.Iterators: product
import Base: ==, <, >, <=, >=


include("parameters.jl")
include("supplementary_functions.jl")
include("initial_conditions.jl")
include("matrices.jl")
include("structure_generation.jl")
include("obj_functions.jl")
include("optimizers.jl")
include("ctime_gfuncs.jl")
#include("interface.jl")

export ParamsB1, ParamsB1_pinned, ParamsNoB1, ParamsNoB1_pinned, ReducedDispersion, Saver
export generate_shared_cash, generate_psamples
export isB1, output, save_data, log_data, seek_minimum, newton_step!, precompute_newton_step!, finalize!
export B, B′₀, B′₁, B′′₀₀, B′′₁₁, B′′₀₁
export bfgsObjFunc

    
end # module
