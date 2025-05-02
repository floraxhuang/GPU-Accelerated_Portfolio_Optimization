# Import necessary libraries
using LinearAlgebra, JuMP, Clarabel, PythonCall
import CUDA
import PythonCall: pyconvert
# Allow scalar operations on GPU arrays if needed
CUDA.allowscalar(true)

# Define the floating-point type to use
MyFloat = Float64

# --- Data Structure for the Portfolio Model (Mean-Variance Conic with Cash/Leverage) ---
mutable struct PortfolioModel
    # --- Input Data ---
    n_assets::Int                   # Number of assets
    λ_risk::MyFloat                 # Risk aversion parameter (scales risk term 't')
    x0::Vector{MyFloat}             # Initial portfolio holdings (asset weights)
    cov::Matrix{MyFloat}            # Full asset covariance matrix
    cost::Vector{MyFloat}           # Transaction cost coefficients per asset
    return_ratio::Vector{MyFloat}   # Expected return ratios for assets (mu)
    leverage_limit::MyFloat         # Maximum allowed leverage (L^tar = sum(|x_i|))
    c_min::MyFloat                  # Minimum cash holding
    c_max::MyFloat                  # Maximum cash holding
    # removed: n_style, y0, expo_T, bias

    # --- Conic Formulation Specific Data ---
    # Lower Cholesky factor L of the covariance matrix (cov = L * L')
    # Used for the conic constraint: t >= || L' * x ||
    L_cov::Matrix{MyFloat}

    # --- JuMP Model Components ---
    model::JuMP.Model               # The optimization model object
    x::Vector{JuMP.VariableRef}     # Asset allocation variables (portfolio weights)
    t::JuMP.VariableRef             # Auxiliary variable for risk (standard deviation)
    c::JuMP.VariableRef             # Cash holding variable
    x_buy_sell::Vector{JuMP.VariableRef} # Variables for transaction costs (|x - x0|)
    con_1::Vector{JuMP.ConstraintRef} # Constraint part 1 for transaction cost linearization
    con_2::Vector{JuMP.ConstraintRef} # Constraint part 2 for transaction cost linearization
end

# --- Constructor for PortfolioModel ---
function CreatePortfolioModel(
    n_assets::Int, λ_risk::MyFloat,
    py_x0::PyVector{MyFloat},
    py_cov::PyMatrix{MyFloat},   # Use full covariance matrix
    py_cost::PyVector{MyFloat},
    py_u_cpu::PyVector{MyFloat}, # Expected returns (mu)
    leverage_limit::MyFloat=1.6, # Example default leverage limit
    c_min::MyFloat=0.2,          # Example default min cash
    c_max::MyFloat=0.8           # Example default max cash
)
    # Convert Python objects to Julia types
    x0 = pyconvert(Vector{MyFloat}, py_x0)
    cov_matrix = pyconvert(Matrix{MyFloat}, py_cov)
    cost = pyconvert(Vector{MyFloat}, py_cost)
    return_ratio = pyconvert(Vector{MyFloat}, py_u_cpu)

    # --- Pre-computation for Conic Formulation ---
    # Compute Cholesky factor L such that cov = L * L'
    local L_cov
    try
        cov_matrix_pd = cov_matrix + LinearAlgebra.I * 1e-9     # Small regularization
        chol = cholesky(Symmetric(cov_matrix_pd), check = true) # check=true throws PosDefException
        L_cov = chol.L                                          # Use the lower triangular factor L
    catch e
        if isa(e, PosDefException)
            error("Covariance matrix is not positive definite after regularization, cannot compute Cholesky factor.")
        else
            println("Cholesky factorization failed with error: ", e)
            rethrow(e)
        end
    end

    # --- Initialize the JuMP Model ---
    model = Model(Clarabel.Optimizer)
    # Set solver attributes
    #set_optimizer_attribute(model, "direct_solve_method", :qdldl) # testing CPU
    set_optimizer_attribute(model, "direct_solve_method", :cudss)   # :cudssmixed or :cudss
    set_optimizer_attribute(model, "verbose", true)
    set_optimizer_attribute(model, "iterative_refinement_enable", false)
    set_optimizer_attribute(model, "presolve_enable", true)
    set_optimizer_attribute(model, "static_regularization_enable", true)
    set_optimizer_attribute(model, "dynamic_regularization_enable", true)
    set_optimizer_attribute(model, "equilibrate_enable", true)

    # --- Return the Initialized Struct ---
    var_vector_empty = Vector{JuMP.VariableRef}()
    con_empty = Vector{JuMP.ConstraintRef}()
    # Need valid VariableRef placeholders before setup_model! is called
    @variable(model, dummy_t >= 0) # Standard deviation >= 0
    @variable(model, dummy_c)      # Cash placeholder

    return PortfolioModel(
        n_assets, λ_risk, x0, cov_matrix, cost, return_ratio,
        leverage_limit, c_min, c_max,    # Store new parameters
        L_cov,                           # Store pre-computed Cholesky factor
        model,
        var_vector_empty,                # x
        dummy_t,                         # t
        dummy_c,                         # Cash
        var_vector_empty,                # x_buy_sell
        con_empty, con_empty             # con_1, con_2
    )
end

# --- Setup Model Variables and Constraints ---
function setup_model!(pm::PortfolioModel)
    # Get the model object and parameters from the struct
    model = pm.model
    L_cov = pm.L_cov # Use the pre-computed Cholesky factor

    # --- Define Decision Variables ---
    @variable(model, x[1:pm.n_assets])      # Asset weights
    set_start_value.(x, pm.x0)              # Set initial guess for asset weights

    # Cash holding variable
    @variable(model, pm.c_min <= c <= pm.c_max)

    # --- Define Auxiliary Variable for Conic Formulation ---
    @variable(model, t >= 0.0)              # Auxiliary variable for standard deviation: t >= sqrt(x' * cov * x)

    # --- Define Variables for Transaction Costs ---
    @variable(model, x_buy_sell[1:pm.n_assets] >= 0.0) # Absolute difference |x - x0|

    # --- Update Struct References ---
    pm.x = x
    pm.t = t                  # Store the actual variable reference for Std Dev
    pm.c = c                  # Store the actual variable reference for cash
    pm.x_buy_sell = x_buy_sell

    # --- Define Model Constraints ---
    # removed original asset weight bounds (can still add explicit bounds if needed)

    # Budget constraint (including cash)
    @constraint(model, budget_cash, sum(x) + c == 1.0)

    # Transaction cost / Trade linearization constraints: x_buy_sell >= |x - x0|
    pm.con_1 = @constraint(model, tc_abs1, x_buy_sell .>= x .- pm.x0)
    pm.con_2 = @constraint(model, tc_abs2, x_buy_sell .>= pm.x0 .- x)

    # Leverage constraint: sum(|x_i - x0_i|) <= L^tar
    # This limits the sum of absolute trades (turnover / total weight change)
    @constraint(model, turnover_leverage, sum(x_buy_sell) <= pm.leverage_limit)

    # --- Define Conic Constraint for Risk ---
    # t >= sqrt(x' * cov * x) <=> t >= || L' * x ||_2
    # Represented using SecondOrderCone: [t; L' * x] in SecondOrderCone
    L_T_x = L_cov' * x
    @constraint(model, risk_socp, [t; L_T_x] in SecondOrderCone())

    # --- Define Objective Function ---
    # Minimize: lambda_risk * StdDev - ExpectedReturn + CostPenalty
    @objective(model, Min,
        pm.λ_risk * t                                     # Scaled Standard Deviation
        - dot(pm.return_ratio, x)                         # Expected Return (to be maximized)
        + (1 / pm.λ_risk) * dot(pm.cost, x_buy_sell)      # Scaled Transaction Cost Penalty
    )
end

# --- Perform the Initial Optimization Solve ---
function initial_solve!(pm::PortfolioModel)
    println("Starting initial solve...")
    solve_time = @elapsed optimize!(pm.model)
    println("Initial solve completed in $solve_time seconds.")

    status = termination_status(pm.model)
    if status != MOI.OPTIMAL && status != MOI.ALMOST_OPTIMAL
        println("Warning: Initial solve did not reach optimality. Status: ", status)
    else
        println("Initial solve successful. Status: ", status)
        # println("Optimal Objective Value: ", objective_value(pm.model))
        # println("Calculated Std Dev (t): ", value(pm.t))
        # println("Calculated Variance (t^2): ", value(pm.t)^2)
    end
end

# --- Update Return Data for Subsequent Solves ---
function update_return_ratio!(pm::PortfolioModel, py_u_cpu::PyVector{MyFloat})
    # Convert new return data and update the model struct
    pm.return_ratio = pyconvert(Vector{MyFloat}, py_u_cpu)
    println("Expected return ratios updated.")
end

# --- Re-solve the Optimization Problem After Updates ---
function resolve!(pm::PortfolioModel)
    println("Starting re-solve...")
    # --- Update Warm-start Values and Constraints ---
    try
        # Update initial portfolio x0 for the next round's transaction costs
        copyto!(pm.x0, value.(pm.x))

        # Update RHS of transaction cost constraints based on the new x0
        set_normalized_rhs.(pm.con_1, .-pm.x0)
        set_normalized_rhs.(pm.con_2, pm.x0)

        # --- Update Objective Coefficients ---
        # Expected return coefficients change
        new_coeffs_x = -pm.return_ratio
        set_objective_coefficient.(pm.model, pm.x, new_coeffs_x)
        # Coefficient for t (pm.λ_risk) and x_buy_sell remain unchanged unless parameters change

        # --- Set Start Values for Warm-starting the Solver ---
        set_start_value.(pm.x, pm.x0)                          # Start from the previous solution
        set_start_value(pm.t, value(pm.t))                     # Use previous optimal std dev
        set_start_value(pm.c, value(pm.c))                     # Use previous optimal cash
        set_start_value.(pm.x_buy_sell, value.(pm.x_buy_sell)) # Use previous optimal value

    catch e
        println("Error accessing previous solution values during warm-start setup: ", e)
        println("Attempting re-solve without warm-start for some variables.")
        try set_start_value.(pm.x, pm.x0) catch; end
    end

    # --- Re-optimize the Model ---
    solve_time = @elapsed optimize!(pm.model)
    println("Re-solve completed in $solve_time seconds.")

    status = termination_status(pm.model)
    if status != MOI.OPTIMAL && status != MOI.ALMOST_OPTIMAL
        println("Warning: Re-solve did not reach optimality. Status: ", status)
    else
        println("Re-solve successful. Status: ", status)
        # println("Optimal Objective Value: ", objective_value(pm.model))
        # println("Calculated Std Dev (t): ", value(pm.t))
    end
end
    
# --- Get the Calculated Risk Value (Standard Deviation) ---
function get_risk(pm::PortfolioModel)
    # Check if the model has a valid solution
    if primal_status(pm.model) == MOI.FEASIBLE_POINT || primal_status(pm.model) == MOI.NEARLY_FEASIBLE_POINT
        # Return the optimal value of the auxiliary risk variable t (standard deviation)
        return value(pm.t)
    else
        println("Warning: Cannot get risk, model does not have a feasible primal solution. Status: ", primal_status(pm.model))
        return NaN
    end
end

# --- Wrapper Functions for Python Compatibility ---
# Wrapper for setup_model!
function setup_model_b(pm::PortfolioModel)
    setup_model!(pm)
end

# Wrapper for initial_solve!
function initial_solve_b(pm::PortfolioModel)
    initial_solve!(pm)
end

# Wrapper for update_return_ratio!
function update_return_ratio_b(pm::PortfolioModel, u::PyVector{MyFloat})
    update_return_ratio!(pm, u)
end

# Wrapper for resolve!
function resolve_b(pm::PortfolioModel)
    resolve!(pm)
end

