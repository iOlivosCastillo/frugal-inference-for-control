import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import solve_discrete_are
from scipy.linalg import inv

# =========================================================================================================
# Solves the standard LQG problem
# Input:
#  task description = {"model": [A, B, H, Q, R], "cost_rate":[Cs, Cu, Cb]}, parameters describing the
#                      world model (linearized dynamics and observation function) and cost function
#                      (cost per unit of deviation, per unit of motion, and per bit of information)
# Output:
#  policy = {"G":[], "K":[], "L":[]}, parameters describing the joint inference and control strategy
# =========================================================================================================

def solve_lqg(task_description):
    A, B, H, Q, R = task_description["model"]
    Cx, Cu, _ = task_description["cost_rate"]

    # Control via a linear quadratic regulator
    Pc = solve_discrete_are(A, B, Cx, Cu)
    # control gain
    L = -inv(Cu + B.T @ Pc @ B) @ B.T @ Pc @ A

    # Inference via a Kalman filter
    tmpA = A.T
    tmpB = A.T @ H.T
    tmpQ = Q
    tmpR = H @ Q @ H.T + R
    tmpE = np.eye(A.shape[0])
    tmpS = Q @ H.T

    Pf = solve_discrete_are(tmpA, tmpB, tmpQ, tmpR, tmpE, tmpS)
    # Kalman gain
    K = Pf @ H.T @ inv(R)
    G = np.dot(np.eye(A.shape[0]) - np.dot(K, H), A + np.dot(B, L))

    return {'L': L, 'K': K, 'G': G}

# ==================================================================================================================
# Analytical performance at equilibrium
# Input:
#  policy = {"G":[], "K":[], "L":[]}, parameters describing the joint inference and control strategy
#  task description = {"model": [A, B, H, Q, R], "cost_rate":[Cs, Cu, Cb]}, parameters describing the
#                      world model (linearized dynamics and observation function) and cost function
#                      (cost per unit of deviation, per unit of motion, and per bit of information)
#  show_res, optional boolean indicating whether to print the results.

# Output:
#   S, Augmented joint, steady-state covariance describing the interactions between states, estimates, and actions
#   Sigma, dictionary containing the covariance blocks that compose S.
#   costs, array containing state cost, action cost, and info_cost the agent pays at equilibrium
# ==================================================================================================================

def buildSigma(policy, task_description, show_res = False):
    A, B, H, Q, R = task_description["model"]
    Cs, Cu, Cb = task_description["cost_rate"]

    dim_s, dim_o, dim_m, dim_u = A.shape[0], H.shape[0], A.shape[0], B.shape[1]

    L, G = policy["L"], policy["G"]

    # Build an augmented state z = [s, mu, u], here z_next = M z + eta with eta following N(0, P)
    if "KH" in policy:
        KH = policy["KH"]
        Zn = np.zeros((dim_s, dim_m))
        M = np.block([[A, Zn, B],                         # s
                      [KH @ A, G, KH @ B],                # mu
                      [L @ KH @ A, L @ G, L @ KH @ B]])   # u

        In = np.eye(dim_s)
        Zn = np.zeros([dim_s, dim_s])
        Ps = np.block([[In, Zn.T],
                       [KH, KH],
                       [L @ KH, L @ KH]])
        Pc = np.block([[Q, Zn.T],
                       [Zn, np.linalg.pinv(H) @ R @ np.linalg.pinv(H).T]])
        P = Ps @ Pc @ Ps.T
    else:
        K = policy["K"]
        Zn = np.zeros((dim_s, dim_m))
        M = np.block([[A, Zn, B],  # s
                      [K @ H @ A, G, K @ H @ B],  # mu
                      [L @ K @ H @ A, L @ G, L @ K @ H @ B]])  # u

        In = np.eye(dim_s)
        Zn = np.zeros([dim_o, dim_s])
        Ps = np.block([[In, Zn.T], [K @ H, K], [L @ K @ H, L @ K]])
        Pc = np.block([[Q, Zn.T], [Zn, R]])
        P = Ps @ Pc @ Ps.T

    # The joint covariance matrix describing state, estimates, and actions at equilibrium
    # is the solution to the discrete Lyapunov equation S = M S M.T + P:
    S = solve_discrete_lyapunov(M, P)

    S_s = S[:dim_s, :dim_s]
    S_m = S[dim_s : dim_s + dim_m, dim_s : dim_s + dim_m]
    S_u = S[dim_s + dim_m : , dim_s + dim_m :]

    # Calculate task performance (state + action costs)
    state_cost = np.trace(Cs @ S_s)
    action_cost = np.trace(Cu @ S_u)

    # Calculate mutual information between states and estimates
    det_S = np.linalg.det(S_s)
    det_M = np.linalg.det(S_m)
    det_SM = np.linalg.det(S[: dim_s + dim_m, : dim_s + dim_m])
    info_cost = 0.5 * (np.log2(det_S * det_M) - np.log2(det_SM))

    # Calculate the error covariance with error = s - estimate
    Se = S_s - S[dim_s:dim_s + dim_m, :dim_s] - S[dim_s:dim_s + dim_m, :dim_s].T + S_m

    if show_res:
        total = state_cost + action_cost + Cb * info_cost
        print()
        print("Expected costs at equilibrium:")
        print(f"SC = {state_cost:.4f}, UC = {action_cost:.4f}, IC = {info_cost:.4f}, ERR = {np.trace(Se):.4f}, Loss = {total:.4f}")

    Sigma = {
        "s":S[:dim_s, :dim_s],
        "m":S[dim_s:dim_s + dim_m, dim_s:dim_s + dim_m],
        "u":S[dim_s + dim_m:, dim_s + dim_m:],
        "ms":S[dim_s:dim_s + dim_m, :dim_s],
        "us":L @ S[dim_s:dim_s + dim_m, :dim_s],
        "um":L @ S[dim_s:dim_s + dim_m, dim_s:dim_s + dim_m],
        "e": Se,
        "se": S[:dim_s, :dim_s] - S[dim_s:dim_s + dim_m, :dim_s]}

    return S, Sigma, [state_cost, action_cost, info_cost]

# ==================================================================================================================
# Project a symmetric matrix onto the PSD cone by clipping tiny negative eigenvalues to zero.
# Input:
#  M, input matrix with tiny negative eigenvalues
#  tol, threshold below which negative eigenvalues are treated as negligible.
#  diagnostic, optional boolean indicating whether to show min eigenvalue before and after the procedure
# Output:
#  M_psd, a positive semidefinite version of M
# ==================================================================================================================
def nearest_psd(M, tol=1e-8, diagnostic=False):
    M = 0.5 * (M + M.T)

    eigvals, eigvecs = np.linalg.eig(M)

    # Clip small negative eigenvalues
    eigvals_clipped = np.where(eigvals > tol, eigvals, tol)

    M_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    M_psd = 0.5 * (M_psd + M_psd.T)

    if diagnostic:
        print("min eigenvalue before clipping:", eigvals.min())
        print("min eigenvalue after clipping:", np.linalg.eigvals(M_psd).min())

    return M_psd

# ==================================================================================================================
# Build an n x n rotation matrix rotating in the (i, j) coordinate plane.
# Input:
#  n: desired dimensionality
#  i, j: coordinates defining the rotation plane.
#  theta: rotation angle in radians.
# Output:
#  R: n x n rotation matrix
# ==================================================================================================================
def rotation_matrix_nd(n, i, j, theta):
    R = np.eye(n)

    c, s = np.cos(theta), np.sin(theta)

    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s

    return R

