import numpy as np
import torch
torch.set_default_dtype(torch.float64)
dtype = torch.float64
device = torch.device("cpu")

class Solver:
    def __init__(self, task_description, zero_floor=1e-8, base_penalty=1e8):
        self.zero_floor = zero_floor
        self.base_penalty = base_penalty

        self.A, self.B, self.H, self.Q, self.R = task_description["model"]
        self.Cs, self.Cu, self.Cb = task_description["cost_rate"]

        self.dims = {"s":self.A.shape[1], "o":self.H.shape[0], "m":self.A.shape[1], "u":self.B.shape[1]}

        self.result = {}
        self.info = {}

    def matT(self, X):
        return X.transpose(-2, -1)

    def to_orthonormal(self, X):
        Q, R = torch.linalg.qr(X)
        signs = torch.sign(torch.diag(R))
        return Q * signs

    def objective_function(self, flattened_params, return_info = False):

        offset_L = self.dims["u"] * self.dims["m"]
        L = flattened_params[ : offset_L].reshape(self.dims["u"], self.dims["m"])

        offset_G = offset_L + self.dims["m"] * self.dims["m"]
        G = flattened_params[offset_L : offset_G].reshape(self.dims["m"], self.dims["m"])

        K = flattened_params[offset_G : ].reshape(self.dims["m"], self.dims["o"])

        # Augmented transition matrix M with z = [s, e]
        cKH = torch.eye(self.dims["m"], dtype=K.dtype, device=K.device) - K @ self.H
        top_row = torch.cat([self.A + self.B @ L, - self.B @ L], dim=1)
        bottom_row = torch.cat([cKH @ (self.A + self.B @ L) - G, -cKH @ self.B @ L + G], dim=1)
        M = torch.cat([top_row, bottom_row], dim=0)

        # Stability penalty.
        eigs_M_abs = torch.abs(torch.real(torch.linalg.eigvals(M)))
        stability_violation = torch.sum(torch.relu(eigs_M_abs - (1-self.zero_floor)))

        # Noise covariance P in the augmented space.
        P = torch.cat([torch.cat([self.Q, self.Q @ self.matT(cKH)], dim=1),
                        torch.cat([cKH @ self.Q, cKH @ self.Q @ self.matT(cKH) + K @ self.R @ K.T], dim=1)], dim=0)

        # Discrete Lyapunov solve: Sigma = M Sigma M.T + P.
        dim = M.shape[0]
        I_dim2 = torch.eye(dim * dim, dtype=M.dtype, device=M.device)
        M_kron = torch.kron(M, M)
        tmp = I_dim2 - M_kron

        sol_flat = torch.linalg.solve(tmp, P.reshape(-1, 1)).reshape(-1)
        S = sol_flat.reshape(dim, dim)

        # Sigma validity penalties.
        eigs_S = torch.real(torch.linalg.eigvalsh(S))
        symmetric_violation = torch.sum(torch.abs(S - self.matT(S)))
        pos_semi_violation = torch.sum(torch.relu(-self.zero_floor - eigs_S))

        S_s = S[:self.dims["s"], :self.dims["s"]]
        S_e = S[self.dims["s"]: , self.dims["s"] :]
        cov_es = S[self.dims["s"] : , : self.dims["s"]]

        S_m = S_s - cov_es - cov_es.T + S_e
        cov_ms = S_s - cov_es
        S_u = L @ S_m @ self.matT(L)

        det_M = torch.linalg.det(S_m)
        det_EgS = torch.linalg.det(S_m - cov_ms @ torch.linalg.inv(S_s) @ cov_ms.T)
        bits = .5 * (torch.log2(torch.abs(det_M)) - torch.log2(torch.abs(det_EgS)))

        # Performance
        state_cost = torch.abs(torch.trace(self.Cs @ S_s))
        action_cost = torch.abs(torch.trace(self.Cu @ S_u))
        performance = state_cost + action_cost + self.Cb * bits

        penalty_loss = stability_violation + symmetric_violation + pos_semi_violation

        loss = performance + self.base_penalty * penalty_loss

        if not return_info:
            return loss

        info = {
            "loss": loss.detach(),
            "state_cost": state_cost.detach(),
            "action_cost": action_cost.detach(),
            "bits": (bits).detach(),
            "err": torch.trace(S_e).detach(),
            "penalty": penalty_loss.detach(),
            "stability_violation": stability_violation.detach(),
            "symmetric_violation":symmetric_violation.detach(),
            "pos_semi_violation": pos_semi_violation.detach(),
            "max_eig_M": max(eigs_M_abs).detach(),
            "min_eig_S": min(eigs_S).detach()
        }
        self.info = info
        return loss, self.info

    def compute_hessian(self, x):
        H = torch.autograd.functional.hessian(
            self.objective_function,
            x,
            create_graph=False,
            strict=False,
            vectorize=True,
        )
        H = H.reshape(x.numel(), x.numel())
        # Numerical differentiation can produce tiny asymmetries.
        return 0.5 * (H + H.T)

    # =========================================================================================================
    # Hybrid Adam / regularized-Newton optimizer.
    #         Adam provides robust global progress through nonconvex regions.
    #         Newton steps are attempted when curvature information is likely useful.
    #         A rejected Newton step simply falls back to Adam.
    # =========================================================================================================
    def minimize(self, x0, tol=1e-4, curvature_tol=1e-4, adam_lr=1e-3, max_steps=100000, hessian_every=20,
                       newton_gradient_threshold=10.0, relative_damping=1e-5, absolute_damping=1e-8,
                       maximum_newton_step=1.0, armijo_constant=1e-4, backtracking_factor=0.5,  # between 0 and 1
                       max_backtracks=30, print_every=100):

        x = x0.detach().clone().requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=adam_lr)

        converged = False
        status = "maximum iterations reached"

        last_min_eigenvalue = float("nan")
        last_step_type = "Adam"
        last_alpha = 0.0

        for step in range(max_steps):

            optimizer.zero_grad(set_to_none=True)

            loss = self.objective_function(x)

            if not torch.isfinite(loss).item():
                status = "non-finite loss"
                break

            loss.backward()

            gradient = x.grad.detach()
            g = gradient.reshape(-1)

            gradient_norm = torch.linalg.vector_norm(g)

            if not torch.isfinite(gradient_norm).item():
                status = "non-finite gradient"
                break

            # Compute the Hessian periodically and whenever the gradient is small.
            evaluate_hessian = (step % hessian_every == 0 or gradient_norm.item() <= newton_gradient_threshold)

            eigenvalues = None
            eigenvectors = None

            if evaluate_hessian:
                x_hessian = x.detach().clone().requires_grad_(True)

                H = self.compute_hessian(x_hessian)

                if not torch.isfinite(H).all().item():
                    status = "non-finite Hessian"
                    break

                eigenvalues, eigenvectors = torch.linalg.eigh(H)

                minimum_eigenvalue = eigenvalues[0]
                maximum_eigenvalue = eigenvalues[-1]

                last_min_eigenvalue = minimum_eigenvalue.item()

                # Correct second-order convergence test.
                if gradient_norm.item() <= tol and minimum_eigenvalue.item() >= -curvature_tol:
                    converged = True
                    status = "local minimum found"
                    break

            accepted_newton = False
            last_step_type = "Adam"
            last_alpha = 0.0

            # -------------------------------------------------------------
            # Attempt a regularized Newton step
            # -------------------------------------------------------------
            if evaluate_hessian and gradient_norm.item() > tol:

                spectral_scale = max(eigenvalues.abs().max().item(), 1.0)

                eigenvalue_floor = max(absolute_damping, relative_damping * spectral_scale, 10.0 * torch.finfo(x.dtype).eps * spectral_scale)

                # This defines the actual positive-definite Hessian model B.
                modified_eigenvalues = eigenvalues.clamp_min(eigenvalue_floor)

                gradient_coordinates = eigenvectors.T @ g

                newton_direction = -eigenvectors @ (gradient_coordinates / modified_eigenvalues)

                directional_derivative = torch.dot(g, newton_direction)

                # The modified Newton direction should be descent.
                if directional_derivative.item() < 0.0:

                    direction_norm = torch.linalg.vector_norm(newton_direction)

                    if direction_norm.item() > maximum_newton_step:
                        newton_direction = (newton_direction * maximum_newton_step / direction_norm)

                        directional_derivative = torch.dot(g, newton_direction)

                    # -----------------------------------------------------
                    # Armijo backtracking
                    # -----------------------------------------------------
                    alpha = 1.0
                    current_loss = loss.detach()

                    for _ in range(max_backtracks):

                        trial_x = (x.detach() + alpha * newton_direction.reshape_as(x))

                        with torch.no_grad():
                            trial_loss = self.objective_function(trial_x)

                        required_loss = (current_loss + armijo_constant * alpha * directional_derivative)

                        if torch.isfinite(trial_loss).item() and trial_loss.item() <= required_loss.item():
                            with torch.no_grad():
                                x.copy_(trial_x)

                            accepted_newton = True
                            last_step_type = "Newton"
                            last_alpha = alpha

                            # Adam's moments describe the old trajectory.
                            # Reset them after a manual Newton update.
                            optimizer.state.clear()
                            break

                        alpha *= backtracking_factor

            # -------------------------------------------------------------
            # Reliable fallback: Adam
            # -------------------------------------------------------------
            if not accepted_newton:
                optimizer.step()
                last_step_type = "Adam"
                last_alpha = adam_lr

            # -------------------------------------------------------------
            # Progress output
            # -------------------------------------------------------------
            if step % print_every == 0:
                _, info = self.objective_function(x.clone().detach(), return_info=True)
                print(f"\r i = {step:5d}   loss = {loss.item():.3f}   grad = {gradient_norm.item():.3f}   H_min = {last_min_eigenvalue:.3f}   e: {info["err"]:.6f}   p: {info["penalty"]:.2f}   ", end='', flush=True)

        # -----------------------------------------------------------------
        # Recompute diagnostics at the actual returned point
        # -----------------------------------------------------------------
        final_x = x.detach().clone().requires_grad_(True)
        final_loss = self.objective_function(final_x)

        final_gradient = torch.autograd.grad(final_loss, final_x)[0]

        final_gradient_norm = torch.linalg.vector_norm(final_gradient.reshape(-1))

        final_H = self.compute_hessian(final_x)
        final_eigenvalues = torch.linalg.eigvalsh(final_H)

        self.result = {
            "x": final_x.detach(),
            "loss": final_loss.item(),
            "gradient_norm": final_gradient_norm.item(),
            "min_hessian_eigenvalue": final_eigenvalues[0].item(),
            "max_hessian_eigenvalue": final_eigenvalues[-1].item(),
            "iterations": step + 1,
            "converged": converged,
            "status": status,
            "last_step_type": last_step_type,
            "last_step_size": last_alpha,
        }

        return self.result

    def cast_solution(self,):
        if self.result != {}:

            offset_L = self.dims["u"] * self.dims["m"]
            L = self.result["x"][: offset_L].reshape(self.dims["u"], self.dims["m"])

            offset_G = offset_L + self.dims["m"] * self.dims["m"]
            G = self.result["x"][offset_L: offset_G].reshape(self.dims["m"], self.dims["m"])

            K = self.result["x"][offset_G:].reshape(self.dims["m"], self.dims["o"])

            if not self.result["converged"]:
                print("WARNING! Sol might not be a local minima")

            policy = {"L": L.numpy(), "G": G.numpy(), "K": K.numpy()}
            return policy

        print("No solution found")
        return {}

def desc2tensor(desc):
    tensor_desc = {}
    for key in desc.keys():
        tmp = [torch.tensor(param, device=device, dtype=dtype) for param in desc[key]]
        tensor_desc[key] = tmp
    return tensor_desc

def initialize(policy):

    L = policy["L"]
    G = policy["G"]
    K = policy["K"]

    x0 = torch.tensor(np.concat([L.flatten(), G.flatten(), K.flatten()]), requires_grad=True, dtype=dtype)
    return x0

