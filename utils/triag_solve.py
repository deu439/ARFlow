import torch
from torch.autograd import Function, gradcheck
from torch.autograd.function import once_differentiable
import numpy as np
import scipy as sp
import time
import torch.nn.functional as F

import triag_solve_cuda


def matrix_vector_product_old(A, B, C, X):
    B_Y = torch.nn.functional.pad(B * X[:, :, :, 0:-1], (1, 0))
    C_Y = torch.nn.functional.pad(C * X[:, :, 0:-1, :], (0, 0, 1, 0))
    return A * X + B_Y + C_Y


def matrix_vector_product(A, B, C, D, X):
    """
    |D|C|
    |B|A|
    """
    B_Y = torch.nn.functional.pad(B * X[:, :, :, 0:-1], (1, 0))
    C_Y = torch.nn.functional.pad(C * X[:, :, 0:-1, :], (0, 0, 1, 0))
    D_Y = torch.nn.functional.pad(D * X[:, :, 0:-1, 0:-1], (1, 0, 1, 0))
    return A * X + B_Y + C_Y + D_Y


def matrix_vector_product_T_old(A, B, C, X):
    B_Y = torch.nn.functional.pad(B * X[:, :, :, 1:], (0, 1))
    C_Y = torch.nn.functional.pad(C * X[:, :, 1:, :], (0, 0, 0, 1))
    return A * X + B_Y + C_Y


def matrix_vector_product_T(A, B, C, D, X):
    B_Y = torch.nn.functional.pad(B * X[:, :, :, 1:], (0, 1))
    C_Y = torch.nn.functional.pad(C * X[:, :, 1:, :], (0, 0, 0, 1))
    D_Y = torch.nn.functional.pad(D * X[:, :, 1:, 1:], (0, 1, 0, 1))
    return A * X + B_Y + C_Y


def forward_substitution(A, B, C, X):
    """
    Solves J*y=x where J is lower triangular matrix, represented by A: K x L x M x N (center pixels), B: K x L x M x N-1
    (left pixels), C: K x L x M-1 x N (above pixels). y = vec(Y: K x L x M x N) and x = vec(X: K x L x M x N).
    """
    M, N = A.shape[2:]
    Y = torch.clone(X)
    for i in range(M):
        for j in range(N):
            if i > 0:
                Y[:, :, i, j] -= Y[:, :, i - 1, j] * C[:, :, i - 1, j]
            if j > 0:
                Y[:, :, i, j] -= Y[:, :, i, j - 1] * B[:, :, i, j - 1]
            Y[:, :, i, j] /= A[:, :, i, j]

    return Y


def backward_substitution(A, B, C, X):
    """
    Solves J*y=x where J is upper triangular matrix, represented by A: K x L x M x N (center pixels), B: K x L x M x N-1
    (right pixels), C: K x L x M-1 x N (below pixels). y = vec(Y: K x L x M x N) and x = vec(X: K x L x M x N).
    """
    M, N = A.shape[2:]
    Y = torch.clone(X)
    for i in reversed(range(M)):
        for j in reversed(range(N)):
            if i < M-1:
                Y[:, :, i, j] -= Y[:, :, i + 1, j] * C[:, :, i, j]
            if j < N-1:
                Y[:, :, i, j] -= Y[:, :, i, j + 1] * B[:, :, i, j]
            Y[:, :, i, j] /= A[:, :, i, j]

    return Y


def ABC_matrix_np(A, B, C):
    """
    Generate an MN x MN matrix from the MxN, MxN-1 and M-1xN arrays A, B, C
    """
    M, N = A.shape
    a = A.ravel()   # Row-wise stacking
    c = C.ravel()
    A_mat = np.diag(a)
    B_mat = sp.linalg.block_diag(*[np.diag(B[i, :], -1) for i in range(M)])
    C_mat = np.diag(c, -N)
    return A_mat + B_mat + C_mat


def forward_substitution_npsolve(A, B, C, X):
    """
    Solves J*y=x where J is lower triangular matrix, represented by A: M x N (center pixels), B: M x N-1
    (left pixels), C: M-1 x N (above pixels). y = vec(Y: M x N) and x = vec(X: M x N) using numpy.linalg.solve.
    Only one system can be solved at once.
    """
    M, N = A.shape
    x = X.ravel()
    J_mat = ABC_matrix_np(A, B, C)
    y = np.linalg.solve(J_mat, x)
    return y.reshape(M, N)


def backward_substitution_npsolve(A, B, C, X):
    """
    Solves J*y=x where J is upper triangular matrix, represented by A: M x N (center pixels), B: M x N-1
    (right pixels), C: M-1 x N (below pixels). y = vec(Y: M x N) and x = vec(X: M x N) using numpy.linalg.solve.
    Only one system can be solved at once.
    """
    M, N = A.shape
    x = X.ravel()
    J_mat = ABC_matrix_np(A, B, C)

    import matplotlib.pyplot as plt
    import pprint
    pprint.pprint(J_mat)
    plt.imshow(J_mat, cmap='gray')
    plt.show()
    y = np.linalg.solve(J_mat.T, x)
    return y.reshape(M, N)


class ForwardSubst(Function):
    @staticmethod
    def forward(ctx, A, B, C, X):
        Y = triag_solve_cuda.forward_substitution(A, B, C, X)
        #Y = forward_substitution(A, B, C, X)
        ctx.save_for_backward(A, B, C, Y)
        return Y

    @staticmethod
    @once_differentiable
    def backward(ctx, dY):
        A, B, C, Y = ctx.saved_tensors
        dX = triag_solve_cuda.backward_substitution(A, B, C, dY)
        #dX = backward_substitution(A, B, C, dY)
        dA = -dX * Y
        dB = -dX[:, :, :, 1:] * Y[:, :, :, :-1]
        dC = -dX[:, :, 1:, :] * Y[:, :, :-1, :]
        return dA, dB, dC, dX


class BackwardSubst(Function):
    @staticmethod
    def forward(ctx, A, B, C, X):
        Y = triag_solve_cuda.backward_substitution(A, B, C, X)
        #Y = backward_substitution(A, B, C, X)
        ctx.save_for_backward(A, B, C, Y)
        return Y

    @staticmethod
    @once_differentiable
    def backward(ctx, dY):
        A, B, C, Y = ctx.saved_tensors
        dX = triag_solve_cuda.forward_substitution(A, B, C, dY)
        #dX = forward_substitution(A, B, C, dY)
        dA = -dX * Y
        dB = -dX[:, :, :, :-1] * Y[:, :, :, 1:]
        dC = -dX[:, :, :-1, :] * Y[:, :, 1:, :]
        return dA, dB, dC, dX


def marginal_variances(A, B, C):
    M, N = A.shape[2:]
    H = torch.zeros_like(A)
    for i in range(M):
        for j in range(N):
            # Initialize unit vector
            X = torch.zeros_like(A)
            X[:,:,i,j] = 1
            # Solve the system
            Y = triag_solve_cuda.forward_substitution(A, B, C, X)
            #Y = forward_substitution(A, B, C, X)
            H[:,:,i,j] = torch.sum(Y*Y, dim=(2, 3))

    return H


def marginal_variances_fast(A, B, C):
    M, N = A.shape[2:]
    H = torch.zeros_like(A)
    Y = torch.zeros_like(A)

    for k in range(M):
        for l in range(N):
            sum = torch.zeros(A.shape[:2] + (1, 1), dtype=A.dtype)
            c = torch.zeros(A.shape[:2] + (1, 1), dtype=A.dtype)

            # For i=k iterate over j=l,...,N
            for j in range(l, N):
                if j == l:
                    Y[:, :, k, j] = 1
                if j > l:
                    Y[:, :, k, j] = -Y[:, :, k, j - 1] * B[:, :, k, j - 1]
                Y[:, :, k, j] /= A[:, :, k, j]

                # Kahan summation
                y = Y[:, :, k, j]*Y[:, :, k, j] - c
                t = sum + y
                c = (t - sum) - y
                sum = t

            # For i=k+1,...,M iterate over j=0,...,N
            for i in range(k+1, M):
                for j in range(N):
                    Y[:, :, i, j] = 0

                    if j >= l or i > k+1:   # If i=k+1 only upper neighbors j=l,...,N are valid
                        Y[:, :, i, j] -= Y[:, :, i - 1, j] * C[:, :, i - 1, j]
                    if j > 0:
                        Y[:, :, i, j] -= Y[:, :, i, j - 1] * B[:, :, i, j - 1]

                    Y[:, :, i, j] /= A[:, :, i, j]

                    # Kahan summation
                    y = Y[:, :, i, j]*Y[:, :, i, j] - c
                    t = sum + y
                    c = (t - sum) - y
                    sum = t

            H[:, :, k, l] = sum

    return H


def inverse_l1norm(A, B, C, n_iter=100):
    """
    Computes an approximation to ||L^{-1}||_1 using Algorithm 5.1 of [1].

    [1] N. J. Higham, “A survey of condition number estimation for triangular matrices,” SIAM Review, vol. 29, no. 4, pp.
    575–596, Dec. 1987.
    """

    M, N = A.shape
    X = torch.ones_like(A) / (M*N)
    A = A.view(1,1,M,N).contiguous() # 4d arrays are needed in the solvers
    B = B.view(1,1,M,N-1).contiguous()
    C = C.view(1,1,M-1,N).contiguous()

    for n in range(n_iter):    # Do not run forever
        # Solve A*y = x
        #Y = forward_substitution(A, B, C, X.view(1,1,M,N)).squeeze()
        Y = triag_solve_cuda.forward_substitution(A, B, C, X.view(1,1,M,N).contiguous()).squeeze()

        # Form xi
        Xi = torch.ones_like(Y)
        Xi[Y < 0] = -1

        # Solve A^T*z = xi
        #Z = backward_substitution(A, B, C, Xi.view(1,1,M,N)).squeeze()
        Z = triag_solve_cuda.backward_substitution(A, B, C, Xi.view(1,1,M,N).contiguous()).squeeze()

        # If ||z||_inf <= z^T*x
        absZ = torch.abs(Z)
        max_w, ind_w = torch.max(absZ, dim=-1)
        max_h, ind_h = torch.max(max_w, dim=-1)
        ind_w = ind_w[ind_h]
        if max_h <= torch.sum(Z * X):
            return torch.sum(torch.abs(Y))

        # X = e_j
        X.zero_()
        X[ind_h, ind_w] = 1

    return torch.Tensor([float('inf')]).type_as(A)


def inverse_l1norm_exact(A, B, C):
    M, N = A.shape
    a = A.ravel()   # Row-wise stacking
    c = C.ravel()
    A_mat = np.diag(a)
    B_mat = sp.linalg.block_diag(*[np.diag(B[i, :], -1) for i in range(M)])
    C_mat = np.diag(c, -N)
    J_mat = A_mat + B_mat + C_mat
    S_mat = np.linalg.inv(J_mat)
    return np.linalg.norm(J_mat) * np.linalg.norm(S_mat, ord=1)


def trans_inverse_l1norm_exact(A, B, C):
    M, N = A.shape
    a = A.ravel()   # Row-wise stacking
    c = C.ravel()
    A_mat = np.diag(a)
    B_mat = sp.linalg.block_diag(*[np.diag(B[i, :], -1) for i in range(M)])
    C_mat = np.diag(c, -N)
    J_mat = A_mat + B_mat + C_mat
    S_mat = np.linalg.inv(J_mat.T)
    #return np.linalg.norm(S_mat, ord=1)
    return np.linalg.norm(J_mat.T) * np.linalg.norm(S_mat, ord=1)


def natural_gradient_np(G, T):
    H = T.T @ np.tril(G)
    Hbb = np.tril(H) - np.diag(np.diag(H))/2
    Q = T @ Hbb
    return Q


def natural_gradient(GA, GB, GC, TA, TB, TC):
    """
    Calculates the natural gradient with respect to T. G represents the Euclidean gradient and T represents the
    lower-triangular Cholesky factor of a precision matrix.
    """
    # Compute H double bar
    h_ll = (TA * GA + F.pad(TB * GB, [0, 1]) + F.pad(TC * GC, [0, 0, 0, 1])) / 2
    h_l1l = TA[:, :, :, 1:] * GB
    h_lNl = TA[:, :, 1:, :] * GC
    h_lN1l = TB[:, :, 1:, :] * GC[:, :, :, 1:]

    # Compute the natural gradient
    q_ll = TA * h_ll
    q_l1l = TA[:, :, :, 1:] * h_l1l + TB * h_ll[:, :, :, :-1]
    q_lNl = TA[:, :, 1:, :] * h_lNl + F.pad(TB[:, :, 1:, :] * h_lN1l, [1, 0]) + TC * h_ll[:, :, :-1, :]
    return q_ll, q_l1l, q_lNl


class NaturalGradientIdentity(Function):
    @staticmethod
    def forward(ctx, A, B, C, X):
        ctx.save_for_backward(A, B, C, X)
        return A, B, C, X

    @staticmethod
    @once_differentiable
    def backward(ctx, dA, dB, dC, dX):
        A, B, C, X = ctx.saved_tensors

        # Natural gradient with respect to X (the mean)
        dX_n = triag_solve_cuda.forward_substitution(A, B, C, dX)
        dX_n = triag_solve_cuda.backward_substitution(A, B, C, dX_n)

        # Natural gradient with respect to T
        dA_n, dB_n, dC_n = natural_gradient(dA, dB, dC, A, B, C)

        return dA_n, dB_n, dC_n, dX_n


def check_gradient():
    M = 5   # Rows
    N = 5   # Cols
    A = 2*torch.ones((2, 2, M, N), requires_grad=True, dtype=torch.double)
    B = torch.randn(2, 2, M, N-1, requires_grad=True, dtype=torch.double) # left
    C = torch.randn(2, 2, M-1, N, requires_grad=True, dtype=torch.double) # up
    X = torch.randn(2, 2, M, N, requires_grad=True, dtype=torch.double)

    # Exact gradient
    res = gradcheck(ForwardSubst().apply, (A, B, C, X))
    if res:
        print("ForwardSubst() gradient ok!")
    res = gradcheck(BackwardSubst().apply, (A, B, C, X))
    if res:
        print("BackwardSubst() gradient ok!")


def check_solver():
    M = 5 # Rows
    N = 6  # Cols
    A = 2*torch.ones((1, 1, M, N), dtype=torch.double)
    B = torch.randn(1, 1, M, N - 1, dtype=torch.double)  # left
    C = torch.randn(1, 1, M - 1, N, dtype=torch.double)  # up
    X = torch.randn(1, 1, M, N, dtype=torch.double)

    # Solve using numpy
    #t = time.process_time()
    Yn = backward_substitution_npsolve(A.squeeze().numpy(), B.squeeze().numpy(), C.squeeze().numpy(), X.squeeze().numpy())
    #print("Runtime numpy: ", time.process_time() - t, "s")
    res = matrix_vector_product_T(A, B, C, torch.tensor(Yn).view(1, 1, M, N)) - X
    print("Residual Numpy: ", torch.sum(torch.abs(res)) / torch.sum(torch.abs(X)))

    # Solve using backward_substitution
    t = time.process_time()
    Yp = backward_substitution(A, B, C, X)
    #torch.cuda.synchronize()
    #print("Runtime python:", time.process_time() - t, "s")
    #print("Python max err:", np.max(np.abs(Yn - Yp.squeeze().numpy())))

    res = matrix_vector_product_T(A, B, C, Yp) - X
    print("Residual: ", torch.sum(torch.abs(res)) / torch.sum(torch.abs(X)))

    # Solve on GPU
    #t = time.process_time()
    #Yc = triag_solve_cuda.forward_substitution(A, B, C, X)
    #torch.cuda.synchronize()
    #print("CUDA: ", time.process_time() - t)


def check_inverse_diagonal():
    M = 10  # Rows
    N = 10  # Cols
    A = 2*torch.ones((1, 1, M, N))
    B = torch.randn(1, 1, M, N - 1)  # left
    C = torch.randn(1, 1, M - 1, N)  # up

    H1 = marginal_variances(A, B, C)
    H2 = marginal_variances_fast(A, B, C)
    #H2 = triag_solve_cuda.inverse_diagonal(A.cuda(), B.cuda(), C.cuda()).cpu()
    rel_error = torch.abs(H1 - H2) / torch.abs(H1)
    print("Relative error: ", rel_error)
    print("Max relative error: ", torch.max(rel_error))


def check_inverse_l1norm():
    M = 10  # Rows
    N = 20  # Cols
    A = 1*torch.ones((M, N))
    B = torch.randn(M, N - 1)  # left
    C = torch.randn(M - 1, N)  # up

    norm = inverse_l1norm(A, B, C, n_iter=10)
    print("Approximate: ", norm.item())
    norm = inverse_l1norm_exact(A, B, C)
    print("Exact: ", norm)
    norm = trans_inverse_l1norm_exact(A, B, C)
    print("Trans Exact: ", norm)


def check_natural_gradient():
    M = 20  # Rows
    N = 30  # Cols
    GA = torch.randn((M, N), dtype=torch.double)
    GB = torch.randn(M, N - 1, dtype=torch.double)  # left
    GC = torch.randn(M - 1, N, dtype=torch.double)  # up
    TA = torch.randn((M, N), dtype=torch.double)
    TB = torch.randn(M, N - 1, dtype=torch.double)  # left
    TC = torch.randn(M - 1, N, dtype=torch.double)  # up

    # In numpy
    G = ABC_matrix_np(GA, GB, GC)
    T = ABC_matrix_np(TA, TB, TC)
    Q = natural_gradient_np(G, T)

    q_ll, q_l1l, q_lNl = natural_gradient(
        GA.view(1, 1, M, N), GB.view(1, 1, M, N-1), GC.view(1, 1, M-1, N),
        TA.view(1, 1, M, N), TB.view(1, 1, M, N-1), TC.view(1, 1, M-1, N)
    )

    q_ll = q_ll.squeeze().detach().numpy().ravel()
    err = np.abs(np.diag(Q) - q_ll)
    print("q_ll: ", np.max(err))

    q_l1l = q_l1l.squeeze().detach().numpy().ravel()
    B = np.ones((M, N-1))
    B_mask = sp.linalg.block_diag(*[np.diag(B[i, :], -1) for i in range(M)]).astype(bool)
    err = np.abs(Q[B_mask] - q_l1l)
    print("q_l1l: ", np.max(err))

    q_lNl = q_lNl.squeeze().detach().numpy().ravel()
    C = np.ones((M-1, N))
    C_mask = np.diag(C.ravel(), -N).astype(bool)
    err = np.abs(Q[C_mask] - q_lNl)
    print("q_lNl: ", np.max(err))

if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    check_inverse_l1norm()