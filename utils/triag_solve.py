import torch
from torch.autograd import Function, gradcheck
from torch.autograd.function import once_differentiable
import numpy as np
import scipy as sp
import time

import triag_solve_cuda


def forward_substitution(A, B, C, X):
    """
    Solves J*x=y where J is lower triangular matrix, represented by A: M x N (center pixels), B: M x N-1
    (left pixels), C: M-1 x N (above pixels). y = vec(Y: M x N) and x = vec(Y: M x N).
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
    Solves J*y=x where J is upper triangular matrix, represented by A: M x N (center pixels), B: M x N-1
    (right pixels), C: M-1 x N (below pixels). y = vec(Y: M x N) and x = vec(Y: M x N).
    """
    M, N = A.shape[-2:]
    Y = torch.clone(X)
    for i in reversed(range(M)):
        for j in reversed(range(N)):
            if i < M-1:
                tmp = Y[..., i, j] - Y[..., i + 1, j] * C[..., i, j]
                Y[..., i, j] = tmp
            if j < N-1:
                tmp = Y[..., i, j] - Y[..., i, j + 1] * B[..., i, j]
                Y[..., i, j] = tmp
            tmp = Y[..., i, j] / A[..., i, j]
            Y[..., i, j] = tmp
            #print(tmp)

    return Y

def my_matmul(A, B, C, Y):
    """
    Compute J*y=x where J is upper triangular matrix, represented by A: M x N (center pixels), B: M x N-1
    (right pixels), C: M-1 x N (below pixels). y = vec(Y: M x N) and x = vec(Y: M x N).
    """

    M, N = A.shape[-2:]
    X = torch.zeros_like(Y)
    for i in range(M):
        for j in range(N):
            X[..., i, j] = Y[..., i, j] * A[..., i, j]
            if i < M - 1:
                X[..., i, j] += Y[..., i+1, j] * C[..., i, j]
            if j < N - 1:
                X[..., i, j] += Y[..., i, j+1] * B[..., i, j]
    return X

def backward_substitution_iter(A, B, C, X,
                               EPSILON=1e-5,
                               max_iterations=10):
    R = X
    D = 0
    done = False
    i = 0
    while not done:
        Y = backward_substitution(A, B, C, R)
        Y = Y - D
        X0 = my_matmul(A, B, C, Y)
        R = X0 - X
        err = R.abs().max()
        i += 1
        print(f"@{i}: {err:.5f}")

        omega = (R/(my_matmul(A.abs(), B.abs(), C.abs(), Y.abs()) + X.abs())
                /(1 + A.shape[-1] * A.shape[-2])).max()
        print(f"{omega=:.6f}")

        if omega <= EPSILON:
            break
        if i > max_iterations:
            break

    print(f"Used {i} iterations.")
    return Y

class ForwardSubst(Function):
    @staticmethod
    def forward(ctx, A, B, C, X):
        Y = triag_solve_cuda.forward_substitution(A, B, C, X)
        #Y = forward_substitution(A, B, C, Y)
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
        #Y = backward_substitution(A, B, C, Y)
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
    H = np.zeros_like(A)
    for i in range(A.size(0)):
        for j in range(A.size(1)):
            # Initialize unit vector
            X = torch.zeros_like(A)
            X[:,:,i,j] = 1
            # Solve the system
            Y = triag_solve_cuda.forward_substitution(A, B, C, X)
            H[:,:,i,j] = torch.pow(Y[:,:,i,j], 2)

    return H


def check_gradient():
    M = 10   # Rows
    N = 10   # Cols
    A = np.exp(5)*torch.ones((2, 2, M, N), requires_grad=True, dtype=torch.double) #.cuda()
    B = torch.randn(2, 2, M, N-1, requires_grad=True, dtype=torch.double) #.cuda() # left
    C = torch.randn(2, 2, M-1, N, requires_grad=True, dtype=torch.double) #.cuda() # up
    X = torch.randn(2, 2, M, N, requires_grad=True, dtype=torch.double) #.cuda()

    # Exact gradient
    res = gradcheck(ForwardSubst().apply, (A, B, C, X))
    if res:
        print("ForwardSubst() gradient ok!")
    res = gradcheck(BackwardSubst().apply, (A, B, C, X))
    if res:
        print("BackwardSubst() gradient ok!")


def check_solver():
    #M = 200  # Rows
    #N = 200  # Cols
    #A = 1.3*torch.ones((1, M, N))
    M = 150  # Rows
    N = 150  # Cols
    #A = 1 + 0.2*torch.ones((1, M, N))
    A = 0.1 + 0.2*torch.randn((1, M, N)).abs()

    cond_num = A.max() / A.min()
    print(f"{cond_num=}")
    print(f"Norm to A-1: {1/A.min()}")

    B = torch.randn(1, M, N - 1)  # left
    C = torch.randn(1, M - 1, N)  # up
    X = torch.randn(1, M, N)

    #A[:, :, :, :-1] += B.abs()
    #A[:, :, :-1, :] += C.abs()
    #A[:, :, :, 1:] += B.abs()
    #A[:, :, 1:, :] += C.abs()

    #B = torch.ones((1, M, N - 1))
    #C = torch.zeros((1, M - 1, N))
    #X = torch.ones(1, M, N)
    #B = torch.randn(1, M, N - 1)  # right
    #C = torch.randn(1, M - 1, N)  # up
    #X = torch.randn(1, M, N)

    #A[:, :, :, 1:] += B
    #A[:, :, 1:, :] += C

    # Solve in python
    #t = time.time()
    Y = backward_substitution_iter(A, B, C, X)
    X2 = my_matmul(A, B, C, Y)

    #Y = triag_solve_cuda.backward_substitution(A.cuda(), B.cuda(), C.cuda(), Y.cuda())
    #print("Python:", time.time() - t)
    diff = X-X2
    #print(f"{Y=}\n{X=}")
    #print(f"{X2=}")
    #print(f"{diff=}")
    print('min: ', torch.min(diff))
    print('max: ', torch.max(diff))

    # Solve in cpp
    #t = time.time()
    #Yp = triag_solve_cuda.backward_substitution(A.cuda(), B.cuda(), C.cuda(), Y.cuda())
    ##print("Cpp:", time.time() - t)
    #print("Max err:", torch.max(torch.abs(Y - Yp.cpu())))

    # Solve with vectorization
    #A = A.squeeze().numpy()
    #B = B.squeeze().numpy()
    #C = C.squeeze().numpy()
    #Y = Y.squeeze().numpy()
    #a = A.ravel()   # Row-wise stacking
    #b = B.ravel()
    #c = C.ravel()
    #x = Y.ravel()
    #A_mat = np.diag(a)
    #B_mat = sp.linalg.block_diag(*[np.diag(B[i, :], -1) for i in range(M)])
    #C_mat = np.diag(c, -N)
    #H_mat = A_mat + B_mat + C_mat
    #np.set_printoptions(precision=3)
    #y = sp.linalg.solve_triangular(H_mat.T, x)
    #Y = y.reshape(M, N)
    #print(Y)



if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    #check_gradient()
    check_solver()