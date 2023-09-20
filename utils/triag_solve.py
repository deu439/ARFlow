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
    (left pixels), C: M-1 x N (above pixels). y = vec(Y: M x N) and x = vec(X: M x N).
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
    Solves J*x=y where J is upper triangular matrix, represented by A: M x N (center pixels), B: M x N-1
    (right pixels), C: M-1 x N (below pixels). y = vec(Y: M x N) and x = vec(X: M x N).
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
    A = np.exp(5)*torch.ones((2, 2, M, N), requires_grad=True, dtype=torch.double).cuda()
    B = torch.randn(2, 2, M, N-1, requires_grad=True, dtype=torch.double).cuda() # left
    C = torch.randn(2, 2, M-1, N, requires_grad=True, dtype=torch.double).cuda() # up
    X = torch.randn(2, 2, M, N, requires_grad=True, dtype=torch.double).cuda()

    # Exact gradient
    res = gradcheck(ForwardSubst().apply, (A, B, C, X))
    if res:
        print("ForwardSubst() gradient ok!")
    res = gradcheck(BackwardSubst().apply, (A, B, C, X))
    if res:
        print("BackwardSubst() gradient ok!")


def check_solver():
    M = 100  # Rows
    N = 150  # Cols
    A = np.exp(-2)*torch.ones((8, 2, M, N))
    B = torch.randn(8, 2, M, N - 1)  # left
    C = torch.randn(8, 2, M - 1, N)  # up
    X = torch.randn(8, 2, M, N)

    A[:, :, :, 1:] += B
    A[:, :, 1:, :] += C

    # Solve in python
    #t = time.time()
    Y = backward_substitution(A, B, C, X)
    #Y = triag_solve_cuda.backward_substitution(A.cuda(), B.cuda(), C.cuda(), X.cuda())
    #print("Python:", time.time() - t)
    print(Y)
    print('min: ', torch.min(Y))
    print('max: ', torch.max(Y))

    # Solve in cpp
    #t = time.time()
    #Yp = triag_solve_cuda.backward_substitution(A.cuda(), B.cuda(), C.cuda(), X.cuda())
    ##print("Cpp:", time.time() - t)
    #print("Max err:", torch.max(torch.abs(Y - Yp.cpu())))

    # Solve with vectorization
    #A = A.squeeze().numpy()
    #B = B.squeeze().numpy()
    #C = C.squeeze().numpy()
    #X = X.squeeze().numpy()
    #a = A.ravel()   # Row-wise stacking
    #b = B.ravel()
    #c = C.ravel()
    #x = X.ravel()
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
    check_solver()