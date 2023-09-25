#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void forward_substitution_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> B,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> C,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> Y) {

    int M = A.size(2);
    int N = A.size(3);

    // Get the index of thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / A.size(1);  // batch
    int j = index % A.size(1);  // channel

    if (i < A.size(0) && j < A.size(1)) {
        for (int k = 0; k < M; k++) {
            for (int l = 0; l < N; l++) {
                if (k > 0)
                    Y[i][j][k][l] = Y[i][j][k][l] -  Y[i][j][k - 1][l] * C[i][j][k - 1][l];
                if (l > 0)
                    Y[i][j][k][l] = Y[i][j][k][l] - Y[i][j][k][l - 1] * B[i][j][k][l - 1];

                Y[i][j][k][l] = Y[i][j][k][l] / A[i][j][k][l];
            }
        }
    }
}

template <typename scalar_t>
__global__ void backward_substitution_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> B,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> C,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> Y) {

    int M = A.size(2);
    int N = A.size(3);

    // Get the index of thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / A.size(1);  // batch
    int j = index % A.size(1);  // channel

    if (i < A.size(0) && j < A.size(1)) {
        for (int k = M-1; k >= 0; k--) {        // y axis
            for (int l = N-1; l >= 0; l--) {    // x axis
                if (k < M-1)
                    Y[i][j][k][l] = Y[i][j][k][l] -  Y[i][j][k + 1][l] * C[i][j][k][l];
                if (l < N-1)
                    Y[i][j][k][l] = Y[i][j][k][l] - Y[i][j][k][l + 1] * B[i][j][k][l];

                Y[i][j][k][l] = Y[i][j][k][l] / A[i][j][k][l];
            }
        }
    }
}


template <typename scalar_t>
__global__ void inverse_diagonal_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> B,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> C,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> Y,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> H) {

    int M = A.size(2);
    int N = A.size(3);

    // Get the index of thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int m = index / A.size(1);  // batch
    int n = index % A.size(1);  // channel

    scalar_t sum;
    //scalar_t c, t, y;

    if (m < A.size(0) && n < A.size(1)) {
        for (int k = 0; k < M; k++){
            for (int l = 0; l < N; l++){
                sum = 0.0;
                //c = 0.0;

                // For i=k iterate over j=l,...,N
                for (int j = l; j < N; j++){
                    if (j == l)
                        Y[m][n][k][j] = 1.0;
                    if (j > l)
                        Y[m][n][k][j] = -Y[m][n][k][j - 1] * B[m][n][k][j - 1];

                    Y[m][n][k][j] = Y[m][n][k][j] / A[m][n][k][j];
                    //sum = sum + pow(Y[m][n][k][j], 2);
                    sum = sum + Y[m][n][k][j]*Y[m][n][k][j];
                    // Kahan summation
                    //y = pow(Y[m][n][k][j], 2) - c;
                    //t = sum + y;
                    //c = (t - sum) - y;
                    //sum = t;
                }

                // For i=k+1,...,M iterate over j=0,...,N
                for (int i = k+1; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        Y[m][n][i][j] = 0.0;

                        if (j >= l || i > k+1)  // If i=k+1 only upper neighbors j=l,...,N are valid
                            Y[m][n][i][j] = Y[m][n][i][j] -  Y[m][n][i - 1][j] * C[m][n][i - 1][j];
                        if (j > 0)
                            Y[m][n][i][j] = Y[m][n][i][j] - Y[m][n][i][j - 1] * B[m][n][i][j - 1];

                        Y[m][n][i][j] = Y[m][n][i][j] / A[m][n][i][j];
                        //sum = sum + pow(Y[m][n][i][j], 2);
                        sum = sum + Y[m][n][i][j]*Y[m][n][i][j];
                        // Kahan summation
                        //y = pow(Y[m][n][i][j], 2) - c;
                        //t = sum + y;
                        //c = (t - sum) - y;
                        //sum = t;
                    }
                }

                H[m][n][k][l] = sum;
            }
        }
    }
}

torch::Tensor forward_substitution_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor Y) {
    // Compute required number of blocks and threads
    int block_size = 32;
    int num_blocks = ceil((A.size(0) * A.size(1)) / (float)block_size);

    // Call corresponding CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "forward_substitution_kernel", ([&] {
        forward_substitution_kernel<scalar_t><<<num_blocks, block_size>>>(
            A.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            B.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            Y.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
    }));

    return Y;
}

torch::Tensor backward_substitution_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor Y) {
    // Compute required number of blocks and threads
    int block_size = 32;
    int num_blocks = ceil((A.size(0) * A.size(1)) / (float)block_size);

    // Call corresponding CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "backward_substitution_kernel", ([&] {
        backward_substitution_kernel<scalar_t><<<num_blocks, block_size>>>(
            A.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            B.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            Y.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
    }));

    return Y;
}

torch::Tensor inverse_diagonal_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // Output & auxiliary tensors
    torch::Tensor H = torch::zeros_like(A);
    torch::Tensor Y = torch::zeros_like(A);

    // Compute required number of blocks and threads
    int block_size = 32;
    int num_blocks = ceil((A.size(0) * A.size(1)) / (float)block_size);

    // Call corresponding CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "inverse_diagonal_kernel", ([&] {
        inverse_diagonal_kernel<scalar_t><<<num_blocks, block_size>>>(
            A.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            B.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            Y.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            H.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
    }));

    return H;
}
