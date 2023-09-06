#include <torch/extension.h>

// CUDA kernel forward declarations
torch::Tensor forward_substitution_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor Y);
torch::Tensor backward_substitution_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor Y);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor forward_substitution(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor X) {
    // Check inputs
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);
    CHECK_INPUT(X);

    // Initialize output array & run
    torch::Tensor Y = torch::clone(X);
    return forward_substitution_cuda(A, B, C, Y);
}

torch::Tensor backward_substitution(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor X) {
    // Check inputs
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);
    CHECK_INPUT(X);

    // Initialize output array & run
    torch::Tensor Y = torch::clone(X);
    return backward_substitution_cuda(A, B, C, Y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_substitution", &forward_substitution, "Forward substitution");
  m.def("backward_substitution", &backward_substitution, "Backward substitution");
}