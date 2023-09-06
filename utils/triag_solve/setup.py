from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='triag_solve_cuda',
      ext_modules=[
            CUDAExtension('triag_solve_cuda', [
                  'triag_solve.cpp',
                  'triag_solve_cuda.cu'
            ])
      ],
      cmdclass={
            'build_ext': BuildExtension
      })