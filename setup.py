#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="diff_gaussian_feature_rasterization",
    author="Hoang-Nhat TRAN",
    author_email="hnhat.tran@gmail.com",
    packages=['diff_gaussian_feature_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_feature_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cpp",
                "ext.cpp"],
            extra_compile_args={
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
