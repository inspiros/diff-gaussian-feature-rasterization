Differentiable Gaussian Features Rasterization
------

### Introduction

This is a modified Gaussian rasterization submodule that can rasterize _arbitrary_[^1] $d$-dimensional feature vectors
$f\in\mathbb{R}^{\left|\mathcal{G}\right| \times d}$ attached to Gaussians $\mathcal{G}$ into a 2D feature image
$F\in\mathbb{R}^{d \times h \times w}$ whose elements $F(p)$ is computed as:

[^1]: You have to register all expected $d$, and the maximum $d$ is actually $d=12281$, see [this section](#runtime-dynamic-block-resolution).

```math
F(p) = \sum_{i=1}^{\left|\mathcal{G}_p\right|}{f_{g_i} \alpha_{g_i} \sum_{j=1}^{i - 1}{\left(1 - \alpha_{g_j}\right)}}
```

#### Registering Feature Dimensions

Due to limitations of the algorithm and CUDA, dimensionality of the feature vector must be known at compile time.
Here we define a list of all known dimensionalities and generate different CUDA kernels for each desired dimensionality,
and select them at runtime.

#### Runtime Dynamic Block Resolution

Originally, the rasterizer uses a block of $16 \times 16$.
In the backward pass, there is a shared memory array which has size grow linearly with this block size.
Hence, for higher dimensional feature vectors, we have to dynamically select a smaller block size
(i.e. $8 \times 8$, $4 \times 4$) so that the shared memory belows the limit of 48KB.
See [this stackoverflow issue](https://stackoverflow.com/questions/23648525/cuda-ptxas-error-function-uses-too-much-shared-data).

The maximum dimension that fits a block resolution $b$ (assuming that block size is $b \times b$ as implemented)
can be computed as follows:

```math
d_{max} = \frac{12 \cdot 1024 - 7 \cdot b^2}{b^2}
```

If you don't want this behavior, you can split $f$ into multiple chunks and splat them
with a desired block size, then concatenate the rendered features outputs afterward.

### Installation

First, please check and add your predefined dimension to the list in `cuda_rasterizer/config.h` before compiling:

```C++
#define DGRF_PREDEFINED_CHANNELS 1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 50, 64, 100, 128, 256, 512, 1024, 2048
```

Then to install, run (requires C++ and CUDA compilers):

```cmd
pip install .
```

### Usage

```python
from diff_gaussian_feature_rasterization import GaussianFeaturesRasterizationSettings, GaussianFeaturesRasterizer

# features: [n_gaussians, d]
features_raster_settings = GaussianFeaturesRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=torch.new_zeros([features.size(-1)]),
    scale_modifier=scaling_modifier,
    viewmatrix=viewpoint_camera.world_view_transform,
    projmatrix=viewpoint_camera.full_proj_transform,
    prefiltered=False,
    debug=pipe.debug
)
features_rasterizer = GaussianFeaturesRasterizer(raster_settings=features_raster_settings)

# rendered_features: [d, h, w]
rendered_features, radii = features_rasterizer(
    means3D=xyz,
    means2D=screenspace_points,
    shs=None,
    colors_precomp=features,
    opacities=opacity,
    scales=scaling,
    rotations=rot,
    cov3D_precomp=None)
```

### Acknowledgement and Licensing

This library is a derivative of my M2 internship at **L2S - CentraleSupélec**.
While it was not proven to be useful for me, I believe other researchers will find good use for it.

Also, cite the original authors' paper:

```bibtex
@article{kerbl3Dgaussians,
    author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal      = {ACM Transactions on Graphics},
    number       = {4},
    volume       = {42},
    month        = {July},
    year         = {2023},
    url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

This code's origin can be traced to https://github.com/graphdeco-inria/diff-gaussian-rasterization,
please check their [LICENSE.md](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/LICENSE.md).
