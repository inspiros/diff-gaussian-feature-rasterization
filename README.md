Differentiable Gaussian Features Rasterization
------

This is a modified Gaussian rasterization submodule that can rasterize _arbitrary $d$-dimensional_ feature vectors
$f\in\mathbb{R}^{\left|\mathcal{G}\right| \times d}$ attached to Gaussians $\mathcal{G}$ into a 2D feature image
$F\in\mathbb{R}^{d \times h \times w}$ whose elements $F(p)$ is computed as:

```math
F(p) = \sum_{i=1}^{\left|\mathcal{G}_p\right|}{f_{g_i} \alpha_{g_i} \sum_{j=1}^{i - 1}{\left(1 - \alpha_{g_j}\right)}}
```

### Installation

Due to limitations of the implementation and CUDA, dimensionality of the feature vector must be known at compile time
so that a CUDA kernel can be generated for the desired dimensionality, and to be selected at runtime.
Please check and add your dimension in the predefined list in `cuda_rasterizer/config.h` before compiling:

```C++
#define DGRF_PREDEFINED_CHANNELS 1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 50, 64, 100, 128, 256, 512, 1024, 2048
```

To install, run (requires C++ and CUDA compiler):

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
