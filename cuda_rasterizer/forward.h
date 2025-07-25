/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
	    int channels,
	    int P,
		const float* orig_points,
		const glm::vec3* scales,
		float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		int W, int H,
		float focal_x, float focal_y,
		float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float4* conic_opacity,
		dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
	    int channels,
		dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_features,
		float* out_features);

	void filter_preprocess(
		int P,
		const float* means3D,
		const glm::vec3* scales,
		float scale_modifier,
		const glm::vec4* rotations,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		int W, int H,
		float focal_x, float focal_y,
		float tan_fovx, float tan_fovy,
		int* radii,
		float* cov3Ds,
		dim3 grid,
		bool prefiltered);
}
