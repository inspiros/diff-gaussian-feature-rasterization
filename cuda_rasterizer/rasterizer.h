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

#include <functional>

namespace CudaRasterizer {
	class Rasterizer {
	public:
		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			int channels,
			int P,
			const float* background,
			int width, int height,
			const float* means3D,
			const float* features,
			const float* opacities,
			const float* scales,
			float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			float tan_fovx, float tan_fovy,
			bool prefiltered,
			float* out_features,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
		    int channels,
			int P, int R,
			const float* background,
			int width, int height,
			const float* means3D,
			const float* features,
			const float* scales,
			float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dfeatures,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dscale,
			float* dL_drot,
			bool debug);

		static void visible_filter(
			std::function<char* (size_t)> geometryBuffer,
			int P,
			int width, int height,
			const float* means3D,
			const float* scales,
			float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			float tan_fovx, float tan_fovy,
			bool prefiltered,
			int* radii,
			bool debug);
	};
}
