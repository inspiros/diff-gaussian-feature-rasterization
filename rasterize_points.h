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

#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeGaussiansFeaturesCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& features,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	float tan_fovx,
	float tan_fovy,
    int image_height,
    int image_width,
	bool prefiltered,
	bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeGaussiansFeaturesBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& features,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	float tan_fovx,
	float tan_fovy,
    const torch::Tensor& dL_dout_features,
	const torch::Tensor& geomBuffer,
	int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	bool debug);

torch::Tensor rasterizeGaussiansFilterCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	float tan_fovx,
	float tan_fovy,
	int image_height,
	int image_width,
	bool prefiltered,
	bool debug);

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix);
