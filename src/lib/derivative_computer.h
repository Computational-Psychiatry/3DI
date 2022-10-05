/*
 * derivative_computer.h
 *
 *  Created on: Aug 25, 2020
 *      Author: root
 */

#ifndef DERIVATIVE_COMPUTER_H_
#define DERIVATIVE_COMPUTER_H_

#include "Optimizer.h"
#include "renderer.h"
#include "rotation_header.h"
#include "solver.h"
#include "logbarrier_initializer.h"

#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>








struct DerivativeComputer
{
	bool use_identity;
	bool use_texture;
	bool use_expression;

	bool use_inequalities;

	DerivativeComputer(bool _use_inequalities, bool use_identity_, bool use_texture_, bool use_expression_) :
				use_identity(use_identity_), use_texture(use_texture_), use_expression(use_expression_),
				use_inequalities(_use_inequalities) {}

	void compute_hessian_and_gradient(ushort t, Optimizer& o,
			RotationComputer& rc, Renderer& r,
			Camera& cam,
			ushort &N_unique_pixels,
			cublasHandle_t &handle,
			Logbarrier_Initializer& li);

	void compute_hessian_and_gradient_for_lambda(Optimizer& o,
            Solver& s_illum, RotationComputer& rc,
			ushort &N_unique_pixels,
			cusolverDnHandle_t& handleDn,
            cublasHandle_t &handle,
			bool set_ks);


	void compute_hessian_and_gradient_for_Lintensity(Optimizer& o,
            ushort &N_unique_pixels,
            cublasHandle_t &handle,
			float *search_dir_Lintensity,
			float *dg_Lintensity,
			bool set_ks);

};

#endif /* DERIVATIVE_COMPUTER_H_ */
