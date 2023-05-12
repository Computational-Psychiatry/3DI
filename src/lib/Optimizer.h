/*
 * Optimizer.h
 *
 *  Created on: Aug 9, 2020
 *      Author: root
 */

#include "cuda.h"
#include "funcs.h"
#include "config.h"
#include "newfuncs.h"
#include "constants.h"
#include "renderer.h"
#include "rotation_header.h"

#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>


#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

struct OptimizationVariables
{
	bool use_identity;
	bool use_texture;
	bool use_expression;

	float *XALL;

	float *tauxs;
	float *taux, *tauy, *tauz;
	float *u1, *u2, *u3;
	float *u;

	float *alphas;
	float *betas;
	float *epsilons;

	float *grad_corr;

	float *slack;
	float *lambdas;
	float *Lintensities;

	float *lambda;
	float *Lintensity;

	float *tau_logbarrier;

	bool for_landmarks;

	uint Kalpha;
	uint Kbeta;
	uint Kepsilon;

	uint Ktotal;
	uint T; // total number of frames
	uint offset;

    uint cframe;

	OptimizationVariables(uint _T, ushort _Kalpha, ushort _Kbeta, ushort _Kepsilon,
			bool use_identity_, bool use_texture_, bool use_expression_,
			bool _for_landmarks = false, bool _with_slack = false) :
		T(_T), Kalpha(_Kalpha), Kbeta(_Kbeta), Kepsilon(_Kepsilon),
		use_identity(use_identity_), use_texture(use_texture_), use_expression(use_expression_),
		for_landmarks(_for_landmarks)
    {
		int Kper_frame = 3 + 3 + Kepsilon*use_expression;
		Ktotal = Kper_frame*T + Kalpha*use_identity + Kbeta*use_texture;
		uint Ktotal__ = Kper_frame*T + Kalpha + Kbeta;

		HANDLE_ERROR( cudaMalloc( (void**)&XALL, sizeof(float)*(Ktotal__)*250) );
		cudaMemset(XALL, 0.0f, Ktotal*sizeof(float));

	    offset = Kalpha*use_identity + Kbeta*use_texture;

	    betas = XALL;
	    alphas = betas + Kbeta*use_texture;
		epsilons = alphas + Kalpha*use_identity ;
		tauxs = epsilons + Kepsilon*T*use_expression;

	    slack = tauxs + 6*T;

	    grad_corr = slack + 1;
	    tau_logbarrier = grad_corr + 1;

	    lambdas = tau_logbarrier + 1;
	    Lintensities = lambdas + 3*T;

	    set_frame(0);

//	    HANDLE_ERROR( cudaMalloc( (void**)&epsilons, sizeof(float)*NEXP_COEFS) );
	}

	void init_from_orthographic(const float *d_orthographic)
	{

	}

	void reset_tau_logbarrier()
	{
		float yummy[1] = {10.0f};
		cudaMemcpy(tau_logbarrier, yummy, sizeof(float), cudaMemcpyHostToDevice);
	}

	void reset()
	{
		int Kper_frame = 3 + 3 + Kepsilon*use_expression;
		uint Ktotal__ = Kper_frame*T + Kalpha + Kbeta;
		cudaMemset(betas, 0, sizeof(float)*Ktotal__*250);
	}


	void set_frame(const uint t) {

        cframe = t;
		epsilons = alphas + Kalpha*use_identity + Kepsilon*t*use_expression;
		taux = tauxs + 6*t;
		tauy = taux + 1;
		tauz = tauy + 1;

		u = tauz + 1;

		lambda = lambdas + 3*t;
		Lintensity = Lintensities + t;
	}

	void print_vars() const
	{
		float c_vars[Ktotal];
		cudaMemcpy(c_vars, betas, Ktotal*sizeof(float), cudaMemcpyDeviceToHost);

		for (int i=0; i<Ktotal; ++i) {
			std::cout << c_vars[i] << ' ';
		}

		std::cout << std::endl;

	}


	~OptimizationVariables()
	{
	    HANDLE_ERROR( cudaFree(XALL) );
//	    HANDLE_ERROR( cudaFree(epsilons) );
	}

};



















struct Optimizer
{

	bool use_identity;
	bool use_texture;
	bool use_expression;
	/**
	 * This is where the diffuse component of the illumination model will be stored
	 */
	float *d_Id_, *d_tex_torender, *d_dI_dlambda, *d_tex_unsorted, *d_tex;
    float *d_TEX_ID_NREF;

    float *vx, *vy, *vz; 		// view-transformed 3D coordinates of the points THAT WILL BE RENDERED
    float *px, *py, *pz; 		// non-view-transformed 3D coordinates (i.e. they include non-rigid variations)
    float *inv_vz, *inv_vz2;	// these are needed for optimization, they are just 1./(vz) and 1./(vz^2)

    float *dI_dalpha; // Eq. 19 in report; matrix of size (Nrendered_pixels x K_alpha)
    float *dI_dbeta; // 
    float *dI_depsilons;

    float *dI_dtaux, 	*dI_dtauy, 	*dI_dtauz;
    float *dI_du1, 		*dI_du2, 	*dI_du3;
/*
    float *dgx_dtaux, 	*dgx_dtauy, 	*dgx_dtauz;
    float *dgx_du1, 	*dgx_du2, 		*dgx_du3;

    float *dgy_dtaux, 	*dgy_dtauy, 	*dgy_dtauz;
    float *dgy_du1, 	*dgy_du2, 		*dgy_du3;
*/
    float *dgx_dtheta, *dgy_dtheta;
    float *dgx_dlambda, *dgy_dlambda;
    float *dgx_dLintensity, *dgy_dLintensity;


    float *gx, *gy;
    float *gxs, *gys;

    float *gx2, *gy2;
    float *gxs2, *gys2;

    float *gx_norm, *gy_norm;
    float *gxs_norm, *gys_norm;

    float *diffx, *diffy;
    float *h, *hs;

    float *JTJ;
    float *dG_dtheta;

    float *grad_corrx;
    float *grad_corry;

    float *dId_dlambda;
    float *dI_dLintensity;

    float *JTJ_lambda;
    float *JTJ_Lintensity;

    float *dG_dtheta_lambda;


	ushort *d_KSX, *d_ks_left, *d_ks_right, *d_ks_above, *d_ks_below, *d_ks_unsorted;
	ushort *d_kl_rel, *d_kr_rel, *d_ka_rel, *d_kb_rel;

	ushort *d_kl_rel_sorted, *d_kr_rel_sorted, *d_ka_rel_sorted, *d_kb_rel_sorted;

	ushort *d_ks_sorted;
	ushort *d_ks_sorted_copy;

	ushort *d_ks_sortidx;
	ushort *d_ks_sortidx_sortidx;
	ushort *d_ks_sortidx_copy;

	ushort *d_cumM0;
	ushort *d_M0;

	ushort Kalpha, Kbeta, Kepsilon;

	OptimizationVariables* ov_ptr;

	Optimizer(OptimizationVariables* _ov_ptr, ushort _Kalpha, ushort _Kbeta, ushort _Kepsilon,
			bool use_identity_, bool use_texture_, bool use_expression_) :
		ov_ptr(_ov_ptr), Kalpha(_Kalpha), Kbeta(_Kbeta), Kepsilon(_Kepsilon),
		use_identity(use_identity_), use_texture(use_texture_), use_expression(use_expression_)
	{

	    int Ktotal = ov_ptr->Ktotal;
	    int T = ov_ptr->T;

	    HANDLE_ERROR( cudaMalloc( (void**)&d_TEX_ID_NREF, sizeof(float)*Nrender_estimated*3850*T) );


	    uint idx_mult = 0;
	    d_tex_unsorted 	= d_TEX_ID_NREF;
	    d_tex 			= d_tex_unsorted + Nrender_estimated;
        d_tex_torender 	= d_tex + config::NPTS;
	    d_Id_ 	= d_tex_torender + Nrender_estimated;
	    d_dI_dlambda = d_Id_ + Nrender_estimated;

	    vx 		= d_dI_dlambda + 3*Nrender_estimated;
	    vy 		= vx + Nrender_estimated;
	    vz 		= vy + Nrender_estimated;
	    px 		= vz + Nrender_estimated;
	    py 		= px + Nrender_estimated;
	    pz 		= py + Nrender_estimated;
	    inv_vz  = pz + Nrender_estimated;
	    inv_vz2 = inv_vz + Nrender_estimated;


	    dI_dbeta = inv_vz2 + Nrender_estimated;
	    dI_dalpha = dI_dbeta + Nrender_estimated*Kbeta*use_texture;

	    dI_depsilons = dI_dalpha + Nrender_estimated*Kalpha*use_identity;


	    dI_dtaux = dI_depsilons + Nrender_estimated*Kepsilon*T*use_expression;
	    dI_dtauy = dI_dtaux + Nrender_estimated;
	    dI_dtauz = dI_dtauy + Nrender_estimated;
	    dI_du1 = dI_dtauz + Nrender_estimated;
	    dI_du2 = dI_du1 + Nrender_estimated;
	    dI_du3 = dI_du2 + Nrender_estimated;


	    dgx_dtheta = dI_dtaux 	+ 	Nrender_estimated*6*T;
	    dgy_dtheta = dgx_dtheta + 	Nrender_estimated*Ktotal;

	    gx 		= dgy_dtheta + Nrender_estimated*Ktotal;
	    gy 		= gx + Nrender_estimated;
	    gxs 	= gy + Nrender_estimated;
	    gys 	= gxs + Nrender_estimated;

	    gx2 	= gys + Nrender_estimated;
	    gy2 	= gx2 + Nrender_estimated;
	    gxs2 	= gy2 + Nrender_estimated;
	    gys2 	= gxs2 + Nrender_estimated;

	    gx_norm 	= gys2 + Nrender_estimated;
	    gy_norm 	= gx_norm + Nrender_estimated;
	    gxs_norm 	= gy_norm + Nrender_estimated;
	    gys_norm 	= gxs_norm + Nrender_estimated;
	    h 	= gys_norm + Nrender_estimated;
	    hs 	= h + Nrender_estimated;

		JTJ = hs + Nrender_estimated;
		dG_dtheta = JTJ + Ktotal*Ktotal;
		grad_corrx = dG_dtheta + Ktotal;
		grad_corry = grad_corrx + 1;

		dId_dlambda = grad_corry + 1;
		dI_dLintensity = dId_dlambda + 3*Nrender_estimated;

		JTJ_lambda = dI_dLintensity + Nrender_estimated;
		JTJ_Lintensity = JTJ_lambda + 3*3;

		dgx_dlambda = JTJ_Lintensity + 1;
		dgy_dlambda = dgx_dlambda + Nrender_estimated*3;

		dG_dtheta_lambda = dgy_dlambda + Nrender_estimated*3;


		dgx_dLintensity  = dG_dtheta_lambda+3;
		dgy_dLintensity  = dgx_dLintensity+Nrender_estimated; // this is of length Nrender_estimated



		// To make memory management simpler, we allocate a block array d_KSX in which we will
		// point to all variables d_ks_.... etc via pointer arithmetic
		HANDLE_ERROR(cudaMalloc((void**)&d_KSX, sizeof(ushort)*(Nrender_estimated*19+NTOTAL_PIXELS*2)));

		d_ks_unsorted = d_KSX;
		d_ks_left 	= d_KSX+Nrender_estimated;
		d_ks_right 	= d_KSX+Nrender_estimated*2;
		d_ks_above 	= d_KSX+Nrender_estimated*3;
		d_ks_below 	= d_KSX+Nrender_estimated*4;
		d_kl_rel 	= d_KSX+Nrender_estimated*5;
		d_kr_rel 	= d_KSX+Nrender_estimated*6;
		d_ka_rel 	= d_KSX+Nrender_estimated*7;
		d_kb_rel 	= d_KSX+Nrender_estimated*8;

		d_ks_sortidx  			= d_KSX+Nrender_estimated*9;
		d_ks_sortidx_sortidx  	= d_KSX+Nrender_estimated*10;
		d_ks_sorted 			= d_KSX+Nrender_estimated*11;
		d_ks_sorted_copy 		= d_KSX+Nrender_estimated*12;
		d_ks_sortidx_copy 		= d_KSX+Nrender_estimated*13;

		d_kl_rel_sorted 	= d_KSX+Nrender_estimated*14;
		d_kr_rel_sorted 	= d_KSX+Nrender_estimated*15;
		d_ka_rel_sorted 	= d_KSX+Nrender_estimated*16;
		d_kb_rel_sorted 	= d_KSX+Nrender_estimated*17;
		d_M0  = d_kb_rel_sorted + Nrender_estimated;
		d_cumM0  = d_M0 + NTOTAL_PIXELS;



		/*
	    HANDLE_ERROR( cudaMalloc( (void**)&d_ks_sortidx,   sizeof(ushort)*Nrender_estimated) );
	    HANDLE_ERROR( cudaMalloc( (void**)&d_ks_sortidx_sortidx,   sizeof(ushort)*Nrender_estimated) );
	    */

/*
		HANDLE_ERROR( cudaMalloc( (void**)&d_M0, sizeof(ushort)*NTOTAL_PIXELS) );
		HANDLE_ERROR( cudaMalloc( (void**)&d_cumM0, sizeof(ushort)*NTOTAL_PIXELS) );
*/


//		HANDLE_ERROR( cudaMalloc( (void**)&JTJ, sizeof(float)*34*34) );
//		HANDLE_ERROR( cudaMalloc( (void**)&dG_dtheta, sizeof(float)*34) );
/************
		HANDLE_ERROR( cudaMalloc( (void**)&JTJ, sizeof(float)*Ktotal*Ktotal) );
		HANDLE_ERROR( cudaMalloc( (void**)&dG_dtheta, sizeof(float)*Ktotal) );
		HANDLE_ERROR( cudaMalloc( (void**)&grad_corrx, sizeof(float)) );
		HANDLE_ERROR( cudaMalloc( (void**)&grad_corry, sizeof(float)) );
***********/
	}


	~Optimizer()
	{
	    HANDLE_ERROR( cudaFree( d_TEX_ID_NREF ));
	    HANDLE_ERROR( cudaFree( d_KSX ));
	    /*
	    HANDLE_ERROR( cudaFree( d_ks_sortidx ) );
	    HANDLE_ERROR( cudaFree( d_ks_sortidx_sortidx ) );
	    */
/*
	    HANDLE_ERROR( cudaFree(d_cumM0) );
	    HANDLE_ERROR( cudaFree(d_M0) );
*/
/**************
	    HANDLE_ERROR( cudaFree(JTJ) );
	    HANDLE_ERROR( cudaFree(dG_dtheta) );
	    HANDLE_ERROR( cudaFree(grad_corrx) );
	    HANDLE_ERROR( cudaFree(grad_corry) );
*************/
	}
};





















#endif /* OPTIMIZER_H_ */
