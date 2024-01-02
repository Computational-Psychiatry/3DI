/*
 * renderer.cu
 *
 *  Created on: Aug 10, 2020
 *      Author: root
 */

#include "renderer.h"
#include <fstream>
#include "config.h"
#include <thrust/remove.h>

template<typename T>
struct is_zero {
    __host__ __device__
    auto operator()(T x) const -> bool {
        return x == 0;
    }
};


Renderer::Renderer(uint T_, ushort _Kalpha, ushort _Kbeta, ushort _Kepsilon,
                   bool use_identity_, bool use_texture_, bool use_expression_,
                   float *h_X0, float *h_Y0, float *h_Z0, float *h_tex_mu)
    : d_ones(config::NPTS, 1.f), T(T_), Kalpha(_Kalpha), Kbeta(_Kbeta), Kepsilon(_Kepsilon),
      use_identity(use_identity_), use_texture(use_texture_), use_expression(use_expression_)
{
    HANDLE_ERROR( cudaMalloc( (void**)&ALL_VARS, sizeof(float)*config::NPTS*600) );
    X0_mean = ALL_VARS;
    Y0_mean = X0_mean + config::NPTS;
    Z0_mean = Y0_mean + config::NPTS;

    X0 = Z0_mean + config::NPTS;
    Y0 = X0 + config::NPTS;
    Z0 = Y0 + config::NPTS;

    X  = Z0 + config::NPTS;
    Y  = X + config::NPTS;
    Z  = Y + config::NPTS;

    HANDLE_ERROR(cudaMalloc((void**)&d_grad, sizeof(float)*NTOTAL_PIXELS));
    HANDLE_ERROR(cudaMalloc((void**)&d_texIm, sizeof(float)*NTOTAL_PIXELS));

    cudaMemset(d_texIm, 0, NTOTAL_PIXELS*sizeof(float));

    HANDLE_ERROR( cudaMalloc( (void**)&d_triangle_idx, sizeof(ushort)*config::Nredundant) );


    HANDLE_ERROR(cudaMalloc((void**)&d_xp, sizeof(float)*config::NPTS));
    HANDLE_ERROR(cudaMalloc((void**)&d_yp, sizeof(float)*config::NPTS));

    HANDLE_ERROR(cudaMalloc((void**)&d_alphas_redundant, sizeof(float)*config::Nredundant));
    HANDLE_ERROR(cudaMalloc((void**)&d_betas_redundant, sizeof(float)*config::Nredundant));
    HANDLE_ERROR(cudaMalloc((void**)&d_gammas_redundant, sizeof(float)*config::Nredundant));

    cudaMemset(d_alphas_redundant, 0, sizeof(float)*config::Nredundant);
    cudaMemset(d_betas_redundant, 0, sizeof(float)*config::Nredundant);
    cudaMemset(d_gammas_redundant, 0, sizeof(float)*config::Nredundant);

    HANDLE_ERROR(cudaMalloc((void**)&d_rend_flag, sizeof(bool)*config::Nredundant));
    HANDLE_ERROR(cudaMalloc((void**)&d_rend_flag_tmp, sizeof(bool)*config::Nredundant));

    HANDLE_ERROR(cudaMalloc((void**)&d_tl,  sizeof(ushort)*config::N_TRIANGLES*3));
    HANDLE_ERROR(cudaMalloc((void**)&d_pixel_idx,  sizeof(ushort)*config::N_TRIANGLES*NTMP));
    HANDLE_ERROR(cudaMalloc((void**)&d_redundant_idx, sizeof(uint)*config::Nredundant));

    HANDLE_ERROR( cudaMalloc( (void**)&d_Z, sizeof(float)*config::NPTS) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_Ztmp, sizeof(float)*config::Nredundant) );


    HANDLE_ERROR(cudaMalloc((void**)&d_Zmins, sizeof(float)*DIMX*DIMY));


    // $$$ return;

    /*
    float *d_REX, *d_REY, *d_REZ;
    float *d_RIX, *d_RIY, *d_RIZ;
    float *d_RTEX;
     */
    HANDLE_ERROR(cudaMalloc((void**)&d_REX, sizeof(float)*config::Nredundant*Kepsilon));
    HANDLE_ERROR(cudaMalloc((void**)&d_REY, sizeof(float)*config::Nredundant*Kepsilon));
    HANDLE_ERROR(cudaMalloc((void**)&d_REZ, sizeof(float)*config::Nredundant*Kepsilon));



    if (use_texture)
    {
        HANDLE_ERROR(cudaMalloc((void**)&d_RTEX, sizeof(float)*config::Nredundant*Kbeta));
    }



    /**
     * Number of pts in the matrix matrix that contains the original the expression basis
     * (i.e. the expression basis that comes with the model -- not the one we render)
     * for each coordinate (ie. the total size is size_E x 3)
     */
    const int size_I = config::NPTS*Kalpha;
    const int size_TEX = config::NPTS*Kbeta;




    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    // Mean face shape in 3D -- BEGIN -- //
    if (h_X0 == NULL)
    {
        vector< vector<float> > X0_vec = read2DVectorFromFile<float>(config::X0_PATH, config::NPTS, 1);
        vector< vector<float> > Y0_vec = read2DVectorFromFile<float>(config::Y0_PATH, config::NPTS, 1);
        vector< vector<float> > Z0_vec = read2DVectorFromFile<float>(config::Z0_PATH, config::NPTS, 1);

        /**
         * Those are the 3D points of the face mesh (of a specific
         * person -- i.e. it's not the average face) before view transformation
         */
        float *X0 = vec2arr(X0_vec, config::NPTS, 1, false);
        float *Y0 = vec2arr(Y0_vec, config::NPTS, 1, false);
        float *Z0 = vec2arr(Z0_vec, config::NPTS, 1, false);

        HANDLE_ERROR( cudaMemcpy( X0_mean, X0, sizeof(float)*config::NPTS , cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( Y0_mean, Y0, sizeof(float)*config::NPTS , cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( Z0_mean, Z0, sizeof(float)*config::NPTS , cudaMemcpyHostToDevice ) );

        free(X0);
        free(Y0);
        free(Z0);
    }
    else
    {
        HANDLE_ERROR( cudaMemcpy( X0_mean, h_X0, sizeof(float)*config::NPTS , cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( Y0_mean, h_Y0, sizeof(float)*config::NPTS , cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( Z0_mean, h_Z0, sizeof(float)*config::NPTS , cudaMemcpyHostToDevice ) );
    }
    // Mean face shape in 3D -- END -- //
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////







    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    // Mean texture -- BEGIN -- //
    vector< vector<float> > tex_mu_vec = read2DVectorFromFile<float>(config::TEXMU_PATH, config::NPTS, 1);
    HANDLE_ERROR( cudaMalloc( (void**)&d_mu_tex, sizeof(float)*config::NPTS) );

    if (h_tex_mu == NULL)
    {
        float *tex_mu = vec2arr(tex_mu_vec, config::NPTS, 1, false);

        HANDLE_ERROR( cudaMemcpy( d_mu_tex, tex_mu, sizeof(float)*config::NPTS , cudaMemcpyHostToDevice ) );
        free(tex_mu);
    }
    else
    {
        HANDLE_ERROR( cudaMemcpy( d_mu_tex, h_tex_mu, sizeof(float)*config::NPTS , cudaMemcpyHostToDevice ) );
    }
    // Mean texture -- END -- //
    ///////////////////////////////////////
    ///////////////////////////////////////

    ///////////////////////////////////////
    ///////////////////////////////////////
    // TRIANGLE IDX FOR 3DMM -- END -- //

    tl_vector = read2DVectorFromFile<int>(config::TL_PATH, config::N_TRIANGLES, 3);

    /**
     * Here we store the triangle indices of the face model. I.e., tl is a matrix of
     * size config::N_TRIANGLESx3 (stored in an array as a column-major matrix), whose each row
     * contains the indices of the face mesh points that form one triangle. Thus,
     * the entries of tl are integers in the range of [0, config::NPTS]
     */
    ushort *tl;


    tl = (ushort*)malloc( config::N_TRIANGLES*3*sizeof(ushort) );


    for (uint i=0; i<config::N_TRIANGLES; ++i) {
        tl[config::N_TRIANGLES*0+i] = (ushort)tl_vector[i][0]-1;
        tl[config::N_TRIANGLES*1+i] = (ushort)tl_vector[i][1]-1;
        tl[config::N_TRIANGLES*2+i] = (ushort)tl_vector[i][2]-1;
    }



    HANDLE_ERROR( cudaMemcpy( d_tl, tl, sizeof(ushort)*config::N_TRIANGLES*3, cudaMemcpyHostToDevice ) );

    free(tl);

    // TRIANGLE IDX FOR 3DMM -- END -- //
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////
    ///////////////////////////////////////

    x0_short =  (ushort*)malloc( T*sizeof(ushort) );
    y0_short =  (ushort*)malloc( T*sizeof(ushort) );

    using std::vector;


    vector< vector<float> > IX_vec = read2DVectorFromFile<float>(config::IX_PATH, config::NPTS, config::NID_COEFS);
    vector< vector<float> > IY_vec = read2DVectorFromFile<float>(config::IY_PATH, config::NPTS, config::NID_COEFS);
    vector< vector<float> > IZ_vec = read2DVectorFromFile<float>(config::IZ_PATH, config::NPTS, config::NID_COEFS);

    vector< vector<float> > TEX_vec;

    if (use_texture)
        TEX_vec = read2DVectorFromFile<float>(config::TEX_PATH, config::NPTS, config::NTEX_COEFS);





    float *IX, *IY, *IZ;
    float *IX_row_major, *IY_row_major, *IZ_row_major;
    float *TEX;
    float *TEX_row_major;

    IX = (float*)malloc( size_I*sizeof(float) );
    IY = (float*)malloc( size_I*sizeof(float) );
    IZ = (float*)malloc( size_I*sizeof(float) );

    if (use_texture)
    {
        TEX = (float*)malloc( size_TEX*sizeof(float) );
        TEX_row_major = (float*)malloc( size_TEX*sizeof(float) );
    }

    IX_row_major = (float*)malloc( size_I*sizeof(float) );
    IY_row_major = (float*)malloc( size_I*sizeof(float) );
    IZ_row_major = (float*)malloc( size_I*sizeof(float) );



    for (int i=0; i<config::NPTS; ++i) {
        for (int j=0; j<Kalpha; ++j) {
            int idx = i+j*config::NPTS;
            IX[idx] = IX_vec[i][j];
            IY[idx] = IY_vec[i][j];
            IZ[idx] = IZ_vec[i][j];

        }
    }




    for (uint idx=0; idx<config::NPTS*Kalpha; ++idx) {
        int col = idx % Kalpha;
        int row = idx/Kalpha;
        IX_row_major[idx] = IX_vec[row][col];
        IY_row_major[idx] = IY_vec[row][col];
        IZ_row_major[idx] = IZ_vec[row][col];
    }

    if (use_texture)
    {
        for (int i=0; i<config::NPTS; ++i) {
            for (int j=0; j<Kbeta; ++j) {
                int idx = i+j*config::NPTS;
                TEX[idx] = TEX_vec[i][j]; ///255.0f;
            }
        }

        for (uint idx=0; idx<config::NPTS*Kbeta; ++idx) {
            int col = idx % Kbeta;
            int row = idx/Kbeta;
            TEX_row_major[idx] = TEX_vec[row][col];///255.0f;
        }
    }





    // Now identity bases
    if (use_identity)
    {
        HANDLE_ERROR(cudaMallocPitch((void**)&d_IX_row_major, &pitch2, Kalpha * sizeof(float), config::NPTS));
        HANDLE_ERROR(cudaMallocPitch((void**)&d_IY_row_major, &pitch2, Kalpha * sizeof(float), config::NPTS));
        HANDLE_ERROR(cudaMallocPitch((void**)&d_IZ_row_major, &pitch2, Kalpha * sizeof(float), config::NPTS));

        HANDLE_ERROR(cudaMemcpy2D(d_IX_row_major, pitch2, IX_row_major, Kalpha * sizeof(float), Kalpha*sizeof(float), config::NPTS,  cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy2D(d_IY_row_major, pitch2, IY_row_major, Kalpha * sizeof(float), Kalpha*sizeof(float), config::NPTS,  cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy2D(d_IZ_row_major, pitch2, IZ_row_major, Kalpha * sizeof(float), Kalpha*sizeof(float), config::NPTS,  cudaMemcpyHostToDevice));

        HANDLE_ERROR( cudaMalloc( (void**)&d_IX, sizeof(float)*size_I) );
        HANDLE_ERROR( cudaMalloc( (void**)&d_IY, sizeof(float)*size_I) );
        HANDLE_ERROR( cudaMalloc( (void**)&d_IZ, sizeof(float)*size_I) );

        HANDLE_ERROR( cudaMemcpy( d_IX, IX, sizeof(float)*size_I, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( d_IY, IY, sizeof(float)*size_I, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( d_IZ, IZ, sizeof(float)*size_I, cudaMemcpyHostToDevice ) );


        HANDLE_ERROR(cudaMalloc((void**)&d_RIX, sizeof(float)*config::Nredundant*Kalpha));
        HANDLE_ERROR(cudaMalloc((void**)&d_RIY, sizeof(float)*config::Nredundant*Kalpha));
        HANDLE_ERROR(cudaMalloc((void**)&d_RIZ, sizeof(float)*config::Nredundant*Kalpha));
    }

    // $$$$$$$
    // return;



    // Finally the texture bases
    if (use_texture)
    {
        HANDLE_ERROR( cudaMalloc( (void**)&d_TEX, sizeof(float)*size_TEX) );
        HANDLE_ERROR( cudaMemcpy( d_TEX, TEX, sizeof(float)*size_TEX, cudaMemcpyHostToDevice ) );

        HANDLE_ERROR( cudaMallocPitch((void**)&d_TEX_row_major, &pitch3, Kbeta * sizeof(float), config::NPTS) );
        HANDLE_ERROR( cudaMemcpy2D(d_TEX_row_major, pitch3, TEX_row_major, Kbeta * sizeof(float), Kbeta*sizeof(float), config::NPTS,  cudaMemcpyHostToDevice) );
    }



    free(IX_row_major);
    free(IY_row_major);
    free(IZ_row_major);

    if (use_texture)
    {
        free(TEX);
        free(TEX_row_major);
    }

    free(IX);
    free(IY);
    free(IZ);

    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////
    //  LOADING EXPRESSION BASES ETC
    const int size_E = config::NPTS*Kepsilon;

    vector< vector<float> > EX_vec = read2DVectorFromFile<float>(config::EX_PATH, config::NPTS, config::K_EPSILON);
    vector< vector<float> > EY_vec = read2DVectorFromFile<float>(config::EY_PATH, config::NPTS, config::K_EPSILON);
    vector< vector<float> > EZ_vec = read2DVectorFromFile<float>(config::EZ_PATH, config::NPTS, config::K_EPSILON);

    float *EX, *EY, *EZ;
    float *EX_row_major, *EY_row_major, *EZ_row_major;

    EX = (float*)malloc( size_E*sizeof(float) );
    EY = (float*)malloc( size_E*sizeof(float) );
    EZ = (float*)malloc( size_E*sizeof(float) );

    EX_row_major = (float*)malloc( size_E*sizeof(float) );
    EY_row_major = (float*)malloc( size_E*sizeof(float) );
    EZ_row_major = (float*)malloc( size_E*sizeof(float) );






    for (int i=0; i<config::NPTS; ++i) {
        for (int j=0; j<Kepsilon; ++j) {
            int idx = i+j*config::NPTS;
            EX[idx] = EX_vec[i][j];
            EY[idx] = EY_vec[i][j];
            EZ[idx] = EZ_vec[i][j];

        }
    }


    for (uint idx=0; idx<config::NPTS*Kepsilon; ++idx) {
        int col = idx % Kepsilon;
        int row = idx/Kepsilon;
        EX_row_major[idx] = EX_vec[row][col];
        EY_row_major[idx] = EY_vec[row][col];
        EZ_row_major[idx] = EZ_vec[row][col];
    }



    if (use_expression)
    {
        /**
         * We'll create two copies. One is row_major the other is column major. Row_major is needed for the
         * texture memory (will be used for rendering basis) and column_major is needed for generating 3D points
         */
        HANDLE_ERROR(cudaMallocPitch((void**)&d_EX_row_major, &pitch, Kepsilon * sizeof(float), config::NPTS));
        HANDLE_ERROR(cudaMallocPitch((void**)&d_EY_row_major, &pitch, Kepsilon * sizeof(float), config::NPTS));
        HANDLE_ERROR(cudaMallocPitch((void**)&d_EZ_row_major, &pitch, Kepsilon * sizeof(float), config::NPTS));


        HANDLE_ERROR(cudaMemcpy2D(d_EX_row_major, pitch, EX_row_major, Kepsilon * sizeof(float), Kepsilon*sizeof(float), config::NPTS,  cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy2D(d_EY_row_major, pitch, EY_row_major, Kepsilon * sizeof(float), Kepsilon*sizeof(float), config::NPTS,  cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy2D(d_EZ_row_major, pitch, EZ_row_major, Kepsilon * sizeof(float), Kepsilon*sizeof(float), config::NPTS,  cudaMemcpyHostToDevice));

        HANDLE_ERROR( cudaMalloc( (void**)&d_EX, sizeof(float)*size_E) );
        HANDLE_ERROR( cudaMalloc( (void**)&d_EY, sizeof(float)*size_E) );
        HANDLE_ERROR( cudaMalloc( (void**)&d_EZ, sizeof(float)*size_E) );

        HANDLE_ERROR( cudaMemcpy( d_EX, EX, sizeof(float)*size_E, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( d_EY, EY, sizeof(float)*size_E, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( d_EZ, EZ, sizeof(float)*size_E, cudaMemcpyHostToDevice ) );
    }


    free(EX);
    free(EY);
    free(EZ);

    free(EX_row_major);
    free(EY_row_major);
    free(EZ_row_major);
    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////
    //////////////////////////////////////////

    initialize_texture_memories();

}


__global__ void kernel_print(cudaTextureObject_t tex)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    float val = tex2D<float>(tex, x, y);
    printf("%.5f, ", val);
}




void Renderer::initialize_texture_memories()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("texturePitchAlignment: %lu\n", prop.texturePitchAlignment);
//    size_t pitch;

    dim3 threads(4, 4);

    if (use_identity && Kalpha > 0)
    {
        memset(&ix_resDesc, 0, sizeof(ix_resDesc));
        ix_resDesc.resType = cudaResourceTypePitch2D;
        ix_resDesc.res.pitch2D.devPtr = d_IX_row_major;
        ix_resDesc.res.pitch2D.width = Kalpha;
        ix_resDesc.res.pitch2D.height = config::NPTS;
        ix_resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        ix_resDesc.res.pitch2D.pitchInBytes = pitch2;

        memset(&ix_texDesc, 0, sizeof(ix_texDesc));
        cudaCreateTextureObject(&ix_tex, &ix_resDesc, &ix_texDesc, NULL);


        memset(&iy_resDesc, 0, sizeof(iy_resDesc));
        iy_resDesc.resType = cudaResourceTypePitch2D;
        iy_resDesc.res.pitch2D.devPtr = d_IY_row_major;
        iy_resDesc.res.pitch2D.width = Kalpha;
        iy_resDesc.res.pitch2D.height = config::NPTS;
        iy_resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        iy_resDesc.res.pitch2D.pitchInBytes = pitch2;

        memset(&iy_texDesc, 0, sizeof(iy_texDesc));
        cudaCreateTextureObject(&iy_tex, &iy_resDesc, &iy_texDesc, NULL);


        memset(&iz_resDesc, 0, sizeof(iz_resDesc));
        iz_resDesc.resType = cudaResourceTypePitch2D;
        iz_resDesc.res.pitch2D.devPtr = d_IZ_row_major;
        iz_resDesc.res.pitch2D.width = Kalpha;
        iz_resDesc.res.pitch2D.height = config::NPTS;
        iz_resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        iz_resDesc.res.pitch2D.pitchInBytes = pitch2;

        memset(&iz_texDesc, 0, sizeof(iz_texDesc));
        cudaCreateTextureObject(&iz_tex, &iz_resDesc, &iz_texDesc, NULL);

        /*
        kernel_print<<<1, threads>>>(ix_tex);
        cudaDeviceSynchronize();
        std::cout << "=======================" << std::endl;
        */
    }


    if (use_expression && Kepsilon > 0)
    {
        memset(&ex_resDesc, 0, sizeof(ex_resDesc));
        ex_resDesc.resType = cudaResourceTypePitch2D;
        ex_resDesc.res.pitch2D.devPtr = d_EX_row_major;
        ex_resDesc.res.pitch2D.width = Kepsilon;
        ex_resDesc.res.pitch2D.height = config::NPTS;
        ex_resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        ex_resDesc.res.pitch2D.pitchInBytes = pitch;

        memset(&ex_texDesc, 0, sizeof(ex_texDesc));
        cudaCreateTextureObject(&ex_tex, &ex_resDesc, &ex_texDesc, NULL);


        memset(&ey_resDesc, 0, sizeof(ey_resDesc));
        ey_resDesc.resType = cudaResourceTypePitch2D;
        ey_resDesc.res.pitch2D.devPtr = d_EY_row_major;
        ey_resDesc.res.pitch2D.width = Kepsilon;
        ey_resDesc.res.pitch2D.height = config::NPTS;
        ey_resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        ey_resDesc.res.pitch2D.pitchInBytes = pitch;

        memset(&ey_texDesc, 0, sizeof(ey_texDesc));
        cudaCreateTextureObject(&ey_tex, &ey_resDesc, &ey_texDesc, NULL);


        memset(&ez_resDesc, 0, sizeof(ez_resDesc));
        ez_resDesc.resType = cudaResourceTypePitch2D;
        ez_resDesc.res.pitch2D.devPtr = d_EZ_row_major;
        ez_resDesc.res.pitch2D.width = Kepsilon;
        ez_resDesc.res.pitch2D.height = config::NPTS;
        ez_resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        ez_resDesc.res.pitch2D.pitchInBytes = pitch;

        memset(&ez_texDesc, 0, sizeof(ez_texDesc));
        cudaCreateTextureObject(&ez_tex, &ez_resDesc, &ez_texDesc, NULL);

        /*
        kernel_print<<<1, threads>>>(ex_tex);
        cudaDeviceSynchronize();
        std::cout << "=======================" << std::endl;
        kernel_print<<<1, threads>>>(ey_tex);
        cudaDeviceSynchronize();
        std::cout << "=======================" << std::endl;
        kernel_print<<<1, threads>>>(ez_tex);
        cudaDeviceSynchronize();
        std::cout << "=======================" << std::endl;
        */
    }


    if (use_texture && Kbeta > 0)
    {
        std::cout << "Kbeta  === " << Kbeta<< std::endl;
        memset(&tex_resDesc, 0, sizeof(tex_resDesc));
        tex_resDesc.resType = cudaResourceTypePitch2D;
        tex_resDesc.res.pitch2D.devPtr = d_TEX_row_major;
        tex_resDesc.res.pitch2D.width = Kbeta;
        tex_resDesc.res.pitch2D.height = config::NPTS;
        tex_resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        tex_resDesc.res.pitch2D.pitchInBytes = pitch3;

        memset(&tex_texDesc, 0, sizeof(tex_texDesc));
        cudaCreateTextureObject(&tex_tex, &tex_resDesc, &tex_texDesc, NULL);

        /*
        kernel_print<<<1, threads>>>(tex_tex);
        cudaDeviceSynchronize();
        */
    }
}





void Renderer::set_x0_short_y0_short(uint t, float *xp, float *yp, size_t array_size, bool pad)
{
    float *x0f = thrust::min_element(thrust::host, xp, xp+array_size);
    float *y0f = thrust::min_element(thrust::host, yp, yp+array_size);

    float *xff = thrust::max_element(thrust::host, xp, xp+array_size);
    float *yff = thrust::max_element(thrust::host, yp, yp+array_size);

    float xsize = *xff-*x0f;
    float ysize = *yff-*y0f;

    if (pad)
    {
        float sz = std::max<float>(xsize, ysize);
        //std::cout << sz << std::endl;
        x0_short[t] = (ushort) (std::max<float>(*x0f-sz/1.5, 0));
        y0_short[t] = (ushort) (std::max<float>(*y0f-sz/1.5, 0));
    }
    else
    {
        x0_short[t] = (ushort) (std::max<float>(*x0f-20, 0));
        y0_short[t] = (ushort) (std::max<float>(*y0f-20, 0));
    }


    // use the lines below for AFLW, otherwise comment them all

    /*************************
    if ((*x0f-DIMX/4.f) > 0.0f)
        x0_short[t] = *x0f-DIMX/4.f;
    else
        x0_short[t] = 0.0f;

    if ((*y0f-DIMY/4.f) > 0.0f)
        y0_short[t] = *y0f-DIMY/4.f;
    else
        y0_short[t] = 0.0f;

        */
}










void Renderer::render(uint t, Optimizer& o, OptimizationVariables& ov, const float *R, cublasHandle_t& handle,
                      ushort *N_unique_pixels, float *d_cropped_face, float *d_buffer_face, bool visualize, bool reset_texim)
{
    thrust::fill(thrust::device, d_Zmins, d_Zmins+DIMX*DIMY, std::numeric_limits<float>::max()); // or 999999.f if you prefer
    cudaMemset(o.gx_norm, 0, sizeof(float)*Nrender_estimated*6);

    if (reset_texim)
        cudaMemset(d_texIm, 0, sizeof(float)*NTOTAL_PIXELS);

    cudaMemset(o.gx, 0, sizeof(float)*Nrender_estimated*8);

    cudaMemset(o.grad_corrx, 0, sizeof(float));
    cudaMemset(o.grad_corry, 0, sizeof(float));

    /*
    reset_ushort_array<<<(NTOTAL_PIXELS + NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_M0, NTOTAL_PIXELS);
    reset_ushort_array<<<(NTOTAL_PIXELS + NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_cumM0, NTOTAL_PIXELS);
    */

    int Nksx = Nrender_estimated*19+NTOTAL_PIXELS*2;
    reset_ushort_array<<<(Nksx + NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_KSX, Nksx);

        get_pixels_to_render<<<(config::N_TRIANGLES*NTMP+NTHREADS-1)/NTHREADS, NTHREADS>>>(d_tl, d_xp, d_yp,  d_rend_flag, d_pixel_idx,
                                                                              d_alphas_redundant, d_betas_redundant, d_gammas_redundant,
                                                                              d_triangle_idx, d_Z, d_Ztmp, x0_short[t], y0_short[t],// cnt_per_pixel,
                                                                              d_Zmins, d_redundant_idx,
                                                                              config::N_TRIANGLES,
                                                                              config::Nredundant);

        keep_only_minZ<<<(config::Nredundant+NTHREADS-1)/NTHREADS, NTHREADS>>>(d_Zmins, d_Ztmp,  d_pixel_idx, d_redundant_idx, config::Nredundant);
        uint *new_end_idx = thrust::remove_if(thrust::device, d_redundant_idx, d_redundant_idx+config::Nredundant, is_zero<uint>());

        uint N1 = new_end_idx-d_redundant_idx;

    if (N1 > 2*Nrender_estimated) {
        if (config::PRINT_DEBUG)
            std::cout << "N1 is " << N1 << " which is probably too much ..." << std::endl;
        return;
    } else if (N1 < 1000) {
        if (config::PRINT_DEBUG)
            std::cout << "N1 is " << N1 << " which is probably too little ..." << std::endl;
        return;
    }


    *N_unique_pixels = N1;


    fill_diffuse_component_and_populate_texture_and_shape<<<(*N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(X0, Y0, Z0,
                                                                                                                X, Y, Z,
                                                                                                                d_redundant_idx, d_tl, d_triangle_idx,
                                                                                                                d_alphas_redundant, d_betas_redundant, d_gammas_redundant,
                                                                                                                d_Ztmp, o.d_Id_, o.d_tex, o.d_tex_torender, o.d_dI_dlambda, o.dI_dLintensity,
                                                                                                                o.vx, o.vy, o.vz,
                                                                                                                o.px, o.py, o.pz,
                                                                                                                o.inv_vz, o.inv_vz2, R,
                                                                                                                ov.taux, ov.lambda, ov.Lintensity,
                                                                                                                *N_unique_pixels, Nrender_estimated, false,
                                                                                                                config::N_TRIANGLES);

    fill_ks_and_M0<<<(*N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(d_pixel_idx, d_redundant_idx, o.d_ks_sorted, o.d_ks_unsorted, o.d_M0, *N_unique_pixels);
    fill_tex_im1<<<(*N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_ks_unsorted, o.d_tex_torender, d_texIm, *N_unique_pixels);

    if (visualize)
    //if (true)
    {
        imshow_opencv(d_texIm, "TIM");
        imshow_opencv(d_cropped_face+t*(DIMX*DIMY), "INPUT");

//        cv::waitKey(1);
        cv::waitKey(0);
    }

    convolutionRowsGPU( d_buffer_face, d_texIm, DIMX, DIMY );
    convolutionColumnsGPU(d_texIm, d_buffer_face, DIMX, DIMY );

    fill_grads<<<(*N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_ks_unsorted, d_texIm,
                                                                     d_cropped_face+t*(DIMX*DIMY),
                                                                     o.gx,  o.gy,  o.gx_norm,  o.gy_norm,  o.h,
                                                                     o.gxs, o.gys, o.gxs_norm, o.gys_norm, o.hs, *N_unique_pixels, Nrender_estimated);

    float h_log_barrier;

    cudaMemcpy(&h_log_barrier, o.ov_ptr->tau_logbarrier, sizeof(float), cudaMemcpyDeviceToHost);


    float plus_one_ = 1.0f;
    float zero_ = 0.0f;

    // Compute gradient correlation
    if (t == 0) { // <!-- careful here -->
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 2*Nrender_estimated, &h_log_barrier, o.gx_norm, 1, o.gxs_norm, 2*Nrender_estimated, &zero_, ov.grad_corr, 1);
    } else {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 2*Nrender_estimated, &h_log_barrier, o.gx_norm, 1, o.gxs_norm, 2*Nrender_estimated, &plus_one_, ov.grad_corr, 1);
    }
}






void Renderer::render_for_illumination_only(uint t, Optimizer& o,  OptimizationVariables& ov,  const float *R, cublasHandle_t& handle, ushort *N_unique_pixels, float *d_cropped_face, float *d_buffer_face, bool visualize)
{
    cudaMemset(d_texIm, 0, sizeof(float)*NTOTAL_PIXELS);
    cudaMemset(o.dG_dtheta, 0, ov.Ktotal*sizeof(float)); // was 19

    cudaMemset(o.grad_corrx, 0, sizeof(float));
    cudaMemset(o.grad_corry, 0, sizeof(float));

    cudaMemset(ov.grad_corr, 0, sizeof(float));

    cudaMemset(o.dgx_dlambda, 0, 3*Nrender_estimated*sizeof(float));
    cudaMemset(o.dgy_dlambda, 0, 3*Nrender_estimated*sizeof(float));
    cudaMemset(o.dgx_dLintensity, 0, Nrender_estimated*sizeof(float));
    cudaMemset(o.dgy_dLintensity, 0, Nrender_estimated*sizeof(float));
    cudaMemset(o.d_dI_dlambda, 0, 3*Nrender_estimated*sizeof(float));
    cudaMemset(o.dI_dLintensity, 0, Nrender_estimated*sizeof(float));

    fill_diffuse_component_and_populate_texture_and_shape<<<(*N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(X0, Y0, Z0,
                                                                                                                X, Y, Z,
                                                                                                                d_redundant_idx, d_tl, d_triangle_idx,
                                                                                                                d_alphas_redundant, d_betas_redundant, d_gammas_redundant,
                                                                                                                d_Ztmp, o.d_Id_, o.d_tex, o.d_tex_torender, o.d_dI_dlambda, o.dI_dLintensity,
                                                                                                                o.vx, o.vy, o.vz,
                                                                                                                o.px, o.py, o.pz,
                                                                                                                o.inv_vz, o.inv_vz2, R,
                                                                                                                ov.taux, ov.lambda, ov.Lintensity,
                                                                                                                *N_unique_pixels, Nrender_estimated, false,
                                                                                                                config::N_TRIANGLES);

    fill_tex_im1<<<(*N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_ks_unsorted, o.d_tex_torender, d_texIm, *N_unique_pixels);


    if (visualize) {
//     if (true) {
        imshow_opencv(d_texIm, "TIM");
        imshow_opencv(d_cropped_face+t*(DIMX*DIMY), "INPUT");

        cv::waitKey(0);
    }

    convolutionRowsGPU( d_buffer_face, d_texIm, DIMX, DIMY );
    convolutionColumnsGPU(d_texIm, d_buffer_face, DIMX, DIMY );

    fill_grads<<<(*N_unique_pixels+NTHREADS-1)/NTHREADS, NTHREADS>>>(o.d_ks_unsorted, d_texIm, d_cropped_face+t*(DIMX*DIMY),
                                                                     o.gx,  o.gy,  o.gx_norm,  o.gy_norm,  o.h,
                                                                     o.gxs, o.gys, o.gxs_norm, o.gys_norm, o.hs, *N_unique_pixels, Nrender_estimated);


    float plus_one_ = 1.0f;
    float zero_ = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 2*Nrender_estimated, &plus_one_,
                o.gx_norm, 1, o.gxs_norm, 2*Nrender_estimated, &zero_, ov.grad_corr, 1);

}














bool Renderer::compute_nonrigid_shape2(cublasHandle_t &handle, const OptimizationVariables& ov, const float* R, const Camera& cam)
{
    cudaMemcpy(X0, X0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Y0, Y0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Z0, Z0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);

    float alpha = 1.f;
    float beta  = 1.f;

    if (use_identity)
    {
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IX, config::NPTS, ov.alphas, 1, &beta, X0, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IY, config::NPTS, ov.alphas, 1, &beta, Y0, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IZ, config::NPTS, ov.alphas, 1, &beta, Z0, 1);
    }


    if (use_expression)
    {
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kepsilon, &alpha, d_EX, config::NPTS, ov.epsilons, 1, &beta, X0, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kepsilon, &alpha, d_EY, config::NPTS, ov.epsilons, 1, &beta, Y0, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kepsilon, &alpha, d_EZ, config::NPTS, ov.epsilons, 1, &beta, Z0, 1);
    }

    view_transform_3d_pts_and_render_2d<<<(config::NPTS+NTHREADS-1)/NTHREADS, NTHREADS>>>(X0, Y0, Z0,
                                                                                  R, ov.taux, ov.tauy, ov.tauz,
                                                                                  cam.phix, cam.phiy, cam.cx, cam.cy,
                                                                                  X, Y, Z,
                                                                                  d_xp, d_yp, config::NPTS);

    float xle, yle, xre, yre, xm, ym;
    /*** WARNING ***/
    cudaMemcpy(&xle, d_xp+config::LIS[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&yle, d_yp+config::LIS[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xre, d_xp+config::LIS[9], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&yre, d_yp+config::LIS[9], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xm, d_xp+config::LIS[40], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ym, d_yp+config::LIS[40], sizeof(float), cudaMemcpyDeviceToHost);

    float d_iod = fabs(xre-xle);
    float d_lem = fabs(yle-ym);
    float d_rem = fabs(yre-ym);

    bool is_face_reasonable = true;

    // if face is too large, we should probably ignore this step during optimization
    if (d_iod > config::REF_FACE_SIZE*2 || d_lem > config::REF_FACE_SIZE*2 || d_rem > config::REF_FACE_SIZE*2)
    {
        if (config::PRINT_DEBUG)
            std::cout << "d_iod: " << d_iod << "//" << "d_lem: " << d_lem << " d_rem: " << d_rem << std::endl;
        is_face_reasonable = false;
    }

    cudaMemcpy(d_Z, Z, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    return is_face_reasonable;
}




void Renderer::compute_nonrigid_shape_identityonly(cublasHandle_t &handle, const OptimizationVariables& ov)
{
    cudaMemcpy(X0, X0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Y0, Y0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Z0, Z0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);

    float alpha = 1.f;
    float beta  = 1.f;

    if (use_identity)
    {
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IX, config::NPTS, ov.alphas, 1, &beta, X0, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IY, config::NPTS, ov.alphas, 1, &beta, Y0, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IZ, config::NPTS, ov.alphas, 1, &beta, Z0, 1);
    }
}




void Renderer::compute_nonrigid_shape_expression_and_rotation(cublasHandle_t &handle, const OptimizationVariables& ov,
                                                            const float* R, float* Xcur, float*  Ycur, float *Zcur)
{
    cudaMemcpy(Xcur, X0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Ycur, Y0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Zcur, Z0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);

    float alpha = 1.f;
    float beta  = 1.f;

    if (use_identity)
    {
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IX, config::NPTS, ov.alphas, 1, &beta, Xcur, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IY, config::NPTS, ov.alphas, 1, &beta, Ycur, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IZ, config::NPTS, ov.alphas, 1, &beta, Zcur, 1);
    }

    if (use_expression)
    {
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kepsilon, &alpha, d_EX, config::NPTS, ov.epsilons, 1, &beta, X0, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kepsilon, &alpha, d_EY, config::NPTS, ov.epsilons, 1, &beta, Y0, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kepsilon, &alpha, d_EZ, config::NPTS, ov.epsilons, 1, &beta, Z0, 1);
    }


    rotate_3d_pts<<<(config::NPTS+NTHREADS-1)/NTHREADS, NTHREADS>>>(Xcur, Ycur, Zcur, R, config::NPTS);
}




void Renderer::compute_nonrigid_shape_identity_and_rotation(cublasHandle_t &handle, const OptimizationVariables& ov,
                                                            const float* R, float* Xcur, float*  Ycur, float *Zcur)
{
    cudaMemcpy(Xcur, X0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Ycur, Y0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    cudaMemcpy(Zcur, Z0_mean, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);

    float alpha = 1.f;
    float beta  = 1.f;

    if (use_identity)
    {
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IX, config::NPTS, ov.alphas, 1, &beta, Xcur, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IY, config::NPTS, ov.alphas, 1, &beta, Ycur, 1);
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kalpha, &alpha, d_IZ, config::NPTS, ov.alphas, 1, &beta, Zcur, 1);
    }

    rotate_3d_pts<<<(config::NPTS+NTHREADS-1)/NTHREADS, NTHREADS>>>(Xcur, Ycur, Zcur, R, config::NPTS);
}





void Renderer::compute_texture(cublasHandle_t &handle, const OptimizationVariables& ov, Optimizer &o)
{
    cudaMemcpy(o.d_tex, d_mu_tex, sizeof(float)*config::NPTS, cudaMemcpyDeviceToDevice);
    float alpha = 1.0f;
    float beta  = 1.0f;

    if (use_texture) {
        cublasSgemv(handle, CUBLAS_OP_N, config::NPTS, Kbeta, &alpha, d_TEX, config::NPTS, ov.betas, 1, &beta, o.d_tex, 1);
    }

}

void Renderer::print_obj(const std::string& obj_path)
{
    float *h_X, *h_Y, *h_Z;

    h_X = (float*)malloc( config::NPTS*sizeof(float) );
    h_Y = (float*)malloc( config::NPTS*sizeof(float) );
    h_Z = (float*)malloc( config::NPTS*sizeof(float) );

    cudaMemcpy(h_X, X, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y, Y, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z, Z, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);

    vector< vector<int> > tlv = read2DVectorFromFile<int>(config::TL_PATH, config::N_TRIANGLES, 3);

    std::ofstream ofs (obj_path, std::ofstream::out);

    for (uint i=0; i<config::NPTS; ++i) {
        ofs << "v " << h_X[i] << " " << h_Y[i] << " " << h_Z[i] << std::endl;
    }

    for (uint i=0; i<config::N_TRIANGLES; ++i) {
        ofs << "f " << tlv[i][0] << " " << tlv[i][1] << " " << tlv[i][2] << std::endl;
    }

    ofs.close();

    free(h_X);
    free(h_Y);
    free(h_Z);
}


void Renderer::print_obj_neutral(const std::string& obj_path)
{
    float *h_X, *h_Y, *h_Z;

    h_X = (float*)malloc( config::NPTS*sizeof(float) );
    h_Y = (float*)malloc( config::NPTS*sizeof(float) );
    h_Z = (float*)malloc( config::NPTS*sizeof(float) );

    cudaMemcpy(h_X, X0, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y, Y0, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z, Z0, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);

    vector< vector<int> > tlv = read2DVectorFromFile<int>(config::TL_PATH, config::N_TRIANGLES, 3);

    std::ofstream ofs (obj_path, std::ofstream::out);

    for (uint i=0; i<config::NPTS; ++i) {
        ofs << "v " << h_X[i] << " " << h_Y[i] << " " << h_Z[i] << std::endl;
    }

    for (uint i=0; i<config::N_TRIANGLES; ++i) {
        ofs << "f " << tlv[i][0] << " " << tlv[i][1] << " " << tlv[i][2] << std::endl;
    }

    ofs.close();

    free(h_X);
    free(h_Y);
    free(h_Z);
}


void Renderer::print_sparse_3Dpts(const std::string& obj_path)
{
    float *h_X, *h_Y, *h_Z;

    h_X = (float*)malloc( config::NPTS*sizeof(float) );
    h_Y = (float*)malloc( config::NPTS*sizeof(float) );
    h_Z = (float*)malloc( config::NPTS*sizeof(float) );

    cudaMemcpy(h_X, X, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y, Y, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z, Z, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);

    std::ofstream ofs (obj_path, std::ofstream::out);

    for (uint i=0; i<config::LIS.size(); ++i) {
        ofs <<  h_X[config::LIS[i]] << " " << h_Y[config::LIS[i]] << " " << h_Z[config::LIS[i]] << std::endl;
    }

    ofs.close();

    free(h_X);
    free(h_Y);
    free(h_Z);
}


#include <opencv2/dnn.hpp>

void Renderer::print_sparse_2Dpts(const std::string& obj_path, float _resize_coefl  )
{
    float *h_xp, *h_yp;

    h_xp = (float*)malloc( config::NPTS*sizeof(float) );
    h_yp = (float*)malloc( config::NPTS*sizeof(float) );

    cudaMemcpy(h_xp, d_xp, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_yp, d_yp, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    std::vector<float> xp, yp;

    for (uint i=0; i<config::LIS.size(); ++i) {
        xp.push_back(_resize_coefl*h_xp[config::LIS[i]] - config::PAD_SINGLE_IMAGE);
        yp.push_back(_resize_coefl*h_yp[config::LIS[i]] - config::PAD_SINGLE_IMAGE);
    }



    std::ofstream ofs (obj_path, std::ofstream::out);

    for (uint i=0; i<config::LIS.size(); ++i) {
        ofs <<  _resize_coefl*h_xp[config::LIS[i]] - config::PAD_SINGLE_IMAGE << " " << _resize_coefl*h_yp[config::LIS[i]] - config::PAD_SINGLE_IMAGE  << std::endl;
    }

    ofs.close();

    /*
    float xm = std::accumulate(xp.begin(), xp.end(), 0.0)/xp.size();
    float ym = std::accumulate(yp.begin(), yp.end(), 0.0)/yp.size();

    float xstd = std::sqrt(std::inner_product(xp.begin(), xp.end(), xp.begin(), 0.0) / xp.size() - xm * xm);
    float ystd = std::sqrt(std::inner_product(yp.begin(), yp.end(), yp.begin(), 0.0) / yp.size() - ym * ym);

    std::vector<float> pvec;
    for (uint i=0; i<NLANDMARKS_51; ++i) {
        pvec.push_back((xp[i]-xm)/xstd);
        pvec.push_back((yp[i]-ym)/ystd);
    }



    cv::Mat inp(1,102,CV_32FC1,&pvec.front());
    cv::dnn::Net cnet = cv::dnn::readNetFromONNX("./models/tf2onnx_onnx_model.onnx");
    cnet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    cnet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cnet.setInput(inp);
    cv::Mat out = cnet.forward().clone();

    inp += out;
    float *out_data = (float*) inp.data;






    std::ofstream ofs (obj_path, std::ofstream::out);

    for (uint i=0; i<NLANDMARKS_51; ++i) {
        uint lix = 2*i;
        uint liy = lix+1;
        ofs <<  (out_data[lix])*xstd+xm << " " <<  (out_data[liy])*ystd +ym<< " " << std::endl;
    }

    ofs.close();
*/

    free(h_xp);
    free(h_yp);
}


void Renderer::print_mat_txt(const std::string& mat_path)
{
    float *h_X, *h_Y, *h_Z;

    h_X = (float*)malloc( config::NPTS*sizeof(float) );
    h_Y = (float*)malloc( config::NPTS*sizeof(float) );
    h_Z = (float*)malloc( config::NPTS*sizeof(float) );

    cudaMemcpy(h_X, X, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y, Y, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z, Z, sizeof(float)*config::NPTS, cudaMemcpyDeviceToHost);


    std::ofstream ofs (mat_path, std::ofstream::out);

    for (uint i=0; i<config::NPTS; ++i) {
        ofs << h_X[i] << " " << h_Y[i] << " " << h_Z[i] << std::endl;
    }

    ofs.close();

    free(h_X);
    free(h_Y);
    free(h_Z);
}


Renderer::~Renderer()
{
    free(x0_short);
    free(y0_short);


    if (use_expression)
    {
        HANDLE_ERROR( cudaFree( d_EX_row_major ));
        HANDLE_ERROR( cudaFree( d_EY_row_major ));
        HANDLE_ERROR( cudaFree( d_EZ_row_major ));

        HANDLE_ERROR( cudaFree( d_EX ));
        HANDLE_ERROR( cudaFree( d_EY ));
        HANDLE_ERROR( cudaFree( d_EZ ));

        HANDLE_ERROR( cudaFree( d_REX ));
        HANDLE_ERROR( cudaFree( d_REY ));
        HANDLE_ERROR( cudaFree( d_REZ ));
    }

    HANDLE_ERROR( cudaFree( d_mu_tex ));


    if (use_identity)
    {
        HANDLE_ERROR( cudaFree( d_IX_row_major ));
        HANDLE_ERROR( cudaFree( d_IY_row_major ));
        HANDLE_ERROR( cudaFree( d_IZ_row_major ));

        HANDLE_ERROR( cudaFree( d_RIX ));
        HANDLE_ERROR( cudaFree( d_RIY ));
        HANDLE_ERROR( cudaFree( d_RIZ ));

        HANDLE_ERROR( cudaFree( d_IX ));
        HANDLE_ERROR( cudaFree( d_IY ));
        HANDLE_ERROR( cudaFree( d_IZ ));
    }



    if (use_texture)
    {
        HANDLE_ERROR(cudaFree(d_TEX_row_major));
        HANDLE_ERROR( cudaFree( d_TEX ));
        HANDLE_ERROR( cudaFree( d_RTEX ));
    }

    HANDLE_ERROR( cudaFree(d_pixel_idx) );
    HANDLE_ERROR( cudaFree(d_triangle_idx) );

    HANDLE_ERROR( cudaFree(d_alphas_redundant) );
    HANDLE_ERROR( cudaFree(d_betas_redundant) );
    HANDLE_ERROR( cudaFree(d_gammas_redundant) );

    HANDLE_ERROR( cudaFree(d_xp) );
    HANDLE_ERROR( cudaFree(d_yp) );

    HANDLE_ERROR( cudaFree(d_rend_flag) );
    HANDLE_ERROR( cudaFree(d_rend_flag_tmp) );

    HANDLE_ERROR( cudaFree(d_tl) );
    HANDLE_ERROR( cudaFree(d_redundant_idx) );

    HANDLE_ERROR( cudaFree(ALL_VARS) );
    HANDLE_ERROR( cudaFree(d_grad) );
    HANDLE_ERROR( cudaFree(d_texIm) );


    HANDLE_ERROR( cudaFree( d_Ztmp ));
    HANDLE_ERROR( cudaFree( d_Z ));
    HANDLE_ERROR( cudaFree( d_Zmins ));


}


