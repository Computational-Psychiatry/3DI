/*
 * constants.h
 *
 *  Created on: Aug 8, 2020
 *      Author: root
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#define N_TRIANGLES 46703
#define NPTS 23660
//#define NTMP 13

//#define NTMP 9
//#define NTMP 12
//#define NTMP 3 // <----
//#define NTMP 5
#define NTMP 4
//#define NTMP 7 // this is too large but we'll use it for dyadic sync analyses. Otherwise we'll probably go back to NTMP=3
//#define NTMP 2
//#define NTMP 1
#define NID_COEFS 199
#define NTEX_COEFS 199
#define K_ALPHA NID_COEFS
#define K_BETA NTEX_COEFS
#define K_ALPHA_L 60
#define K_BETA_L 0
#define USE_TEXTURE 1
#define USE_IDENTITY 1
#define USE_EXPRESSION 1
#define NLANDMARKS_51 51
#define NLANDMARKS_68 68
#define LEYE_ID 19
#define REYE_ID 28
//#define RESIZE_COEF 0.6f
//#define RESIZE_COEF 0.8f
//#define RESIZE_COEF 0.45f
#define ANGLE_MIN -30.0
#define ANGLE_MAX 30.0
#define ANGLE_STEP 10
#define N_ANGLE_COMBINATIONS 36

#define N_REDUNDANT NTMP*N_TRIANGLES

const uint Nredundant = N_TRIANGLES*NTMP;
//const uint Nrender_estimated = 25000;
//const uint Nrender_estimated = 16000;
//const uint Nrender_estimated = 26000;
const uint Nrender_estimated = 32000; // this is too large but we'll use it for dyadic sync analyses. Otherwise we'll probably go back to something like 16000
//const uint Nrender_estimated = 10000;
//const uint Nrender_estimated = 8500;
//const uint Nrender_estimated = 6000;

#define DIMX 256
#define DIMY 256
#define NTHREADS 1024

#define RAD2DEG(angleRadians) ((angleRadians) * 180.0f / M_PI)
#define DEG2RAD(angleDegrees) ((angleDegrees) * M_PI) / 180.0

#define NTOTAL_PIXELS DIMX*DIMY

#define HANDLE_ERROR
#define PRINT_EM 0
#define WRITE_VARS_TO_DISK

#define WRITE_SPARSE_LANDMARKS

//#define MEASURE_TIME 0


#endif /* CONSTANTS_H_ */
