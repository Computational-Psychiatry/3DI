#include "GLfuncs.h"
#include "constants.h"
#include <math.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

// Initializes information for drawing within OpenGL.
void initGLwindow()
{

    int _argc = 0;
    char **_argv = NULL;
    /* Initialize OpenGL stuff */
    glutInit(&_argc, _argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(50, 100);    // Set up display window.
    glutInitWindowSize(1080, 1080);
    glutCreateWindow("3D Morphable Face");



    GLfloat sun_direction[] = { 5.0, 0.0, 3.0, 1.0 };
    GLfloat sun_intensity[] = { 0.7, 0.7, 0.7, 1.0 };
    GLfloat ambient_intensity[] = { 0.3, 0.3, 0.3, 1.0 };

    glClearColor(0.0, 1.0, 0.0, 0.0);   // Set window color to white.
    //    computeLocation();
    glMatrixMode(GL_PROJECTION);        // Set projection parameters.
    glLoadIdentity();

    gluPerspective(15.0, 1.0, 0.001, 250.0);
    glMatrixMode (GL_MODELVIEW);

    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    glEnable(GL_DEPTH_TEST);            // Draw only closest surfaces
    glEnable(GL_LIGHTING);              // Set up ambient light.
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_intensity);

    glEnable(GL_LIGHT0);                // Set up sunlight.
    glLightfv(GL_LIGHT0, GL_POSITION, sun_direction);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, sun_intensity);

    glEnable(GL_COLOR_MATERIAL);        // Configure glColor().
    glEnable( GL_LINE_SMOOTH );

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}




cv::Mat drawFace_fromptr(const std::vector<std::vector<int > >& tl_vector, float *X0, float *Y0, float *Z0)
{

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear window.
    glColor3f(1.0, 1.0, 1.0);
    glShadeModel(GL_SMOOTH);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    std::vector<float> normalsx(NPTS, 0.0f);
    std::vector<float> normalsy(NPTS, 0.0f);
    std::vector<float> normalsz(NPTS, 0.0f);

    //! First compute the normals per point
    for (uint i=0; i<N_TRIANGLES; ++i)
    {
        uint a = tl_vector[i][0]-1;
        uint b = tl_vector[i][1]-1;
        uint c = tl_vector[i][2]-1;

        float xa = X0[a];
        float ya = Y0[a];
        float za = Z0[a];

        float xb = X0[b];
        float yb = Y0[b];
        float zb = Z0[b];

        float xc = X0[c];
        float yc = Y0[c];
        float zc = Z0[c];

        float n_vx = xb-xa;
        float n_vy = yb-ya;
        float n_vz = zb-za;

        float n_wx = xc-xa;
        float n_wy = yc-ya;
        float n_wz = zc-za;

        float nrefx = n_vy*n_wz-n_vz*n_wy;
        float nrefy = n_vz*n_wx-n_vx*n_wz;
        float nrefz = n_vx*n_wy-n_vy*n_wx;

        float normn = sqrtf(nrefx*nrefx+nrefy*nrefy+nrefz*nrefz);

        float nref_norm[3] = {nrefx/normn, nrefy/normn, nrefz/normn};

        normalsx[a] += nref_norm[0];
        normalsx[b] += nref_norm[0];
        normalsx[c] += nref_norm[0];

        normalsy[a] += nref_norm[1];
        normalsy[b] += nref_norm[1];
        normalsy[c] += nref_norm[1];

        normalsz[a] += nref_norm[2];
        normalsz[b] += nref_norm[2];
        normalsz[c] += nref_norm[2];
    }

    for (uint i=0; i<NPTS; ++i)
    {
        float normn = sqrtf(normalsx[i]*normalsx[i]+normalsy[i]*normalsy[i]+normalsz[i]*normalsz[i]);

        normalsx[i] /= normn;
        normalsy[i] /= normn;
        normalsz[i] /= normn;
    }



    //! And then draw the points -- each triangle of the mesh
    for (uint i=0; i<N_TRIANGLES; i++) {
        uint a = tl_vector[i][0]-1;
        uint b = tl_vector[i][1]-1;
        uint c = tl_vector[i][2]-1;

        float coef = 1;
        float tz = 24.0f;

        float xa = coef*X0[a];
        float ya = -coef*Y0[a];
        float za = -coef*(Z0[a]+tz);

        float xb = coef*X0[b];
        float yb = -coef*Y0[b];
        float zb = -coef*(Z0[b]+tz);

        float xc = coef*X0[c];
        float yc = -coef*Y0[c];
        float zc = -coef*(Z0[c]+tz);

        glBegin(GL_TRIANGLES);

        glNormal3f(normalsx[a], normalsy[a], normalsz[a]);
        glVertex3f(xa, ya, za);

        glNormal3f(normalsx[b], normalsy[b], normalsz[b]);
        glVertex3f(xb, yb, zb);

        glNormal3f(normalsx[c], normalsy[c], normalsz[c]);
        glVertex3f(xc, yc, zc);

        glEnd();
    }

    cv::Mat im = opengl2opencv();
    /*
    cv::imshow("out", im);

    cv::waitKey(0);

    //    drawSphere(1.0, 60, 60); // glutSolidSphere(1.0, 10, 10);
    */
    glutSwapBuffers();
    return im;
}










cv::Mat drawFace(const std::vector<std::vector<int > >& tl_vector, const std::vector<float>& X0, const std::vector<float>& Y0, const std::vector<float>& Z0)
{

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear window.
    glColor3f(1.0, 1.0, 1.0);
    glShadeModel(GL_SMOOTH);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    std::vector<float> normalsx(NPTS, 0.0f);
    std::vector<float> normalsy(NPTS, 0.0f);
    std::vector<float> normalsz(NPTS, 0.0f);

    //! First compute the normals per point
    for (uint i=0; i<N_TRIANGLES; ++i)
    {
        uint a = tl_vector[i][0]-1;
        uint b = tl_vector[i][1]-1;
        uint c = tl_vector[i][2]-1;

        float xa = X0[a];
        float ya = Y0[a];
        float za = Z0[a];

        float xb = X0[b];
        float yb = Y0[b];
        float zb = Z0[b];

        float xc = X0[c];
        float yc = Y0[c];
        float zc = Z0[c];

        float n_vx = xb-xa;
        float n_vy = yb-ya;
        float n_vz = zb-za;

        float n_wx = xc-xa;
        float n_wy = yc-ya;
        float n_wz = zc-za;

        float nrefx = n_vy*n_wz-n_vz*n_wy;
        float nrefy = n_vz*n_wx-n_vx*n_wz;
        float nrefz = n_vx*n_wy-n_vy*n_wx;

        float normn = sqrtf(nrefx*nrefx+nrefy*nrefy+nrefz*nrefz);

        float nref_norm[3] = {nrefx/normn, nrefy/normn, nrefz/normn};

        normalsx[a] += nref_norm[0];
        normalsx[b] += nref_norm[0];
        normalsx[c] += nref_norm[0];

        normalsy[a] += nref_norm[1];
        normalsy[b] += nref_norm[1];
        normalsy[c] += nref_norm[1];

        normalsz[a] += nref_norm[2];
        normalsz[b] += nref_norm[2];
        normalsz[c] += nref_norm[2];
    }

    for (uint i=0; i<NPTS; ++i)
    {
        float normn = sqrtf(normalsx[i]*normalsx[i]+normalsy[i]*normalsy[i]+normalsz[i]*normalsz[i]);

        normalsx[i] /= normn;
        normalsy[i] /= normn;
        normalsz[i] /= normn;
    }



    //! And then draw the points -- each triangle of the mesh
    for (uint i=0; i<N_TRIANGLES; i++) {
        uint a = tl_vector[i][0]-1;
        uint b = tl_vector[i][1]-1;
        uint c = tl_vector[i][2]-1;

        float coef = 1;
        float tz = 24.0f;

        float xa = coef*X0[a];
        float ya = -coef*Y0[a];
        float za = -coef*(Z0[a]+tz);

        float xb = coef*X0[b];
        float yb = -coef*Y0[b];
        float zb = -coef*(Z0[b]+tz);

        float xc = coef*X0[c];
        float yc = -coef*Y0[c];
        float zc = -coef*(Z0[c]+tz);

        glBegin(GL_TRIANGLES);

        glNormal3f(normalsx[a], normalsy[a], normalsz[a]);
        glVertex3f(xa, ya, za);

        glNormal3f(normalsx[b], normalsy[b], normalsz[b]);
        glVertex3f(xb, yb, zb);

        glNormal3f(normalsx[c], normalsy[c], normalsz[c]);
        glVertex3f(xc, yc, zc);

        glEnd();
    }

    cv::Mat im = opengl2opencv();
    /*
    cv::imshow("out", im);

    cv::waitKey(0);

    //    drawSphere(1.0, 60, 60); // glutSolidSphere(1.0, 10, 10);
    */
    glutSwapBuffers();
    return im;
}



cv::Mat opengl2opencv()
{
    int w = 1080;
    int h = 1080;

    cv::Mat in(w,h,CV_8UC4, cv::Scalar::all(0));
    cv::Mat out(h,w,CV_8UC4, cv::Scalar::all(0));

    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, in.data);

    // following operations break image >>
    cv::cvtColor(in, out, cv::COLOR_RGBA2BGRA);
    cv::flip(out, in, 0);
    cv::cvtColor(in, out, cv::COLOR_BGRA2RGBA);
    // << prev operations break image

    GLuint tex2 = 0;
    glBindTexture(GL_TEXTURE_2D, (GLuint) tex2);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 w,
                 h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 out.ptr());

    glBindTexture(GL_TEXTURE_2D, 0);

    return out;
}






