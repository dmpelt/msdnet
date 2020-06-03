//-----------------------------------------------------------------------
//Copyright 2019 Centrum Wiskunde & Informatica, Amsterdam
//
//Author: Daniel M. Pelt
//Contact: D.M.Pelt@cwi.nl
//Website: http://dmpelt.github.io/msdnet/
//License: MIT
//
//This file is part of MSDNet, a Python implementation of the
//Mixed-Scale Dense Convolutional Neural Network.
//-----------------------------------------------------------------------

#include <math.h>

#ifdef _MSC_VER

#define DECLDIR __declspec(dllexport)

DECLDIR void set_threads(const unsigned int nthrd){}

#else

#define DECLDIR

#include <omp.h>
// OpenMP set number of threads
DECLDIR void set_threads(const unsigned int nthrd){
    omp_set_num_threads(nthrd);
}

#endif

// Flattened array operations
DECLDIR void relu(float * const data, const unsigned long n){
    long i;
    #pragma omp parallel for private(i)
    for(i=0; i<n; i++){
        if(data[i]<0){
            data[i]=0;
        }
    }
}

DECLDIR void leakyrelu(float * const data, const unsigned long n, const float w){
    long i;
    #pragma omp parallel for private(i)
    for(i=0; i<n; i++){
        if(data[i]<0){
            data[i]*=w;
        }
    }
}

DECLDIR void relu2(const float * const inp, float * const out, const unsigned long n){
    long i;
    #pragma omp parallel for private(i)
    for(i=0; i<n; i++){
        if(inp[i]<=0){
            out[i]=0;
        }
    }
}

DECLDIR void leakyrelu2(const float * const inp, float * const out, const unsigned long n, const float w){
    long i;
    #pragma omp parallel for private(i)
    for(i=0; i<n; i++){
        if(inp[i]<=0){
            out[i]*=w;
        }
    }
}


DECLDIR void combine(const float * const inp, float * const out, const unsigned long n, const float w){
    long i;
    #pragma omp parallel for private(i)
    for(i=0; i<n; i++){
        out[i]+=w*inp[i];
    }
}

DECLDIR float sum(const float * const inp, const unsigned long n){
    long double sum=0;
    long i;
    #pragma omp parallel for reduction(+:sum) private(i)
    for(i=0; i<n; i++){
        sum+=inp[i];
    }
    return (float)sum;
}

DECLDIR float std(const float * const inp, const float mn, const unsigned long n){
    long double sum=0;
    long i;
    #pragma omp parallel for reduction(+:sum) private(i)
    for(i=0; i<n; i++){
        sum+=(inp[i]-mn)*(inp[i]-mn);
    }
    return (float)sqrt(sum/n);
}

DECLDIR float multsum(const float * const a, const float * const b, const unsigned long n){
    long double sum=0;
    long i;
    #pragma omp parallel for reduction(+:sum) private(i)
    for(i=0; i<n; i++){
        sum+=a[i]*b[i];
    }
    return (float)sum;
}

DECLDIR void softmax(float * const im, const unsigned long n, const unsigned int nim){
    #pragma omp parallel
    {
        float mx,sm;
        long i;
        unsigned int j;
        #pragma omp for
        for(i=0; i<n;i++){
            mx=im[i];
            for(j=1; j<nim; j++){
                if(im[j*n+i]>mx){
                    mx = im[j*n+i];
                }
            }
            sm=0;
            for(j=0; j<nim; j++){
                im[j*n+i] = expf(im[j*n+i]-mx);
                sm += im[j*n+i];
            }
            for(j=0; j<nim; j++){
                im[j*n+i] /= sm;
            }
        }
    }
}

DECLDIR void softmaxderiv(float * const out, const float * const err, const float * const act, const unsigned long n, const unsigned int nim){
    #pragma omp parallel
    {
        float tmp;
        long i;
        unsigned int j, k;
        #pragma omp for
        for(i=0; i<n; i++){
            for(j=0; j<nim; j++){
                tmp = err[j*n+i]*act[j*n+i]*(1-act[j*n+i]);
                for(k=0; k<j; k++) tmp -= act[j*n+i] * act[k*n+i] * err[k*n+i];
                for(k=j+1; k<nim; k++) tmp -= act[j*n+i] * act[k*n+i] * err[k*n+i];
                out[j*n+i] = tmp;
            }
        }
    }
}



DECLDIR long double squaresum(const float * const a, const unsigned long n){
    long double sum=0;
    long i;
    #pragma omp parallel for reduction(+:sum) private(i)
    for(i=0; i<n; i++){
        sum+=a[i]*a[i];
    }
    return sum;
}


// 2D operations
DECLDIR void conv2d(const float * const inp, float * const out, const float * const f, const unsigned int nx, const unsigned int ny, const int * const shx, const int * const shy){
    #pragma omp parallel
    {
        long double tmp;
        const float *rl, *rc, *rr;
        int i;
        #pragma omp for
        for(i=0; i<nx; i++){
            rl = inp + (shx[2*i])*ny;
            rc = inp + i*ny;
            rr = inp + (shx[2*i+1])*ny;
            for(unsigned int j=0; j<ny; j++){
                tmp=0;
                tmp += rl[shy[2*j]]*f[0];
                tmp += rl[j]*f[1];
                tmp += rl[shy[2*j+1]]*f[2];
                tmp += rc[shy[2*j]]*f[3];
                tmp += rc[j]*f[4];
                tmp += rc[shy[2*j+1]]*f[5];
                tmp += rr[shy[2*j]]*f[6];
                tmp += rr[j]*f[7];
                tmp += rr[shy[2*j+1]]*f[8];
                out[i*ny+j] += tmp;
            }
        }
    }
}


DECLDIR float gradientmap2d(const float * const inp, const float * const delta, const unsigned int nx, const unsigned int ny, const int * const shx, const int * const shy){
    long double sum=0;
    #pragma omp parallel
    {
        const float *ix, *dx;
        int i;
        #pragma omp for reduction(+:sum)
        for(i=0; i<nx; i++){
            ix = inp + shx[i]*ny;
            dx = delta + i*ny;
            for(unsigned int j=0; j<ny; j++){
                sum += ix[shy[j]]*dx[j];
            }
        }
    }
    return (float)sum;
}

