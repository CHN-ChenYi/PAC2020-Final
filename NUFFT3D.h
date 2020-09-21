#ifndef NUFFT3D_H
#define NUFFT3D_H

#include <fftw3.h>
#include <malloc.h>

#include <complex>

#include "MathOps.h"

using namespace std;

// can only exist one object at a time
class NUFFT3D {
 private:
  complex<float>* f;
  int N;
  int OF;
  int N2;
  float* wx;
  float* wy;
  float* wz;
  int P;
  int prechopX;
  int prechopY;
  int prechopZ;
  int postchopX;
  int postchopY;
  int postchopZ;
  int offsetX;
  int offsetY;
  int offsetZ;
  int W;
  int L;
  fftwf_plan fwdPlan;
  fftwf_plan adjPlan;
  float* LUT;
  void buildLUT();
  void getScalingFunction();
  float* q;

  static const int N_X, N_Y, N_Z;
  static int *task_count;
  struct cmp {
    bool operator()(const int left, const int right) {
      return task_count[left] > task_count[right];
    }
  };
  void ConvolutionAdj(complex<float>*);

 public:
  NUFFT3D(int, int, float*, float*, float*, int, int, int, int, int, int, int,
          int, int, int, int, int);
  ~NUFFT3D();
  static void init(int);
  void fwd(complex<float>*, complex<float>*);
  void adj(complex<float>*, complex<float>*);
};

#endif
