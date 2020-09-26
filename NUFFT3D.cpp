#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

#include "common.h"

using std::atomic;
using std::thread;
using std::unique_lock;
using std::vector;

TDEF(fftw)
TDEF(nufft)

// 64
const int NUFFT3D::N_X = 16;
const int NUFFT3D::N_Y = 16;
const int NUFFT3D::N_Z = 16;
int* const NUFFT3D::task_count = new int[N_X * N_Y * N_Z];
const int NUFFT3D::GrayCode[8] = {0, 1, 3, 2, 6, 7, 5, 4};
const int NUFFT3D::GrayCodeOrder[8] = {0, 1, 3, 2, 7, 6, 4, 5};

/* Constructor */
NUFFT3D::NUFFT3D(int N, int OF, float* wx, float* wy, float* wz, int P,
                 int prechopX, int prechopY, int prechopZ, int postchopX,
                 int postchopY, int postchopZ, int offsetX, int offsetY,
                 int offsetZ, int W, int L) {
  // Assignments
  this->N = N;
  this->OF = OF;
  N2 = N * OF;
  this->wx = wx;
  this->wy = wy;
  this->wz = wz;
  this->P = P;
  this->prechopX = prechopX;
  this->prechopY = prechopY;
  this->prechopZ = prechopZ;
  this->postchopX = postchopX;
  this->postchopY = postchopY;
  this->postchopZ = postchopZ;
  this->offsetX = offsetX;
  this->offsetY = offsetY;
  this->offsetZ = offsetZ;
  this->W = W;
  this->L = L;

  int DIMS[3] = {N2, N2, N2};
  f = (complex<double>*)memalign(16, N2 * N2 * N2 * sizeof(complex<double>));
  fwdPlan = fftw_plan_dft(3, DIMS, reinterpret_cast<fftw_complex*>(f),
                          reinterpret_cast<fftw_complex*>(f), FFTW_FORWARD,
                          FFTW_ESTIMATE);
  adjPlan = fftw_plan_dft(3, DIMS, reinterpret_cast<fftw_complex*>(f),
                          reinterpret_cast<fftw_complex*>(f), FFTW_BACKWARD,
                          FFTW_ESTIMATE);
  buildLUT();
  getScalingFunction();
}

/* Destructor */
NUFFT3D::~NUFFT3D() {
  fftw_destroy_plan(fwdPlan);
  fftw_destroy_plan(adjPlan);
  free(f);
  f = NULL;
}

/* Initialize multithreaded FFTW (p threads) */
void NUFFT3D::init(int nThreads) {
  fftw_init_threads();
  fftw_plan_with_nthreads(nThreads);
}

/* Forward NUFFT transform */
void NUFFT3D::fwd(complex<float>* u, complex<float>* raw) {
  // Apodization and zero-padding
  int startX = N * (OF - 1) / 2;
  int startY = N * (OF - 1) / 2;
  int startZ = N * (OF - 1) / 2;
  TSTART(nufft);
  TSTART(fftw);
  for (int i = 0; i < N2 * N2 * N2; i++) {
    f[i] = 0;
  }
  TEND(fftw);
  TPRINT(fftw, "  Init_F FWD");
  TSTART(fftw);
  for (int x = 0; x < N; x++) {
    for (int y = 0; y < N; y++) {
      for (int z = 0; z < N; z++) {
        f[(x + startX + offsetX) * N2 * N2 + (y + startY + offsetY) * N2 +
          (z + startZ + offsetZ)] =
            u[x * N * N + y * N + z] / q[x * N * N + y * N + z];
      }
    }
  }
  TEND(fftw);
  TPRINT(fftw, "  Roundoff_Corr FWD");

  // (Oversampled) FFT
  TSTART(fftw);
  chop3D(f, N2, N2, N2, prechopX, prechopY, prechopZ);
  TEND(fftw);
  TPRINT(fftw, "  PreChop FWD")
  TSTART(fftw);
  fftw_execute(fwdPlan);
  TEND(fftw);
  TPRINT(fftw, "  FFTW FWD");
  TSTART(fftw)
  chop3D(f, N2, N2, N2, postchopX, postchopY, postchopZ);
  TEND(fftw);
  TPRINT(fftw, "  PostChop FWD");

  // Pull from grid
  TSTART(fftw);
#pragma omp parallel for schedule(guided)
  for (int p = 0; p < P; p++) {
    int kx2[2 * W + 1];
    int ky2[2 * W + 1];
    int kz2[2 * W + 1];
    float winX[2 * W + 1];
    float winY[2 * W + 1];
    float winZ[2 * W + 1];

    // Form x interpolation kernel
    float kx = N2 * (wx[p] + 0.5);
    int x1 = (int)ceil(kx - W);
    int x2 = (int)floor(kx + W);
    int lx = x2 - x1 + 1;
    for (int nx = 0; nx < lx; nx++) {
      kx2[nx] = mod(nx + x1, N2);
      winX[nx] = LUT[(int)round(((L - 1) / W) * abs(nx + x1 - kx))];
    }

    // Form y interpolation kernel
    float ky = N2 * (wy[p] + 0.5);
    int y1 = (int)ceil(ky - W);
    int y2 = (int)floor(ky + W);
    int ly = y2 - y1 + 1;
    for (int ny = 0; ny < ly; ny++) {
      ky2[ny] = mod(ny + y1, N2);
      winY[ny] = LUT[(int)round(((L - 1) / W) * abs(ny + y1 - ky))];
    }

    // Form z interpolation kernel
    float kz = N2 * (wz[p] + 0.5);
    int z1 = (int)ceil(kz - W);
    int z2 = (int)floor(kz + W);
    int lz = z2 - z1 + 1;
    for (int nz = 0; nz < lz; nz++) {
      kz2[nz] = mod(nz + z1, N2);
      winZ[nz] = LUT[(int)round(((L - 1) / W) * abs(nz + z1 - kz))];
    }

    // Interpolation
    raw[p] = 0;
    for (int nx = 0; nx < lx; nx++) {
      for (int ny = 0; ny < ly; ny++) {
        for (int nz = 0; nz < lz; nz++) {
          raw[p] += f[kx2[nx] * N2 * N2 + ky2[ny] * N2 + kz2[nz]] * winX[nx] *
                    winY[ny] * winZ[nz];
        }
      }
    }
  }
  TEND(fftw);
  TPRINT(fftw, "  Convolution FWD");
  TEND(nufft);
  TPRINT(nufft, "NUFFT FWD");
}

void NUFFT3D::ConvolutionAdjCore(complex<float>* raw, vector<int>& task) {
  for (int& p : task) {
    int kx2[2 * W + 1];
    int ky2[2 * W + 1];
    int kz2[2 * W + 1];
    float winX[2 * W + 1];
    float winY[2 * W + 1];
    float winZ[2 * W + 1];

    // Form x interpolation kernel
    float kx = N2 * (wx[p] + 0.5);
    int x1 = (int)ceil(kx - W);
    int x2 = (int)floor(kx + W);
    int lx = x2 - x1 + 1;
#pragma omp simd
    for (int nx = 0; nx < lx; nx++) {
      kx2[nx] = mod(nx + x1, N2);
      winX[nx] = LUT[(int)round(((L - 1) / W) * abs(nx + x1 - kx))];
    }

    // Form y interpolation kernel
    float ky = N2 * (wy[p] + 0.5);
    int y1 = (int)ceil(ky - W);
    int y2 = (int)floor(ky + W);
    int ly = y2 - y1 + 1;
#pragma omp simd
    for (int ny = 0; ny < ly; ny++) {
      ky2[ny] = mod(ny + y1, N2);
      winY[ny] = LUT[(int)round(((L - 1) / W) * abs(ny + y1 - ky))];
    }

    // Form z interpolation kernel
    float kz = N2 * (wz[p] + 0.5);
    int z1 = (int)ceil(kz - W);
    int z2 = (int)floor(kz + W);
    int lz = z2 - z1 + 1;
#pragma omp simd
    for (int nz = 0; nz < lz; nz++) {
      kz2[nz] = mod(nz + z1, N2);
      winZ[nz] = LUT[(int)round(((L - 1) / W) * abs(nz + z1 - kz))];
    }

    // Interpolation
    for (int nx = 0; nx < lx; nx++) {
      for (int ny = 0; ny < ly; ny++) {
#pragma omp simd
        for (int nz = 0; nz < lz; nz++) {
          f[kx2[nx] * N2 * N2 + ky2[ny] * N2 + kz2[nz]] +=
              raw[p] * winX[nx] * winY[ny] * winZ[nz];
        }
      }
    }
  }
}

extern int numThreads;

#define Probe(dimension, delta, dimension_max)                                 \
  {                                                                            \
    dimension += delta;                                                        \
    const int probe_id = x * N_Y * N_Z + y * N_Z + z;                          \
    dimension += delta;                                                        \
    if (!in_queue[probe_id] && (dimension < 0 || dimension >= dimension_max || \
                                vis[x * N_Y * N_Z + y * N_Z + z])) {           \
      task_list.push(probe_id);                                                \
      in_queue[probe_id] = true;                                               \
      cv_task.notify_one();                                                    \
    }                                                                          \
  }

inline void find_id(const int& avg, const int& ratio, const int& P, int id[],
                    float w[]) {
  float min_w = w[0], max_w = w[0];
  int width_of_counter, sum;
#pragma omp parallel for schedule(static) reduction(max : max_w)
  for (int i = 1; i < P; i++) max_w = max_w > w[i] ? max_w : w[i];
#pragma omp parallel for schedule(static) reduction(min : min_w)
  for (int i = 1; i < P; i++) min_w = min_w < w[i] ? min_w : w[i];
  width_of_counter = (max_w - min_w) * ratio + 1;
  int* counter = new int[width_of_counter];
#pragma omp parallel for schedule(static)
  for (int i = 0; i < width_of_counter; i++) counter[i] = 0;
#pragma omp parallel for schedule(static)
  for (int p = 0; p < P; p++) id[p] = (w[p] - min_w) * ratio;
  for (int p = 0; p < P; p++) counter[id[p]]++;
  sum = counter[0];
  counter[0] = 0;
  for (int i = 0, sum = 0; i < width_of_counter; i++) {
    if (sum > avg) {
      sum = counter[i];
      counter[i] = counter[i - 1] + 1;
    } else {
      sum += counter[i];
      counter[i] = counter[i - 1];
    }
  }
#pragma omp parallel for schedule(guided)
  for (int p = 0; p < P; p++) id[p] = counter[id[p]];
  delete[] counter;
}

void analyze(int count[], int n, int m) {
  const double avg = m / n;
  double dev = 0;
  for (int i = 0; i < n; i++) dev += std::pow(count[i] - avg, 2);
  dev = std::sqrt(dev / n);
  printf("divide %d to %d with sd %lf (avg %lf)\n", m, n, dev, avg);
}

void NUFFT3D::ConvolutionAdj(complex<float>* raw) {
  // TDEF(init);
  // printf("%d: ", P);
  // TSTART(init);

  // find the task for each example
  int *id_x = new int[P], *id_y = new int[P], *id_z = new int[P];
  const int ratio = N2 / W * 1;
  find_id(P / N_X, ratio, P, id_x, wx);
  find_id(P / N_Y, ratio, P, id_y, wy);
  find_id(P / N_Z, ratio, P, id_z, wz);
  vector<int>* task = new vector<int>[N_X * N_Y * N_Z];
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N_X * N_Y * N_Z; i++) task_count[i] = 0;
  for (int p = 0; p < P; p++) {
    const int id = id_x[p] * N_Y * N_Z + id_y[p] * N_Z + id_z[p];
    task[id].push_back(p);
    task_count[id]++;
  }
  delete[] id_x;
  delete[] id_y;
  delete[] id_z;

  // analyze(task_count, N_X * N_Y * N_Z, P);

  // assign the task to tasklists
  bool *in_queue = new bool[N_X * N_Y * N_Z], *vis = new bool[N_X * N_Y * N_Z];
  thread* thread_pool = new thread[numThreads];
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N_X * N_Y * N_Z; i++) in_queue[i] = vis[i] = false;
  task_left = N_X * N_Y * N_Z;
  for (int i = 0; i < N_X; i += 2) {
    for (int j = 0; j < N_Y; j += 2) {
      for (int k = 0; k < N_Z; k += 2)
        task_list.push(i * N_Y * N_Z + j * N_Z + k);
    }
  }

  // TEND(init);
  // TPRINT(init, "  Init Convolution ADJ");

  for (int i = 0; i < numThreads; i++) {
    thread_pool[i] = thread([&, this, raw, task, vis] {
      int id = -1;
      while (task_left.load()) {
        if (id >= 0) {
          unique_lock<mutex> lock{this->m_lock};
          int x = id / (N_Y * N_Z);
          id %= (N_Y * N_Z);
          int y = id / N_Z;
          int z = id % N_Z;
          const int code = (x & 1) << 2 | (y & 1) << 1 | (z & 1);
          if (GrayCodeOrder[code] != 7) {
            const int nxt_code = GrayCode[GrayCodeOrder[code] + 1];
            const int delta = code ^ nxt_code;
            switch (delta) {
              case 4:
                if (x + 1 < N_X) {
                  Probe(x, 1, N_X);
                  x -= 2;
                }
                if (x - 1 >= 0) Probe(x, -1, N_X);
                break;
              case 2:
                if (y + 1 < N_Y) {
                  Probe(y, 1, N_Y);
                  y -= 2;
                }
                if (y - 1 >= 0) Probe(y, -1, N_Y);
                break;
              case 1:
                if (z + 1 < N_Z) {
                  Probe(z, 1, N_Z);
                  z -= 2;
                }
                if (z - 1 >= 0) Probe(z, -1, N_Z);
                break;
            }
          }
          id = -1;
        }
        while (task_left.load()) {
          unique_lock<mutex> lock{this->m_lock};
          cv_task.wait(lock, [&, this, task] {
            return !this->task_left.load() || !task_list.empty();
          });
          if (!this->task_left.load()) break;
          id = task_list.top();
          task_list.pop();
          task_left--;
          break;
        }
        if (id < 0) {
          cv_task.notify_all();
          break;
        }
        ConvolutionAdjCore(raw, task[id]);
        vis[id] = true;
      }
    });
  }
  for (int i = 0; i < numThreads; i++) thread_pool[i].join();
  delete[] thread_pool;
  delete[] in_queue;
  delete[] vis;
  delete[] task;
}
#undef Probe

/* Adjoint NUFFT transform */
void NUFFT3D::adj(complex<float>* raw, complex<float>* u) {
  TSTART(nufft);

  // Push to grid
  TSTART(fftw);
  for (int i = 0; i < N2 * N2 * N2; i++) {
    f[i] = 0;
  }
  TEND(fftw);
  TPRINT(fftw, "  Init_F ADJ");
  TSTART(fftw)
  ConvolutionAdj(raw);
  TEND(fftw);
  TPRINT(fftw, "  Convolution ADJ");
  // (Oversampled) FFT
  TSTART(fftw);
  chop3D(f, N2, N2, N2, postchopX, postchopY, postchopZ);
  TEND(fftw);
  TPRINT(fftw, "  PostChop ADJ");
  TSTART(fftw);
  fftw_execute(adjPlan);
  TEND(fftw);
  TPRINT(fftw, "  FFTW ADJ");
  TSTART(fftw);
  chop3D(f, N2, N2, N2, prechopX, prechopY, prechopZ);
  TEND(fftw);
  TPRINT(fftw, "  PreChop ADJ");
  // Deapodize and truncate
  int startX = N * (OF - 1) / 2;
  int startY = N * (OF - 1) / 2;
  int startZ = N * (OF - 1) / 2;
  TSTART(fftw);
  for (int i = 0; i < N * N * N; i++) {
    u[i] = 0;
  }
  TEND(fftw);
  TPRINT(fftw, "  Init_U ADJ");
  TSTART(fftw)
  for (int x = 0; x < N; x++) {
    for (int y = 0; y < N; y++) {
      for (int z = 0; z < N; z++) {
        u[x * N * N + y * N + z] =
            f[(x + startX + offsetX) * N2 * N2 + (y + startY + offsetY) * N2 +
              (z + startZ + offsetZ)] /
            q[x * N * N + y * N + z];
      }
    }
  }
  TEND(fftw);
  TPRINT(fftw, "  Roundoff_Corr ADJ");
  TEND(nufft);
  TPRINT(nufft, "NUFFT ADJ")
  return;
}

/* Internal lookup table generation function for interpolation kernel
 * (Kaiser-Bessel) */
void NUFFT3D::buildLUT() {
  LUT = new float[L];
  float* d = linspace<float>(0, W, L);
  float pi = boost::math::constants::pi<float>();
  float alpha = pi * sqrt(((2 * (float)W / OF) * (OF - 0.5)) *
                              ((2 * (float)W / OF) * (OF - 0.5)) -
                          0.8);
  for (int l = 0; l < L; l++) {
    LUT[l] = boost::math::cyl_bessel_i(
                 0, alpha * sqrt(1 - (d[l] * d[l]) / (W * W))) /
             boost::math::cyl_bessel_i(0, alpha);
  }
}

/* Internal scaling generation function */
void NUFFT3D::getScalingFunction() {
  float dx, dy, dz;
  float s = 0;

  // Create a volume with a copy of the interpolation kernel centered at the
  // origin, then normalize
  for (int i = 0; i < N2 * N2 * N2; i++) {
    f[i] = 0;
  }
  for (int x = N2 / 2 - W; x <= N2 / 2 + W; x++) {
    dx = abs(((float)x - N2 / 2) / W);
    for (int y = N2 / 2 - W; y <= N2 / 2 + W; y++) {
      dy = abs(((float)y - N2 / 2) / W);
      for (int z = N2 / 2 - W; z <= N2 / 2 + W; z++) {
        dz = abs(((float)z - N2 / 2) / W);
        f[x * N2 * N2 + y * N2 + z] = complex<float>(
            LUT[(int)round((L - 1) * dx)] * LUT[(int)round((L - 1) * dy)] *
                LUT[(int)round((L - 1) * dz)],
            0);
        s = s + norm(f[x * N2 * N2 + y * N2 + z]);
      }
    }
  }
  s = sqrt(s);
  for (int x = N2 / 2 - W; x <= N2 / 2 + W; x++) {
    for (int y = N2 / 2 - W; y <= N2 / 2 + W; y++) {
      for (int z = N2 / 2 - W; z <= N2 / 2 + W; z++) {
        f[x * N2 * N2 + y * N2 + z] = f[x * N2 * N2 + y * N2 + z] / s;
      }
    }
  }

  // (Oversampled) FFT
  chop3D(f, N2, N2, N2, postchopX, postchopY, postchopZ);
  fftw_execute(adjPlan);
  chop3D(f, N2, N2, N2, prechopX, prechopY, prechopZ);

  // Truncate and keep only the real component (presuming Fourier domain
  // symmetry)
  q = new float[N * N * N];
  int startX = N * (OF - 1) / 2;
  int startY = N * (OF - 1) / 2;
  int startZ = N * (OF - 1) / 2;
  for (int i = 0; i < N * N * N; i++) {
    q[i] = 0;
  }
  for (int x = 0; x < N; x++) {
    for (int y = 0; y < N; y++) {
      for (int z = 0; z < N; z++) {
        q[x * N * N + y * N + z] =
            real(f[(x + startX + offsetX) * N2 * N2 +
                   (y + startY + offsetY) * N2 + (z + startZ + offsetZ)]);
      }
    }
  }

  return;
}
