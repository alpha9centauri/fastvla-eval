// Exact convex-convex distance (GJK) -- a one-to-one C++17 port of
// /data_new/vchandr2/vla-safety-interp/src/physics/convex_distance.py
// (gjk_distance and _closest_on_simplex), md5 of the source recorded in
// fastvla/labeler/DESIGN.md.
//
// M2 ports the algorithm without changing it: same iteration structure, same
// termination rules, same literal tolerances (1e-10, 1e-12, -1e-12), same
// first-wins tie-breaking in the subset search and in the support-function argmax.
// The ONE deliberate difference is that numpy.linalg.solve on a (k+1)x(k+1) KKT
// system is replaced by a fixed-size Gaussian elimination with partial pivoting,
// which is what LAPACK's dgesv does for these sizes. Nothing else is touched.
//
// Matching numpy's details is what makes 1e-6 m parity achievable:
//   * np.argmax / np.argmin return the FIRST extremal index, so the support
//     function uses strict > / < and never replaces on a tie.
//   * the subset loop keeps a strictly smaller squared norm, so on a tie the
//     lowest mask wins; the mask order here is identical.
//   * np.linalg.solve raises only on an exactly singular factor, so a zero pivot
//     skips the subset (Python `continue`) and a merely tiny one does not.
#pragma once

#include <cmath>
#include <cstdint>

namespace fastgjk {

// Solve A x = b for n <= 5, A row-major n*n, destroying A and b.
// Returns false on an exactly singular pivot, mirroring np.linalg.LinAlgError.
inline bool solve_lin(double* A, double* b, int n, double* x) {
  int piv[5];
  for (int i = 0; i < n; ++i) piv[i] = i;
  for (int c = 0; c < n; ++c) {
    // partial pivoting: largest magnitude in the column
    int best = c;
    double bestv = std::fabs(A[c * n + c]);
    for (int r = c + 1; r < n; ++r) {
      const double v = std::fabs(A[r * n + c]);
      if (v > bestv) { bestv = v; best = r; }
    }
    if (bestv == 0.0) return false;   // exactly singular, as LAPACK reports it
    if (best != c) {
      for (int k = 0; k < n; ++k) {
        const double t = A[c * n + k]; A[c * n + k] = A[best * n + k]; A[best * n + k] = t;
      }
      const double t = b[c]; b[c] = b[best]; b[best] = t;
    }
    const double inv = 1.0 / A[c * n + c];
    for (int r = c + 1; r < n; ++r) {
      const double f = A[r * n + c] * inv;
      if (f == 0.0) continue;
      for (int k = c; k < n; ++k) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    double s = b[i];
    for (int k = i + 1; k < n; ++k) s -= A[i * n + k] * x[k];
    x[i] = s / A[i * n + i];
  }
  return true;
}

// Closest point to the origin on conv(W), |W| = n <= 4.
// Writes the point to v and the surviving subset to idx; returns its size.
inline int closest_on_simplex(const double W[4][3], int n, double v[3], int* idx) {
  double best_v[3] = {0.0, 0.0, 0.0};
  double best_n2 = INFINITY;
  int best_idx[4] = {0, 0, 0, 0};
  int best_k = 0;

  for (int mask = 1; mask < (1 << n); ++mask) {
    int sub[4], k = 0;
    for (int i = 0; i < n; ++i) if (mask & (1 << i)) sub[k++] = i;

    double lam[4], cand[3];
    if (k == 1) {
      lam[0] = 1.0;
      cand[0] = W[sub[0]][0]; cand[1] = W[sub[0]][1]; cand[2] = W[sub[0]][2];
    } else {
      // KKT: [[G, 1], [1^T, 0]] [lam; mu] = [0; 1], G = P P^T
      const int m = k + 1;
      double A[25] = {0.0}, b[5] = {0.0}, sol[5] = {0.0};
      for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
          const double* wi = W[sub[i]];
          const double* wj = W[sub[j]];
          A[i * m + j] = wi[0] * wj[0] + wi[1] * wj[1] + wi[2] * wj[2];
        }
        A[i * m + k] = 1.0;
        A[k * m + i] = 1.0;
      }
      b[k] = 1.0;
      if (!solve_lin(A, b, m, sol)) continue;   // singular -> skip, as Python does
      bool neg = false;
      for (int i = 0; i < k; ++i) {
        lam[i] = sol[i];
        if (lam[i] < -1e-12) neg = true;
      }
      if (neg) continue;
      cand[0] = cand[1] = cand[2] = 0.0;
      for (int i = 0; i < k; ++i) {
        const double* wi = W[sub[i]];
        cand[0] += lam[i] * wi[0];
        cand[1] += lam[i] * wi[1];
        cand[2] += lam[i] * wi[2];
      }
    }
    const double n2 = cand[0] * cand[0] + cand[1] * cand[1] + cand[2] * cand[2];
    if (n2 < best_n2) {     // strictly smaller: lowest mask wins a tie, as in Python
      best_n2 = n2;
      best_v[0] = cand[0]; best_v[1] = cand[1]; best_v[2] = cand[2];
      best_k = k;
      for (int i = 0; i < k; ++i) best_idx[i] = sub[i];
    }
  }

  if (best_k == 0) {        // Python: `if best_v is None: return Wm[0], [0]`
    v[0] = W[0][0]; v[1] = W[0][1]; v[2] = W[0][2];
    idx[0] = 0;
    return 1;
  }
  v[0] = best_v[0]; v[1] = best_v[1]; v[2] = best_v[2];
  for (int i = 0; i < best_k; ++i) idx[i] = best_idx[i];
  return best_k;
}

// Distance between conv(A) and conv(B); 0.0 if they intersect.
// A is na*3 row-major, B is nb*3 row-major.
inline double gjk_distance(const double* A, int na, const double* B, int nb,
                           int max_iter = 64, double tol = 1e-12) {
  double v[3] = {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
  if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < tol) {
    v[0] = 1e-9; v[1] = 0.0; v[2] = 0.0;
  }
  double W[4][3];
  int nW = 0;

  for (int it = 0; it < max_iter; ++it) {
    const double nv = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (nv < 1e-10) return 0.0;
    const double d[3] = {-v[0] / nv, -v[1] / nv, -v[2] / nv};

    // support: first maximal index for A, first minimal for B (np.argmax/argmin)
    int ia = 0;
    double sa = A[0] * d[0] + A[1] * d[1] + A[2] * d[2];
    for (int i = 1; i < na; ++i) {
      const double s = A[i * 3] * d[0] + A[i * 3 + 1] * d[1] + A[i * 3 + 2] * d[2];
      if (s > sa) { sa = s; ia = i; }
    }
    int ib = 0;
    double sb = B[0] * d[0] + B[1] * d[1] + B[2] * d[2];
    for (int i = 1; i < nb; ++i) {
      const double s = B[i * 3] * d[0] + B[i * 3 + 1] * d[1] + B[i * 3 + 2] * d[2];
      if (s < sb) { sb = s; ib = i; }
    }
    const double w[3] = {A[ia * 3] - B[ib * 3],
                         A[ia * 3 + 1] - B[ib * 3 + 1],
                         A[ia * 3 + 2] - B[ib * 3 + 2]};

    const double vw = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    if (nv - vw / nv <= 1e-10) return nv;

    W[nW][0] = w[0]; W[nW][1] = w[1]; W[nW][2] = w[2];
    ++nW;

    int idx[4];
    const int k = closest_on_simplex(W, nW, v, idx);
    double Wn[4][3];
    for (int i = 0; i < k; ++i) {
      Wn[i][0] = W[idx[i]][0]; Wn[i][1] = W[idx[i]][1]; Wn[i][2] = W[idx[i]][2];
    }
    for (int i = 0; i < k; ++i) {
      W[i][0] = Wn[i][0]; W[i][1] = Wn[i][1]; W[i][2] = Wn[i][2];
    }
    nW = k;

    if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < tol) return 0.0;
    if (nW == 4) return 0.0;   // origin enclosed -> intersecting
  }
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// DistanceEngine.distance: pads are the swept radii of the two cores.
inline double distance_padded(const double* A, int na, double padA,
                              const double* B, int nb, double padB) {
  const double d = gjk_distance(A, na, B, nb) - padA - padB;
  return d > 0.0 ? d : 0.0;
}

}  // namespace fastgjk
