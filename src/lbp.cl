/*
 *  facelbp - Face detection using Multi-scale Block Local Binary Pattern algorithm
 *
 *  Copyright (C) 2013 Keith Mok <ek9852@gmail.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/* support for int32 global base atomic is mandatory for opencl > 1.0 */
#if __OPENCL_VERSION__ == 100
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

#ifdef cl_khr_fp64
#if __OPENCL_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
typedef double F;
typedef double2 F2;
typedef double4 F4;
typedef double8 F8;
typedef double16 F16;
typedef long8 FI8;
#define convert_F2 convert_double2
#else
typedef float F;
typedef float2 F2;
typedef float4 F4;
typedef float8 F8;
typedef float16 F16;
typedef int8 FI8;
#define convert_F2 convert_float2
#endif

typedef struct
{
  int x;
  int y;
  float scale;
} lbp_task;

typedef struct {
  int x;
  int y;
  int w;
  int h;
} lbp_rect;

typedef struct {
  int rect_idx;
  int lbpmap[8];
  float pos;
  float neg;
} weak_classifier;

typedef struct {
  float stage_threshold;
  int num_weak_classifiers;
  int classifier_start_index;
} stage;

F get_value_bilinear(
    __global const uint *img,
    float2 pt)
{
  int pt_offset = mad24(pt.y, WIDTH, pt.x); //(int)x + (int)y * width;
  uint2 upper, lower;
  F2 upperf, lowerf;
  float2 intpart;
  float2 st;

  upper = vload2(0, img + pt_offset);
  lower = vload2(0, img + pt_offset + WIDTH);

  upperf = convert_F2(upper);
  lowerf = convert_F2(lower);

  st = modf(pt, &intpart);

  F2 mid = mix(upperf, lowerf, (F)st.y);
  return mix(mid.x, mid.y, (F)st.x);
}

void get_interpolated_integral_value(
    __global const lbp_rect *r,
    __global const uint *img,
    int x,
    int y,
    float scale,
    F8 *p, F *center)
{
  /*  0  1  2  3
   *  4  5  6  7
   *  8  9 10 11
   * 12 13 14 15
   */
  F16 i;
  F4 fx, fy;

  fx = (F4)(x + r->x * scale);
  fx.s1 += r->w * scale;
  fx.s2 += r->w * scale * 2;
  fx.s3 += r->w * scale * 3;
  fy = (F4)(y + r->y * scale);
  fy.s1 += r->h * scale;
  fy.s2 += r->h * scale * 2;
  fy.s3 += r->h * scale * 3;

  i.s0 = get_value_bilinear(img, (float2)(fx.s0, fy.s0));
  i.s1 = get_value_bilinear(img, (float2)(fx.s1, fy.s0));
  i.s2 = get_value_bilinear(img, (float2)(fx.s2, fy.s0));
  i.s3 = get_value_bilinear(img, (float2)(fx.s3, fy.s0));

  i.s4 = get_value_bilinear(img, (float2)(fx.s0, fy.s1));
  i.s5 = get_value_bilinear(img, (float2)(fx.s1, fy.s1));
  i.s6 = get_value_bilinear(img, (float2)(fx.s2, fy.s1));
  i.s7 = get_value_bilinear(img, (float2)(fx.s3, fy.s1));

  i.s8 = get_value_bilinear(img, (float2)(fx.s0, fy.s2));
  i.s9 = get_value_bilinear(img, (float2)(fx.s1, fy.s2));
  i.sa = get_value_bilinear(img, (float2)(fx.s2, fy.s2));
  i.sb = get_value_bilinear(img, (float2)(fx.s3, fy.s2));

  i.sc = get_value_bilinear(img, (float2)(fx.s0, fy.s3));
  i.sd = get_value_bilinear(img, (float2)(fx.s1, fy.s3));
  i.se = get_value_bilinear(img, (float2)(fx.s2, fy.s3));
  i.sf = get_value_bilinear(img, (float2)(fx.s3, fy.s3));

  *center = i.s5 - i.s6 - i.s9 + i.sa;
  (*p).s01234567 = i.s0124689a - i.s123579ab - i.s4568acde + i.s5679bdef;
}

float lbp_classify(
    __global const lbp_rect *r,
    __global const weak_classifier *c,
    __global const uint *img,
    int x,
    int y,
    float scale)
{
  /* 0 1 2
   * 3 c 4 
   * 5 6 7
   */
  F8 p;
  F8 cf;
  F center;
  int lbp_code;

  get_interpolated_integral_value(&r[c->rect_idx], img, x, y, scale, &p, &center);

  lbp_code = 0;
  cf = (F8)(center);
  FI8 lbpmask = isgreaterequal(p, cf);

  lbpmask &= (FI8)(128, 64, 32, 1, 16, 2, 4, 8);
  lbp_code = lbpmask.s0 | lbpmask.s1 | lbpmask.s2 | lbpmask.s3 |
             lbpmask.s4 | lbpmask.s5 | lbpmask.s6 | lbpmask.s7;

  if (c->lbpmap[lbp_code >> 5] & (1 << (lbp_code & 31))) {
    return c->neg;
  } else {
    return c->pos;
  }
}

__kernel void lbp(
    __global const lbp_rect *rect,
    __global const weak_classifier *c,
    __global const stage *s,
    __global const lbp_task *t,
#if __OPENCL_VERSION__ == 100
    __global unsigned int *result_counter,
#else
    volatile __global unsigned int *result_counter,
#endif
    __global int *result,
    __global const uint *img
    )
{
  int gid = get_global_id(0);

  for (int i = 0; i < NUM_STAGES; i++) {
    /* loop all weak classifiers */
    float threshold = 0;
    int start_idx = s[i].classifier_start_index;
    for (int j = 0; j < s[i].num_weak_classifiers; j++) {
      threshold += lbp_classify(rect, &c[start_idx + j], img, t[gid].x, t[gid].y, t[gid].scale);
    }
    if (threshold < s[i].stage_threshold) {
      return;
    }
  }
#if __OPENCL_VERSION__ == 100
  unsigned int ind = atom_inc(result_counter);
#else
  unsigned int ind = atomic_inc(result_counter);
#endif
  result[ind] = gid;
}
