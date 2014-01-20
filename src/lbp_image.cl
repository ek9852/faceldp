/*
 * facelbp - Face detection using Multi-scale Block Local Binary Pattern algorithm
 *
 * Copyright (C) 2013 Keith Mok <ek9852@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#if __OPENCL_VERSION__ == 100
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

// CLX_FILTER_LINEAR on UINT32 is not support by opencl
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

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
} lbp_rect ;

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

void get_interpolated_integral_value(
    __global const lbp_rect *r,
    __global const weak_classifier *c,
    image2d_t img,
    int x,
    int y,
    float scale,
    uint8 *p, uint *center)
{
  /*  0 1 2 3
   *   0 1 2
   *  4 5 6 7
   *   3 c 4
   *  8 9 a b
   *   5 6 7
   *  c d e f
   */
  uint16 i;
  float4 fx, fy;

  fx.s0 = (float)x + r->x * scale;
  fx.s1 = fx.s0 + r->w * scale;
  fx.s2 = fx.s0 + r->w * scale * 2;
  fx.s3 = fx.s0 + r->w * scale * 3;
  fy.s0 = (float)y + r->y * scale;
  fy.s1 = fy.s0 + r->h * scale;
  fy.s2 = fy.s0 + r->h * scale * 2;
  fy.s3 = fy.s0 + r->h * scale * 3;

  i.s0 = read_imageui(img, sampler, (float2)(fx.s0, fy.s0)).x;
  i.s1 = read_imageui(img, sampler, (float2)(fx.s1, fy.s0)).x;
  i.s2 = read_imageui(img, sampler, (float2)(fx.s2, fy.s0)).x;
  i.s3 = read_imageui(img, sampler, (float2)(fx.s3, fy.s0)).x;

  i.s4 = read_imageui(img, sampler, (float2)(fx.s0, fy.s1)).x;
  i.s5 = read_imageui(img, sampler, (float2)(fx.s1, fy.s1)).x;
  i.s6 = read_imageui(img, sampler, (float2)(fx.s2, fy.s1)).x;
  i.s7 = read_imageui(img, sampler, (float2)(fx.s3, fy.s1)).x;

  i.s8 = read_imageui(img, sampler, (float2)(fx.s0, fy.s2)).x;
  i.s9 = read_imageui(img, sampler, (float2)(fx.s1, fy.s2)).x;
  i.sa = read_imageui(img, sampler, (float2)(fx.s2, fy.s2)).x;
  i.sb = read_imageui(img, sampler, (float2)(fx.s3, fy.s2)).x;

  i.sc = read_imageui(img, sampler, (float2)(fx.s0, fy.s3)).x;
  i.sd = read_imageui(img, sampler, (float2)(fx.s1, fy.s3)).x;
  i.se = read_imageui(img, sampler, (float2)(fx.s2, fy.s3)).x;
  i.sf = read_imageui(img, sampler, (float2)(fx.s3, fy.s3)).x;

  *center = i.s5 - i.s6 - i.s9 + i.sa;
  (*p).s01234567 = i.s0124689a - i.s123579ab - i.s4568acde + i.s5679bdef;
}

float lbp_classify(
    __global const lbp_rect *r,
    __global const weak_classifier *c,
    image2d_t img,
    int x,
    int y,
    float scale)
{
  /* 0 1 2
   * 3 c 4 
   * 5 6 7
   */
  uint8 p;
  uint center;
  int lbp_code;

  get_interpolated_integral_value(&r[c->rect_idx], c, img, x, y, scale, &p, &center);

  lbp_code = 0;
  if (p.s0 >= center) lbp_code |= 128;
  if (p.s1 >= center) lbp_code |= 64;
  if (p.s2 >= center) lbp_code |= 32;
  if (p.s3 >= center) lbp_code |= 1;
  if (p.s4 >= center) lbp_code |= 16;
  if (p.s5 >= center) lbp_code |= 2;
  if (p.s6 >= center) lbp_code |= 4;
  if (p.s7 >= center) lbp_code |= 8;

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
    image2d_t img // input integral image
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
