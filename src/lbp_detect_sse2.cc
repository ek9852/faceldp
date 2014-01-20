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

#include <stdio.h>
#include <stdint.h>
#include <emmintrin.h>
#include <math.h>
#include "common.h"
#include "lbp.h"

__attribute__((constructor)) static void lbp_detect_sse2_init( void );

static inline unsigned int
get_value_bilinear(unsigned int *img, float x, float y, int width, int height)
{
    unsigned int pt[4];
    unsigned int *p = img + (int)x + (int)y * width;
    double x1, x2;
    double y1, y2;

    pt[0] = *p;
    pt[1] = *(p+1);
    pt[2] = *(p+width);
    pt[3] = *(p+width+1);

    x1 = floorf(x);
    x2 = x1 + 1; 
    y1 = floorf(y);
    y2 = y1 + 1; 

    return (unsigned int)(pt[0] * (x2 - x) * (y2 - y) +
               pt[1] * (x - x1) * (y2 - y) +
               pt[2] * (x2 - x) * (y - y1) +
               pt[3] * (x - x1) * (y - y1));
}

static void
get_interpolated_integral_value(struct lbp_rect *r, struct weak_classifier *c, unsigned int *img, int x, int y, int width, int height, float scale, unsigned int *p)
{
    /*  0  1  2  3
     *  4  5  6  7
     *  8  9 10 11
     * 12 13 14 15
     */
    unsigned int i[16];
    float fx[4], fy[4];

    fx[0] = (float)x + r->x * scale;
    fx[1] = fx[0] + r->w*scale;
    fx[2] = fx[0] + r->w*scale*2;
    fx[3] = fx[0] + r->w*scale*3;
    fy[0] = (float)y + r->y * scale;
    fy[1] = fy[0] + r->h*scale;
    fy[2] = fy[0] + r->h*scale*2;
    fy[3] = fy[0] + r->h*scale*3;

    i[0] = get_value_bilinear(img, fx[0], fy[0], width, height);
    i[1] = get_value_bilinear(img, fx[1], fy[0], width, height);
    i[2] = get_value_bilinear(img, fx[2], fy[0], width, height);
    i[3] = get_value_bilinear(img, fx[3], fy[0], width, height);

    i[4] = get_value_bilinear(img, fx[0], fy[1], width, height);
    i[5] = get_value_bilinear(img, fx[1], fy[1], width, height);
    i[6] = get_value_bilinear(img, fx[2], fy[1], width, height);
    i[7] = get_value_bilinear(img, fx[3], fy[1], width, height);

    i[8]  = get_value_bilinear(img, fx[0], fy[2], width, height);
    i[9]  = get_value_bilinear(img, fx[1], fy[2], width, height);
    i[10] = get_value_bilinear(img, fx[2], fy[2], width, height);
    i[11] = get_value_bilinear(img, fx[3], fy[2], width, height);

    i[12] = get_value_bilinear(img, fx[0], fy[3], width, height);
    i[13] = get_value_bilinear(img, fx[1], fy[3], width, height);
    i[14] = get_value_bilinear(img, fx[2], fy[3], width, height);
    i[15] = get_value_bilinear(img, fx[3], fy[3], width, height);

    p[0] = (i[0] - i[1] - i[4] + i[5]);
    p[1] = (i[1] - i[2] - i[5] + i[6]);
    p[2] = (i[2] - i[3] - i[6] + i[7]);
    p[3] = (i[4] - i[5] - i[8] + i[9]);
    p[4] = (i[5] - i[6] - i[9] + i[10]);
    p[5] = (i[6] - i[7] - i[10] + i[11]);
    p[6] = (i[8] - i[9] - i[12] + i[13]);
    p[7] = (i[9] - i[10] - i[13] + i[14]);
    p[8] = (i[10] - i[11] - i[14] + i[15]);
}

#define DECLARE_ASM_CONST(n,t,v)    static const t __attribute__((used)) __attribute__ ((aligned (n))) v

DECLARE_ASM_CONST(16, uint8_t, lbp_weight)[] = {0x80, 0x40, 0x20, 0x1, 0, 0x10, 0x2, 0x4, 0x8, 0, 0, 0, 0, 0, 0, 0};
DECLARE_ASM_CONST(16, uint32_t, sign)[] = {0x80000000, 0x80000000, 0x80000000, 0x80000000};

static float
lbp_classify_sse2(struct lbp_rect *r, struct weak_classifier *c, unsigned int *img, int x, int y, int width, int height, float scale)
{
    /* REVISIT performance almost same as plain c */
    /* 0 1 2
     * 3 4 5 
     * 6 7 8
     */
    union U {
        __m128i m[3];
        signed short s[8];
        unsigned int p[9];
    } res;

    get_interpolated_integral_value(&r[c->rect_idx], c, img, x, y, width, height, scale, res.p);

    __asm__ volatile (
        /* xmm0 zero
         * xmm1 res.m[0]
         * xmm2 res.m[1]
         * xmm3 res.m[2]
         * xmm4 sign
         * xmm5 center
         * xmm6 lbp_weight
         */
        "pxor %%xmm0,       %%xmm0       \n\t"
        "movdqa %[sign],    %%xmm4       \n\t"
        "pshufd $0, %[center], %%xmm5    \n\t"
        "movdqa %[lbp_weight], %%xmm6    \n\t"

        "movdqa %[m0],      %%xmm1       \n\t"
        "movdqa %[m1],      %%xmm2       \n\t"
        "movdqa %[m2],      %%xmm3       \n\t"

        /* we want to compare unsigned */
        "pxor %%xmm4,       %%xmm1       \n\t"
        "pxor %%xmm4,       %%xmm2       \n\t"
        "pxor %%xmm4,       %%xmm3       \n\t"
        "pxor %%xmm4,       %%xmm5       \n\t"

        /* Suppose to be greater than or equal, but sse2 only support greater than */
        "pcmpgtd %%xmm5,    %%xmm1       \n\t"
        "pcmpgtd %%xmm5,    %%xmm2       \n\t"
        "pcmpgtd %%xmm5,    %%xmm3       \n\t"

        /* pack mask into bytes */
        "packssdw %%xmm2,   %%xmm1       \n\t"
        "packssdw %%xmm3,   %%xmm3       \n\t"
        "packsswb %%xmm3,   %%xmm1       \n\t"

        /* AND mask with lbp weight and sum the code */
        "pand %%xmm6,       %%xmm1       \n\t"
        "psadbw %%xmm1,     %%xmm0       \n\t"

        "movdqa %%xmm0,     %[m0]        \n\t"

        : [m0] "+m" (res.m[0])
        : [m1] "m" (res.m[1]),
          [m2] "m" (res.m[2]),
          [center] "m" (res.p[4]),
          [sign] "m" (*sign),
          [lbp_weight] "m" (*lbp_weight)
    );
    
    int lbp_code = res.s[4] + res.s[0];

    if (c->lbpmap[lbp_code >> 5] & (1 << (lbp_code & 31))) {
        return c->neg;
    } else {
        return c->pos;
    }
}

#define cpuid(func,ax,bx,cx,dx)\
	__asm__ __volatile__ ("cpuid":\
	"=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));
/* %edx */
#define bit_SSE2        (1 << 26)

static void lbp_detect_sse2_init(void)
{
    /* check if cpu support sse2 */
 
    unsigned int eax, ebx, ecx, edx;
    
    cpuid(1, eax, ebx, ecx, edx);
    
    if (edx & bit_SSE2) {
        pf_lbp_classify = lbp_classify_sse2;
    }
}


