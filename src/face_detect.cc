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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "face_detect.h"
#include "lbp_detect.h"
#include "integral_image.h"
#include "common.h"

struct face_det {
    struct lbp *l;
    int width;
    int height;
    unsigned int *integral_img;
};

int
face_detector_detect(struct face_det *f, unsigned char *y, struct face *fa, int *maxfaces)
{
    face_detector_gen_integral_image(f->integral_img, y, f->width, f->height);
    face_detector_lbp_detect(f->l, f->integral_img, fa, maxfaces);

    return 0;
}

int
face_detector_tracking(struct face_det *f, unsigned char *y, struct face *fa, int faces, int *maxfaces)
{
    face_detector_gen_integral_image(f->integral_img, y, f->width, f->height);
    face_detector_lbp_tracking(f->l, f->integral_img, fa, faces, maxfaces);

    return 0;
}

struct face_det *
face_detector_create(int width, int height, int minimum_face_width)
{
    struct face_det *f;
    f = (struct face_det *)calloc(1, sizeof(struct face_det));

    f->l = face_detector_lbp_create(width, height, minimum_face_width);

    if (!f->l) {
        free(f);
        return NULL;
    }
    f->width = width;
    f->height = height;
    f->integral_img = (unsigned int *)malloc(width * height * sizeof(unsigned int));

    return f;
}

void
face_detector_destroy(struct face_det *f)
{
    if (f->l)
        face_detector_lbp_destroy(f->l);
    free(f->integral_img);
    free(f);
}
