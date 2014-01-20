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

#ifndef _LBP_H
#define _LBP_H

#include <stdint.h>

struct lbp_task {
    int x;
    int y;
    float scale;
};

struct lbp_rect {
    int x;
    int y;
    int w;
    int h;
};

struct weak_classifier {
    int rect_idx;
    int32_t lbpmap[8]; /* loading need to be singed */
    float pos;
    float neg;
};

struct stage {
    float stage_threshold;
    int num_weak_classifiers;
    struct weak_classifier *classifiers;
};

struct lbp_data {
    int feature_width;
    int feature_height;
    int num_stages;
    struct stage *s;
    int num_rects;
    struct lbp_rect *r;
};

struct lbp_para {
    float scaling_factor;
    int step_scale_x;
    int step_scale_y;
    float tracking_scale_down;
    float tracking_scale_up;
    float tracking_offset; // width percentage
    float eps;
    int group_threshold;
    int min_face_width;
};

typedef float (*lbp_classify_t) (struct lbp_rect *r, struct weak_classifier *c, unsigned int *img, int x, int y, int width, int height, float scale);
extern lbp_classify_t pf_lbp_classify;

#endif
