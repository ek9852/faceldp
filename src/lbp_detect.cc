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

#include <vector>
#include <iostream>
#include <fstream>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include "lbp_detect.h"
#include "group_rectangle.h"
#include "lbp.h"
#include "common.h"
#ifdef USE_OPENCL
#include "lbp_cl.h"
#endif

#define DATA_FILE_PATH "frontalface.txt"

/* default handler */
static float lbp_classify(struct lbp_rect *r, struct weak_classifier *c, unsigned int *img, int x, int y, int width, int height, float scale);
lbp_classify_t pf_lbp_classify = lbp_classify;

struct lbp {
    struct lbp_data data;
    struct lbp_para para;
    int width;
    int height;

#ifdef USE_OPENCL
    struct lbp_cl* cl;
#endif

    std::vector<struct lbp_rect> detected_r; /* must use new/delete instead of malloc/free because of this */
    std::vector<struct lbp_task> tasks; /* task to scan all size and positions  */
};

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
    float fx, fy;

    fx = (float)x + r->x * scale;
    fy = (float)y + r->y * scale;

    i[0] = get_value_bilinear(img, fx,                  fy, width, height);
    i[1] = get_value_bilinear(img, fx + r->w*scale,     fy, width, height);
    i[2] = get_value_bilinear(img, fx + r->w*scale * 2, fy, width, height);
    i[3] = get_value_bilinear(img, fx + r->w*scale * 3, fy, width, height);

    i[4] = get_value_bilinear(img, fx,                  fy + r->h*scale, width, height);
    i[5] = get_value_bilinear(img, fx + r->w*scale,     fy + r->h*scale, width, height);
    i[6] = get_value_bilinear(img, fx + r->w*scale * 2, fy + r->h*scale, width, height);
    i[7] = get_value_bilinear(img, fx + r->w*scale * 3, fy + r->h*scale, width, height);

    i[8]  = get_value_bilinear(img, fx,                  fy + r->h*scale * 2, width, height);
    i[9]  = get_value_bilinear(img, fx + r->w*scale,     fy + r->h*scale * 2, width, height);
    i[10] = get_value_bilinear(img, fx + r->w*scale * 2, fy + r->h*scale * 2, width, height);
    i[11] = get_value_bilinear(img, fx + r->w*scale * 3, fy + r->h*scale * 2, width, height);

    i[12] = get_value_bilinear(img, fx,                  fy + r->h*scale * 3, width, height);
    i[13] = get_value_bilinear(img, fx + r->w*scale,     fy + r->h*scale * 3, width, height);
    i[14] = get_value_bilinear(img, fx + r->w*scale * 2, fy + r->h*scale * 3, width, height);
    i[15] = get_value_bilinear(img, fx + r->w*scale * 3, fy + r->h*scale * 3, width, height);

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

static float
lbp_classify(struct lbp_rect *r, struct weak_classifier *c, unsigned int *img, int x, int y, int width, int height, float scale)
{
    /* 0 1 2
     * 3 4 5 
     * 6 7 8
     */
    unsigned int p[9];
    unsigned char lbp_code;

    get_interpolated_integral_value(&r[c->rect_idx], c, img, x, y, width, height, scale, p);

    lbp_code = 0;
    if (p[0] >= p[4]) lbp_code |= 128;
    if (p[1] >= p[4]) lbp_code |= 64;
    if (p[2] >= p[4]) lbp_code |= 32;
    if (p[3] >= p[4]) lbp_code |= 1;
    if (p[5] >= p[4]) lbp_code |= 16;
    if (p[6] >= p[4]) lbp_code |= 2;
    if (p[7] >= p[4]) lbp_code |= 4;
    if (p[8] >= p[4]) lbp_code |= 8;

    if (c->lbpmap[lbp_code >> 5] & (1 << (lbp_code & 31))) {
        return c->neg;
    } else {
        return c->pos;
    }
}

static int
lbp_detect(struct lbp *l, unsigned int *img, int x, int y, int width, int height, float scale)
{
    /* loop for all stages */
    float threshold;
    int i, j;
    for (i = 0; i < l->data.num_stages; i++) {
        /* loop all weak classifiers */
        threshold = 0;
        for (j = 0; j < l->data.s[i].num_weak_classifiers; j++) {
            threshold += pf_lbp_classify(l->data.r, &l->data.s[i].classifiers[j], img, x, y, width, height, scale);
        }
        if (threshold < l->data.s[i].stage_threshold) {
            /* not matched */
            return 0;
        }
    }
    /* here we pass all the stages and found a match */
    return 1;
}

static void
add_lbp_object(struct lbp *l, int x, int y, float scale)
{
    struct lbp_rect r;
    r.x = x;
    r.y = y;
    r.w = l->data.feature_width * scale;
    r.h = l->data.feature_height * scale;
    l->detected_r.push_back(r);
}

static void
init_task(struct lbp *l)
{
    float scale;
    float scale_max = fminf((float)l->width / l->data.feature_width, (float)l->height / l->data.feature_height);
    float scale_min = (float)l->para.min_face_width / l->data.feature_width;

    for (scale = scale_min; scale < scale_max; scale *= l->para.scaling_factor) {
        int x, y;
        float scaled_width = l->data.feature_width * scale;
        float scaled_height = l->data.feature_height * scale;
        int step_x = scaled_width / l->para.step_scale_x;
        int step_y = scaled_height / l->para.step_scale_y;
        for (x = 0; (x + scaled_width) < (l->width - 1); x += step_x) {
            for (y = 0; (y + scaled_height) < (l->height - 1); y += step_y) {
                struct lbp_task t;
                t.x = x;
                t.y = y;
                t.scale = scale;
                l->tasks.push_back(t);
            }
        }
    }
}

int
face_detector_lbp_tracking(struct lbp *l, unsigned int *img, struct face *fa, int faces, int *maxfaces)
{
    // create a subset of tasks based on previous detected face
    std::vector<struct lbp_task> tasks;
    int i;
    for (i = 0;i < faces; i++) {
        float scale;
        float scale_max = fminf((float)l->width / l->data.feature_width, (float)fa[i].width / l->data.feature_width * l->para.tracking_scale_up);
        float scale_min = fmaxf((float)l->para.min_face_width / l->data.feature_width, (float)fa[i].width / l->data.feature_width * l->para.tracking_scale_down);
        int min_x, min_y, max_x, max_y;
        min_x = fa[i].x - fa[i].width * l->para.tracking_offset;
        min_y = fa[i].y - fa[i].height * l->para.tracking_offset;
        max_x = fa[i].x + fa[i].width * (1 + l->para.tracking_offset);
        max_y = fa[i].y + fa[i].height * (1 + l->para.tracking_offset);

        if (min_x < 0) min_x = 0;
        if (min_y < 0) min_y = 0;
        if (max_x > l->width - 1) max_x = l->width - 1;
        if (max_y > l->height - 1) max_y = l->height - 1;
        for (scale = scale_min; scale < scale_max; scale *= l->para.scaling_factor) {
            int x, y;
            float scaled_width = l->data.feature_width * scale;
            float scaled_height = l->data.feature_height * scale;
            int step_x, step_y;
            step_x = scaled_width / l->para.step_scale_x;
            step_y = scaled_height / l->para.step_scale_y;
            for (x = min_x; (x + scaled_width) < max_x; x += step_x) {
                for (y = min_y; (y + scaled_height) < max_y; y += step_y) {
                    struct lbp_task t;
                    t.x = x;
                    t.y = y;
                    t.scale = scale;
                    tasks.push_back(t);
                }
            }
        }
        
    }
#ifdef USE_OPENCL
    lbp_cl_tracking(l->cl, img, &tasks, l->detected_r);
#else
    int found;
    int width, height;
    width = l->width;
    height = l->height;

    #pragma omp parallel for private(found)
    for (i = 0; i < tasks.size(); i++) {
        found = lbp_detect(l, img, tasks[i].x, tasks[i].y, width, height, tasks[i].scale);
        if (found) {
            #pragma omp critical
            add_lbp_object(l, tasks[i].x, tasks[i].y, tasks[i].scale);
        }
    }
    /* merge overlapped rectangles */
    face_detector_group_rectangle(l->detected_r, l->para.group_threshold, l->para.eps);
#endif
    /* return faces detected after merging */
    if (l->detected_r.size() > (unsigned int)*maxfaces) {
        LOGW("User provided maxface size not large enough");
    } else {
        *maxfaces = l->detected_r.size();
    }

    LOGD("Tracking LBP tested: %ld", tasks.size());
    
    for (i = 0; i < *maxfaces; i++) {
        fa[i].x = l->detected_r[i].x;
        fa[i].y = l->detected_r[i].y;
        fa[i].width = l->detected_r[i].w;
        fa[i].height = l->detected_r[i].h;
        fa[i].confidence_level = 50; /* TODO */
    }
    l->detected_r.clear();

    return 0;
}

int
face_detector_lbp_detect(struct lbp *l, unsigned int *img, struct face *fa, int *maxfaces)
{
    int i;
#ifdef USE_OPENCL
    lbp_cl_detect(l->cl, img, l->detected_r);
#else
    int found;
    int width, height;
    width = l->width;
    height = l->height;

    #pragma omp parallel for private(found)
    for (i = 0; i < l->tasks.size(); i++) {
        found = lbp_detect(l, img, l->tasks[i].x, l->tasks[i].y, width, height, l->tasks[i].scale);
        if (found) {
            #pragma omp critical
            add_lbp_object(l, l->tasks[i].x, l->tasks[i].y, l->tasks[i].scale);
        }
    }
    /* merge overlapped rectangles */
    face_detector_group_rectangle(l->detected_r, l->para.group_threshold, l->para.eps);
#endif

    /* return faces detected after merging */
    if (l->detected_r.size() > (unsigned int)*maxfaces) {
        LOGW("User provided maxface size not large enough");
    } else {
        *maxfaces = l->detected_r.size();
    }

    LOGD("LBP tested: %ld", l->tasks.size());
    
    for (i = 0; i < *maxfaces; i++) {
        fa[i].x = l->detected_r[i].x;
        fa[i].y = l->detected_r[i].y;
        fa[i].width = l->detected_r[i].w;
        fa[i].height = l->detected_r[i].h;
        fa[i].confidence_level = 50; /* TODO */
    }
    l->detected_r.clear();

    return 0;
}

static void
dump_stages_info(struct lbp *l)
{
    int i, total_weak_classifiers;

    LOGD("Num Of stages: %d", l->data.num_stages);

    total_weak_classifiers = 0;

    for (i = 0; i < l->data.num_stages; i++) {
        total_weak_classifiers += l->data.s[i].num_weak_classifiers;
    }

    LOGD("Total weak classifiers: %d", total_weak_classifiers);
}

static int
load_lbp_data(struct lbp *l)
{
    std::ifstream in;
    int i, j;

    in.open(DATADIR"/"DATA_FILE_PATH);

    if (in.fail()) {
        LOGE("Cannot open data file: %s", DATA_FILE_PATH);
        return -EIO;
    }

    in >> l->data.feature_height >> l->data.feature_width >> l->data.num_stages;

    if (in.fail() || in.eof() || (l->data.num_stages <= 0)) {
        l->data.num_stages = 0;
        LOGE("Unexpected end of file %s", DATA_FILE_PATH);
        return -EIO;
    }

    l->data.s = (struct stage *)calloc(l->data.num_stages, sizeof(struct stage));

    for (i = 0; i < l->data.num_stages; i++) {
        in >> l->data.s[i].num_weak_classifiers >> l->data.s[i].stage_threshold;
        if (in.fail() || in.eof() || (l->data.s[i].num_weak_classifiers <= 0)) {
            LOGE("Unexpected end of file %s, stage: %d, weak_classifiers: %d", DATA_FILE_PATH, i + 1, l->data.s[i].num_weak_classifiers);
            l->data.s[i].num_weak_classifiers = 0;
            return -EIO;
        }
        l->data.s[i].classifiers = (struct weak_classifier *)calloc(l->data.s[i].num_weak_classifiers, sizeof(struct weak_classifier));
        for (j = 0; j < l->data.s[i].num_weak_classifiers; j++) {
            in >> l->data.s[i].classifiers[j].rect_idx /* REVISIT sanity check for out of bound ? */
               >> l->data.s[i].classifiers[j].lbpmap[0]
               >> l->data.s[i].classifiers[j].lbpmap[1]
               >> l->data.s[i].classifiers[j].lbpmap[2]
               >> l->data.s[i].classifiers[j].lbpmap[3]
               >> l->data.s[i].classifiers[j].lbpmap[4]
               >> l->data.s[i].classifiers[j].lbpmap[5]
               >> l->data.s[i].classifiers[j].lbpmap[6]
               >> l->data.s[i].classifiers[j].lbpmap[7]
               >> l->data.s[i].classifiers[j].neg
               >> l->data.s[i].classifiers[j].pos;
        }
    }
    in >> l->data.num_rects;
    if (in.fail() || in.eof() || (l->data.num_rects <= 0)) {
        l->data.num_rects = 0;
        LOGE("Unexpected end of file %s", DATA_FILE_PATH);
        return -EIO;
    }
    l->data.r = (struct lbp_rect *)calloc(l->data.num_rects, sizeof(struct lbp_rect));
    for (i = 0; i < l->data.num_rects; i++) {
        in >> l->data.r[i].x >> l->data.r[i].y >> l->data.r[i].w >> l->data.r[i].h;
    }
    if (in.fail()) {
        LOGE("Unexpected end of file %s", DATA_FILE_PATH);
        return -EIO;
    }

    dump_stages_info(l);

    return 0;
}

struct lbp *
face_detector_lbp_create(int width, int height, int minimum_face_width)
{
    struct lbp *l;
    int ret;

    l = new struct lbp();

    l->para.scaling_factor = 1.125;
    l->para.step_scale_x = 8;
    l->para.step_scale_y = 8;
    l->para.tracking_scale_down = 0.5;
    l->para.tracking_scale_up = 1.5;
    l->para.tracking_offset = 0.5;
    l->para.group_threshold = 2;
    l->para.eps = 0.2;
    l->para.min_face_width = minimum_face_width;
    l->width = width;
    l->height = height;

    ret = load_lbp_data(l);
    if (ret) {
        face_detector_lbp_destroy(l);
        return NULL;
    }
    init_task(l);
#ifdef USE_OPENCL
    l->cl = lbp_cl_init(&l->data, &l->para, &l->tasks, width, height);
    if (l->cl == NULL) {
        face_detector_lbp_destroy(l);
        return NULL;
    }
#endif

    return l;
}

void
face_detector_lbp_destroy(struct lbp *l)
{
    int i;
#ifdef USE_OPENCL
    if (l->cl) {
        lbp_cl_destroy(l->cl);
    }
#endif
    for (i = 0; i < l->data.num_stages; i++) {
        free(l->data.s[i].classifiers);
    }
    free(l->data.s);
    free(l->data.r);
    delete l;
}
