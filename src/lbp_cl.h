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

#ifndef _LBP_CL_H
#define _LBP_CL_H

#include <vector>
#include "lbp.h"

struct lbp_cl;

struct lbp_cl *lbp_cl_init(struct lbp_data *data, struct lbp_para *para, 
    std::vector<struct lbp_task> *full_tasks, int width, int height);
int lbp_cl_detect(struct lbp_cl *cl, unsigned int *img, std::vector<struct lbp_rect>& rects);
int lbp_cl_tracking(struct lbp_cl *cl, unsigned int *img, 
    std::vector<struct lbp_task> *tasks,
    std::vector<struct lbp_rect>& rects);
void lbp_cl_destroy(struct lbp_cl *cl);

#endif
