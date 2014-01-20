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

#ifndef _FACE_DETECT_H
#define _FACE_DETECT_H

#include "face_object.h"

struct face_det;

int face_detector_detect(struct face_det *f, unsigned char *y, struct face *fa, int *maxfaces);
int face_detector_tracking(struct face_det *f, unsigned char *y, struct face *fa, int faces, int *maxfaces);
struct face_det *face_detector_create(int width, int height, int minimum_face_width);
void face_detector_destroy(struct face_det *f);

#endif
