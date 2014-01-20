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

#include "integral_image.h"
#include "common.h"

#if 0
static double
get_realtime() {
   struct timeval tv;
   gettimeofday(&tv,0);
   return (double)tv.tv_sec+1.0e-6*(double)tv.tv_usec;
}
#endif

void
face_detector_gen_integral_image(unsigned int *i_data, unsigned char *data, int width, int height)
{
    // first row only
    int i, j;
    unsigned int rs = 0;
    for (j = 0; j < width; j++) {
        rs += data[j]; 
        i_data[j] = rs;
    }
    // remaining cells are sum above and to the left
    for (i = 1; i < height; ++i) {
        rs = 0;
        for (j = 0; j < width; ++j) {
            rs += data[i * width + j]; 
            i_data[i * width + j] = rs + i_data[(i - 1) * width + j];
        }
    }
}
