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

#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include "face_detect.h"

int
main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <width> <height> <image.y>\n", argv[0]);
        return 1;
    }

    int width, height;
    width = atoi(argv[1]);
    height = atoi(argv[2]);

    if ((width <= 0) || (height <= 0)) {
        fprintf(stderr, "width/height invalid\n");
        return 1;
    }

    struct face_det *det;
    det = face_detector_create(width, height, 24);
    if (!det) {
        fprintf(stderr, "init face detector error\n");
        return 1;
    }

    unsigned char *y;
    ssize_t s;
    int fd;

    y = (unsigned char *)malloc(width * height);
    fd = open(argv[3], O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open file: %s\n", argv[3]);
        return 1;
    }

    s = read(fd, y, width * height);
    if (s != width * height) {
        fprintf(stderr, "Error reading file: %s\n", argv[3]);
        return 1;
    }

    struct face f[30];
    int num_faces = 30;
    int i;
    
    face_detector_detect(det, y, f, &num_faces);

    for (i = 0 ;i < num_faces; i++) {
        printf("Face: %d %d %d %d\n", f[i].x, f[i].y, f[i].width, f[i].height);
    }
#if 0
    face_detector_tracking(det, y, f, num_faces, &num_faces);
    for (i = 0 ;i < num_faces; i++) {
        printf("Face: %d %d %d %d\n", f[i].x, f[i].y, f[i].width, f[i].height);
    }
#endif
    printf("Total: %d\n", num_faces);

    face_detector_destroy(det);
}
