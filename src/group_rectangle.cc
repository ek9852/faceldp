// This file is originally based on OpenCV, license as belows

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "group_rectangle.h"

static inline int
myMax(int a, int b)
{
    return (a >= b) ? a : b;
}

static inline int
myMin(int a, int b)
{
    return (a <= b) ? a : b;
}

static inline int
myRound(float value)
{
    return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

static inline int
myAbs(int n)
{
    return (n >= 0) ? n : -n;
}

static int
predicate(float eps, struct lbp_rect& r1, struct lbp_rect& r2)
{
    float delta = eps*(myMin(r1.w, r2.w) + myMin(r1.h, r2.h))*0.5;
    return myAbs(r1.x - r2.x) <= delta &&
        myAbs(r1.y - r2.y) <= delta &&
        myAbs(r1.x + r1.w - r2.x - r2.w) <= delta &&
        myAbs(r1.y + r1.h - r2.y - r2.h) <= delta;
}

static int
partition(std::vector<struct lbp_rect>& _vec, std::vector<int>& labels, float eps)
{
    int i, j, N = (int)_vec.size();

    struct lbp_rect* vec = &_vec[0];

    const int PARENT=0;
    const int RANK=1;

    std::vector<int> _nodes(N*2);

    int (*nodes)[2] = (int(*)[2])&_nodes[0];

    /* The first O(N) pass: create N single-vertex trees */
    for (i = 0; i < N; i++) {
        nodes[i][PARENT]=-1;
        nodes[i][RANK] = 0;
    }

    /* The main O(N^2) pass: merge connected components */
    for (i = 0; i < N; i++) {
        int root = i;

        /* find root */
        while (nodes[root][PARENT] >= 0)
            root = nodes[root][PARENT];

        for (j = 0; j < N; j++ ) {
            if( i == j || !predicate(eps, vec[i], vec[j]))
                continue;
            int root2 = j;

            while (nodes[root2][PARENT] >= 0)
                root2 = nodes[root2][PARENT];

            if (root2 != root) {
                /* unite both trees */
                int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
                if (rank > rank2)
                    nodes[root2][PARENT] = root;
                else {
                    nodes[root][PARENT] = root2;
                    nodes[root2][RANK] += rank == rank2;
                    root = root2;
                }

                int k = j, parent;

                /* compress the path from node2 to root */
                while ((parent = nodes[k][PARENT]) >= 0) {
                    nodes[k][PARENT] = root;
                    k = parent;
                }

                /* compress the path from node to root */
                k = i;
                while ((parent = nodes[k][PARENT]) >= 0) {
                    nodes[k][PARENT] = root;
                    k = parent;
                }
            }
        }
    }

    /* Final O(N) pass: enumerate classes */
    labels.resize(N);
    int nclasses = 0;

    for (i = 0; i < N; i++) {
        int root = i;
        while (nodes[root][PARENT] >= 0)
            root = nodes[root][PARENT];
        /* re-use the rank as the class label */
        if (nodes[root][RANK] >= 0)
            nodes[root][RANK] = ~nclasses++;
        labels[i] = ~nodes[root][RANK];
    }

    return nclasses;
}

void
face_detector_group_rectangle(std::vector<struct lbp_rect>& rect_list, int group_threshold, float eps)
{
    if (group_threshold <= 0 || rect_list.empty())
        return;

    std::vector<int> labels;

    int nclasses = partition(rect_list, labels, eps);

    std::vector<struct lbp_rect> rrects(nclasses);
    std::vector<int> rweights(nclasses);

    int i, j, nlabels = (int)labels.size();


    for (i = 0; i < nlabels; i++) {
        int cls = labels[i];
        rrects[cls].x += rect_list[i].x;
        rrects[cls].y += rect_list[i].y;
        rrects[cls].w += rect_list[i].w;
        rrects[cls].h += rect_list[i].h;
        rweights[cls]++;
    }
    for (i = 0; i < nclasses; i++) {
        struct lbp_rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i].x = myRound(r.x*s);
        rrects[i].y = myRound(r.y*s);
        rrects[i].w = myRound(r.w*s);
        rrects[i].h = myRound(r.h*s);

    }

    rect_list.clear();

    for (i = 0; i < nclasses; i++) {
        struct lbp_rect r1 = rrects[i];
        int n1 = rweights[i];
        if( n1 <= group_threshold )
            continue;
        /* filter out small face rectangles inside large rectangles */
        for (j = 0; j < nclasses; j++) {
            int n2 = rweights[j];
            /*********************************
             * if it is the same rectangle, 
             * or the number of rectangles in class j is < group threshold, 
             * do nothing 
             ********************************/
            if (j == i || n2 <= group_threshold)
                continue;
            struct lbp_rect r2 = rrects[j];

            int dx = myRound( r2.w * eps );
            int dy = myRound( r2.h * eps );

            if (i != j &&
                    r1.x >= r2.x - dx &&
                    r1.y >= r2.y - dy &&
                    r1.x + r1.w <= r2.x + r2.w + dx &&
                    r1.y + r1.h <= r2.y + r2.h + dy &&
                    (n2 > myMax(3, n1) || n1 < 3))
                break;
        }

        if (j == nclasses) {
            rect_list.push_back(r1); // insert back r1
        }
    }
}
