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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <vector>
#include "common.h"
#include "group_rectangle.h"
#include "lbp.h"

#define CL_FILE_PATH "lbp.cl"
#define CL_IMAGE_FILE_PATH "lbp_image.cl"

struct cl_stage {
    float stage_threshold;
    int num_weak_classifiers;
    int classifier_start_index;
};

struct lbp_cl {
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    int width;
    int height;
    int cl_image_support;
    cl_mem int_texture;

    std::vector<struct lbp_task> *full_tasks; // reference to full scan

    cl_mem input_rect;
    cl_mem input_classifier;
    cl_mem input_stage;
    cl_mem input_task;
    cl_mem input_subtask;
    cl_mem input_img;
    cl_mem output_result_counter;
    cl_mem output_result;

    int feature_width;
    int feature_height;
    struct lbp_para para;

    unsigned int *detected_task_index;
};

static char *
load_cl_file(size_t *len, const char *file)
{
    char *src;
    int fd = open(file, O_RDONLY);
    if (fd < 0) {
        ALOGE("Failed to open cl file");
        return NULL;
    }
    size_t size = lseek(fd, 0, SEEK_END);
    size_t read_size;
    lseek(fd, 0, SEEK_SET);
    src = (char *)malloc(size);
    read_size = read(fd, src, size);
    if (read_size != size) {
        ALOGE("Read cl file failed");
        free(src);
        close(fd);
        return NULL;
    }
    close(fd);
    *len = size;
    return src;
}

struct lbp_cl *
lbp_cl_init(struct lbp_data *data, struct lbp_para *para, 
    std::vector<struct lbp_task> *full_tasks, int width, int height)
{
    struct lbp_cl *cl;
    std::vector<struct weak_classifier> classifiers;
    std::vector<struct cl_stage> stages;
    int i, j;
    int err;

    cl = (struct lbp_cl *)calloc(1, sizeof(struct lbp_cl));

    cl->para = *para;
    cl->full_tasks = full_tasks;

    cl_platform_id platform0;
    err = clGetPlatformIDs(1, &platform0, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Failed to get cl platforms, %d", err);
        goto err1;
    }

    err = clGetDeviceIDs(platform0, CL_DEVICE_TYPE_CPU, 1, &cl->device_id, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Failed to create a device group, %d", err);
        goto err1;
    }

    cl_bool image_support;
    err = clGetDeviceInfo(cl->device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Failed to get device info %d", err);
        goto err1;
    }
    ALOGD("Device support image: %d", image_support);
    cl->cl_image_support  = image_support ? 1 : 0;
    /* REVISIST opencl does not support unsigned int 32 bits image to do linear interpolation, disable it for now, it is broken */
    cl->cl_image_support  = 0; 

    cl->context = clCreateContext(0, 1, &cl->device_id, NULL, NULL, &err);
    if (!cl->context) {
        ALOGE("Failed to create a compute context!");
        goto err1;
    }
 
    if (cl->cl_image_support) {
        cl_image_format format;
        format.image_channel_data_type = CL_UNSIGNED_INT32;
        format.image_channel_order     = CL_R;
#if CL_VERSION_1_2
        cl_image_desc desc;
        desc.image_type       = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width      = width;
        desc.image_height     = height;
        desc.image_depth      = 0;
        desc.image_array_size = 1;
        desc.image_row_pitch  = 0;
        desc.image_slice_pitch = 0;
        desc.buffer           = NULL;
        desc.num_mip_levels   = 0;
        desc.num_samples      = 0;
        cl->int_texture = clCreateImage(Context::getContext()->impl->clContext, CL_MEM_READ_ONLY, &format, &desc, NULL, &err);
#else
        cl->int_texture = clCreateImage2D(
                  cl->context,
                  CL_MEM_READ_ONLY,
                  &format,
                  width,
                  height,
                  0,
                  NULL,
                  &err);
#endif
        if (err != CL_SUCCESS) {
            ALOGW("Create image texture failed, fallback using non-image texture cl, err: %d", err);
            cl->cl_image_support = 0;
        }
    }
  
    cl->commands = clCreateCommandQueue(cl->context, cl->device_id, 0, &err);
    if (!cl->commands) {
        ALOGE("Failed to create a command commands!");
        goto err2;
    }

    size_t src_length;
    char *kernel_src;
    kernel_src  = load_cl_file(&src_length, cl->cl_image_support ? DATADIR"/"CL_IMAGE_FILE_PATH : DATADIR"/"CL_FILE_PATH);
    if (!kernel_src) {
        ALOGE("Failed to load cl!");
        goto err3;
    }
 
    cl->program = clCreateProgramWithSource(cl->context, 1, (const char **) &kernel_src, &src_length, &err);
    free(kernel_src);
    if (!cl->program) {
        ALOGE("Failed to create compute program!");
        goto err3;
    }

    char build_options[64];
    sprintf(build_options, "-DNUM_STAGES=%u -DWIDTH=%u", data->num_stages, width); 
 
    err = clBuildProgram(cl->program, 0, NULL, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[8 * 1024];
 
        ALOGE("Failed to build program executable!");
        clGetProgramBuildInfo(cl->program, cl->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        ALOGE("%s", buffer);
        goto err4;
    }
 
    cl->kernel = clCreateKernel(cl->program, "lbp", &err);
    if (!cl->kernel || err != CL_SUCCESS) {
        ALOGE("Failed to create compute kernel!");
        goto err4;
    }
 
    /* flatten the stages pass in */
    for (i = 0; i < data->num_stages; i++) {
        struct cl_stage s;
        s.stage_threshold = data->s[i].stage_threshold;
        s.num_weak_classifiers = data->s[i].num_weak_classifiers;
        if (i > 0) {
            s.classifier_start_index = stages[i - 1].classifier_start_index;
            s.classifier_start_index += stages[i - 1].num_weak_classifiers;
        } else {
            s.classifier_start_index = 0;
        }
        for (j = 0; j < data->s[i].num_weak_classifiers; j++) {
            struct weak_classifier c;
            c = data->s[i].classifiers[j];
            classifiers.push_back(c);
        }
        stages.push_back(s);
    }
    
    cl->input_rect = clCreateBuffer(cl->context,  CL_MEM_READ_ONLY,  sizeof(struct lbp_rect) * data->num_rects, NULL, NULL);
    cl->input_classifier = clCreateBuffer(cl->context,  CL_MEM_READ_ONLY,  sizeof(struct weak_classifier) * classifiers.size(), NULL, NULL);
    cl->input_stage = clCreateBuffer(cl->context,  CL_MEM_READ_ONLY,  sizeof(struct cl_stage) * data->num_stages, NULL, NULL);
    cl->input_task = clCreateBuffer(cl->context,  CL_MEM_READ_ONLY,  sizeof(struct lbp_task) * cl->full_tasks->size(), NULL, NULL);
    cl->input_subtask = clCreateBuffer(cl->context,  CL_MEM_READ_ONLY,  sizeof(struct lbp_task) * cl->full_tasks->size(), NULL, NULL);
    cl->input_img = clCreateBuffer(cl->context,  CL_MEM_READ_ONLY,  sizeof(unsigned int) * width * height, NULL, NULL);
    cl->output_result_counter = clCreateBuffer(cl->context,  CL_MEM_WRITE_ONLY,  sizeof(unsigned int), NULL, NULL);
    cl->output_result = clCreateBuffer(cl->context,  CL_MEM_WRITE_ONLY,  sizeof(unsigned int) * cl->full_tasks->size(), NULL, NULL);

    if (!cl->input_rect || !cl->input_classifier ||
        !cl->input_stage || !cl->input_task || !cl->input_subtask ||
        !cl->input_img || !cl->output_result_counter ||
        !cl->output_result) {
        ALOGE("Failed to allocate device memory!");
        goto err6;
    }    
    err = clEnqueueWriteBuffer(cl->commands, cl->input_rect, CL_TRUE, 0, sizeof(struct lbp_rect) * data->num_rects, data->r, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Failed to write to source array!");
        goto err6;
    }
    err = clEnqueueWriteBuffer(cl->commands, cl->input_classifier, CL_TRUE, 0, sizeof(struct weak_classifier) * classifiers.size(), classifiers.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Failed to write to source array!");
        goto err6;
    }
    err = clEnqueueWriteBuffer(cl->commands, cl->input_stage, CL_TRUE, 0, sizeof(struct cl_stage) * data->num_stages, stages.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Failed to write to source array!");
        goto err6;
    }
    err = clEnqueueWriteBuffer(cl->commands, cl->input_task, CL_TRUE, 0, sizeof(struct lbp_task) * cl->full_tasks->size(), cl->full_tasks->data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Failed to write to source array!");
        goto err6;
    }
    err = clSetKernelArg(cl->kernel, 0, sizeof(cl_mem), &cl->input_rect);
    err |= clSetKernelArg(cl->kernel, 1, sizeof(cl_mem), &cl->input_classifier);
    err |= clSetKernelArg(cl->kernel, 2, sizeof(cl_mem), &cl->input_stage);
    err |= clSetKernelArg(cl->kernel, 3, sizeof(cl_mem), &cl->input_task);
    err |= clSetKernelArg(cl->kernel, 4, sizeof(cl_mem), &cl->output_result_counter);
    err |= clSetKernelArg(cl->kernel, 5, sizeof(cl_mem), &cl->output_result);
    if (cl->cl_image_support) {
        err |= clSetKernelArg(cl->kernel, 6, sizeof(cl_mem), &cl->int_texture);
    } else {
        err |= clSetKernelArg(cl->kernel, 6, sizeof(cl_mem), &cl->input_img);
    }
    if (err != CL_SUCCESS) {
        ALOGE("Error: Failed to set kernel arguments! %d", err);
        goto err6;
    }

    cl->width = width;
    cl->height = height;
    cl->feature_width = data->feature_width;
    cl->feature_height = data->feature_height;

    cl->detected_task_index = (unsigned int *)malloc(sizeof(unsigned int) * cl->full_tasks->size());
 
    return cl;
 
err6:
    if (cl->input_rect)
        clReleaseMemObject(cl->input_rect);
    if (cl->input_classifier)
        clReleaseMemObject(cl->input_classifier);
    if (cl->input_stage)
        clReleaseMemObject(cl->input_stage);
    if (cl->input_subtask)
        clReleaseMemObject(cl->input_subtask);
    if (cl->input_task)
        clReleaseMemObject(cl->input_task);
    if (cl->input_img)
        clReleaseMemObject(cl->input_img);
    if (cl->output_result_counter)
        clReleaseMemObject(cl->output_result_counter);
    if (cl->output_result)
        clReleaseMemObject(cl->output_result);
err5:
    clReleaseKernel(cl->kernel);
err4:
    clReleaseProgram(cl->program);
err3:
    clReleaseCommandQueue(cl->commands);
err2:
    if (cl->int_texture)
        clReleaseMemObject(cl->int_texture);
    clReleaseContext(cl->context);
err1:
    free(cl);
 
    return NULL;
}
 
static int
cl_detect(struct lbp_cl *cl, unsigned int *img, 
    std::vector<struct lbp_task> *tasks,
    std::vector<struct lbp_rect>& rects)
{
    int err;
    size_t global;
    std::vector<struct lbp_task> *cl_tasks;

    err = clEnqueueWriteBuffer(cl->commands, cl->input_img, CL_TRUE, 0, sizeof(unsigned int) * cl->width * cl->height, img, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Failed to write to source image!");
        return -1;
    }

    // select to run a full run, or just scan previous results
    if ((tasks != NULL) && (tasks->size() < cl->full_tasks->size())) {
        if (tasks->size() == 0) {
            ALOGE("No task ?!");
            return -1;
        }
        global  = tasks->size();
        cl_tasks = tasks;
        err = clEnqueueWriteBuffer(cl->commands, cl->input_subtask, CL_TRUE, 0, sizeof(struct lbp_task) * tasks->size(), tasks->data(), 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            ALOGE("Failed to write to subtask %d", err);
            return -1;
        }
        err = clSetKernelArg(cl->kernel, 3, sizeof(cl_mem), &cl->input_subtask);
        if (err != CL_SUCCESS) {
            ALOGE("Failed to set input sub task!");
            return -1;
        }
    } else {
        global  = cl->full_tasks->size();
        cl_tasks = cl->full_tasks;
        err = clSetKernelArg(cl->kernel, 3, sizeof(cl_mem), &cl->input_task);
        if (err != CL_SUCCESS) {
            ALOGE("Failed to set input task!");
            return -1;
        }
    }

    // clear the result count
    unsigned int result_count = 0;
    err = clEnqueueWriteBuffer(cl->commands, cl->output_result_counter, CL_TRUE, 0, sizeof(unsigned int), &result_count, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Failed to write to result count!");
        return -1;
    }
 
    if (cl->cl_image_support) {
        const size_t origin[3] = {0, 0, 0};
        const size_t region[3] = {cl->width, cl->height, 1};
        err = clEnqueueWriteImage(cl->commands, cl->int_texture, CL_TRUE, origin, region, cl->width * sizeof(unsigned int), 0, img, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            ALOGE("Failed to write to int image texture !");
            return -1;
        }
    }
 
    err = clEnqueueNDRangeKernel(cl->commands, cl->kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ALOGE("Error: Failed to execute kernel %d", err);
        return -1;
    }
 
    clFinish(cl->commands);

    unsigned int detected_rects;
 
    err = clEnqueueReadBuffer(cl->commands, cl->output_result_counter, CL_TRUE, 0, sizeof(unsigned int), &detected_rects, 0, NULL, NULL );  
    if (err != CL_SUCCESS) {
        ALOGE("Error: Failed to read output array! %d", err);
        return -1;
    }

    if (detected_rects == 0) {
        return 0;
    }

    err = clEnqueueReadBuffer(cl->commands, cl->output_result, CL_TRUE, 0, sizeof(unsigned int) * detected_rects, cl->detected_task_index, 0, NULL, NULL );  
    if (err != CL_SUCCESS) {
        ALOGE("Error: Failed to read output array! %d", err);
        return -1;
    }

    /* walk throught the indes */
    for (int i = 0; i < detected_rects; i++) {
        struct lbp_rect r;
        r.x = (*cl_tasks)[cl->detected_task_index[i]].x;
        r.y = (*cl_tasks)[cl->detected_task_index[i]].y;
        r.w = cl->feature_width * (*cl_tasks)[cl->detected_task_index[i]].scale;
        r.h = cl->feature_height * (*cl_tasks)[cl->detected_task_index[i]].scale;
        rects.push_back(r);
    }
    face_detector_group_rectangle(rects, cl->para.group_threshold, cl->para.eps);
    
    return 0;
}
 
/* we only do a subset of scan based on previous located features */
int
lbp_cl_tracking(struct lbp_cl *cl, unsigned int *img, 
    std::vector<struct lbp_task> *tasks,
    std::vector<struct lbp_rect>& rects)
{
    return cl_detect(cl, img, tasks, rects);
}

int
lbp_cl_detect(struct lbp_cl *cl, unsigned int *img, std::vector<struct lbp_rect>& rects)
{
    return cl_detect(cl, img, NULL, rects);
}

void
lbp_cl_destroy(struct lbp_cl *cl)
{
    if (cl->int_texture)
        clReleaseMemObject(cl->int_texture);
    clReleaseMemObject(cl->input_rect);
    clReleaseMemObject(cl->input_classifier);
    clReleaseMemObject(cl->input_stage);
    clReleaseMemObject(cl->input_subtask);
    clReleaseMemObject(cl->input_task);
    clReleaseMemObject(cl->input_img);
    clReleaseMemObject(cl->output_result_counter);
    clReleaseMemObject(cl->output_result);
    clReleaseKernel(cl->kernel);
    clReleaseProgram(cl->program);
    clReleaseCommandQueue(cl->commands);
    clReleaseContext(cl->context);
    free(cl->detected_task_index);
    free(cl);
}
