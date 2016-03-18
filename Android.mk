LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_CPP_EXTENSION := .cc

# Set up the target identity
LOCAL_MODULE := libfaceldp

LOCAL_SRC_FILES := \
  src/face_detect.cc \
  src/group_rectangle.cc \
  src/integral_image.cc \
  src/lbp_detect.cc

LOCAL_SRC_FILES_x86 := \
  src/lbp_detect_sse2.cc

# x86_64 uses the same source as x86.
LOCAL_SRC_FILES_x86_64 := $(LOCAL_SRC_FILES_x86)

LOCAL_CFLAGS += \
  -DDATADIR="\"/system/etc/faceldp\"" \
  -Wno-endif-labels \
  -Wno-import \
  -Wno-format \

#LOCAL_C_INCLUDES += $(LOCAL_PATH) external/jpeg

LOCAL_LDLIBS := -llog

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_CPP_EXTENSION := .cc
LOCAL_SRC_FILES := \
  examples/facelbp_test.cc
LOCAL_MODULE:= facelbp_test
LOCAL_MODULE_TAGS := eng
LOCAL_SHARED_LIBRARIES := libfaceldp
LOCAL_C_INCLUDES += $(LOCAL_PATH)/src
LOCAL_LDLIBS := -llog
include $(BUILD_EXECUTABLE)
