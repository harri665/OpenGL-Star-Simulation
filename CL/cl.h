#ifndef __OPENCL_CL_H
#define __OPENCL_CL_H

#include "cl_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error Codes */
#define CL_SUCCESS                                  0
#define CL_DEVICE_NOT_FOUND                        -1
#define CL_DEVICE_NOT_AVAILABLE                    -2
#define CL_BUILD_PROGRAM_FAILURE                  -11

/* Device Types */
#define CL_DEVICE_TYPE_DEFAULT        (1 << 0)
#define CL_DEVICE_TYPE_CPU            (1 << 1)
#define CL_DEVICE_TYPE_GPU            (1 << 2)
#define CL_DEVICE_TYPE_ACCELERATOR    (1 << 3)

/* Memory Flags */
#define CL_MEM_READ_WRITE    (1 << 0)
#define CL_MEM_WRITE_ONLY    (1 << 1)
#define CL_MEM_READ_ONLY     (1 << 2)
#define CL_MEM_USE_HOST_PTR  (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)
#define CL_MEM_COPY_HOST_PTR (1 << 5)

/* Command Queue Properties */
#define CL_QUEUE_PROFILING_ENABLE    (1 << 0)

/* Program Build Info */
#define CL_PROGRAM_BUILD_LOG         0x1183

/* Boolean */
#define CL_FALSE 0
#define CL_TRUE  1

/* Function declarations (subset used) */
cl_int    clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
cl_int    clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
                        cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);

cl_context clCreateContext(const cl_context_properties *properties, cl_uint num_devices,
                          const cl_device_id *devices, void (*pfn_notify)(const char*, const void*, size_t, void*),
                          void *user_data, cl_int *errcode_ret);

cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device,
                                     cl_command_queue_properties properties, cl_int *errcode_ret);

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings,
                                    const size_t *lengths, cl_int *errcode_ret);

cl_int    clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list,
                        const char *options, void (*pfn_notify)(cl_program, void *), void *user_data);

cl_int    clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                               size_t param_value_size, void *param_value, size_t *param_value_size_ret);

cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret);

cl_mem    clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);

cl_int    clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
                              size_t offset, size_t cb, const void *ptr,
                              cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

cl_int    clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
                             size_t offset, size_t cb, void *ptr,
                             cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

cl_int    clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                                const size_t *global_work_offset, const size_t *global_work_size,
                                const size_t *local_work_size, cl_uint num_events_in_wait_list,
                                const cl_event *event_wait_list, cl_event *event);

cl_int    clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);

cl_int    clFinish(cl_command_queue command_queue);

cl_int    clReleaseMemObject(cl_mem memobj);
cl_int    clReleaseKernel(cl_kernel kernel);
cl_int    clReleaseProgram(cl_program program);
cl_int    clReleaseCommandQueue(cl_command_queue command_queue);
cl_int    clReleaseContext(cl_context context);

#ifdef __cplusplus
}
#endif

#endif /* __OPENCL_CL_H */
