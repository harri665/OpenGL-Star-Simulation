#ifndef __OPENCL_CL_PLATFORM_H
#define __OPENCL_CL_PLATFORM_H

#if defined(_WIN32)
#include <stddef.h>
#include <stdint.h>
typedef unsigned int    cl_uint;
typedef int             cl_int;
typedef unsigned short  cl_ushort;
typedef unsigned long   cl_ulong;
typedef unsigned long long cl_ulonglong;
typedef signed long long   cl_longlong;
typedef unsigned char   cl_uchar;
typedef unsigned char   cl_bool;
typedef intptr_t        cl_intptr_t;
typedef uintptr_t       cl_uintptr_t;
typedef intptr_t        cl_context_properties;
typedef unsigned long   cl_bitfield;
#else
#include <stddef.h>
#include <stdint.h>
typedef unsigned int    cl_uint;
typedef int             cl_int;
typedef unsigned short  cl_ushort;
typedef unsigned long   cl_ulong;
typedef unsigned long long cl_ulonglong;
typedef signed long long   cl_longlong;
typedef unsigned char   cl_uchar;
typedef unsigned char   cl_bool;
typedef intptr_t        cl_intptr_t;
typedef uintptr_t       cl_uintptr_t;
typedef intptr_t        cl_context_properties;
typedef unsigned long   cl_bitfield;
#endif

typedef cl_uint   cl_device_type;
typedef cl_uint   cl_platform_info;
typedef cl_uint   cl_device_info;
typedef cl_uint   cl_mem_flags;
typedef cl_uint   cl_command_queue_properties;
typedef cl_uint   cl_program_build_info;

typedef struct _cl_platform_id   *cl_platform_id;
typedef struct _cl_device_id     *cl_device_id;
typedef struct _cl_context       *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem           *cl_mem;
typedef struct _cl_program       *cl_program;
typedef struct _cl_kernel        *cl_kernel;
typedef struct _cl_event         *cl_event;
typedef struct _cl_sampler       *cl_sampler;

#endif /* __OPENCL_CL_PLATFORM_H */
