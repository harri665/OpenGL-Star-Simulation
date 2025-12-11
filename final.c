/*
 * Real-time Star Magnetic Field Particle Simulation
 * OpenGL / GLUT / C + OpenCL GPU particle integration
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

/* OpenCL */
#ifdef _WIN32
#include "CL/cl.h"
#include <stddef.h> /* for ptrdiff_t */
#else
#include "CL/cl.h"
#endif

/* --- OpenGL Extension Definitions for Windows (No GLEW) --- */
#ifdef _WIN32
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84

typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;

typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC) (GLenum type);
typedef void (APIENTRY *PFNGLSHADERSOURCEPROC) (GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
typedef void (APIENTRY *PFNGLCOMPILESHADERPROC) (GLuint shader);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC) (void);
typedef void (APIENTRY *PFNGLATTACHSHADERPROC) (GLuint program, GLuint shader);
typedef void (APIENTRY *PFNGLLINKPROGRAMPROC) (GLuint program);
typedef void (APIENTRY *PFNGLUSEPROGRAMPROC) (GLuint program);
typedef void (APIENTRY *PFNGLGETSHADERIVPROC) (GLuint shader, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETPROGRAMIVPROC) (GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETSHADERINFOLOGPROC) (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (APIENTRY *PFNGLGETPROGRAMINFOLOGPROC) (GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef GLint (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC) (GLuint program, const GLchar *name);
typedef void (APIENTRY *PFNGLUNIFORM1FPROC) (GLint location, GLfloat v0);

PFNGLCREATESHADERPROC glCreateShader = NULL;
PFNGLSHADERSOURCEPROC glShaderSource = NULL;
PFNGLCOMPILESHADERPROC glCompileShader = NULL;
PFNGLCREATEPROGRAMPROC glCreateProgram = NULL;
PFNGLATTACHSHADERPROC glAttachShader = NULL;
PFNGLLINKPROGRAMPROC glLinkProgram = NULL;
PFNGLUSEPROGRAMPROC glUseProgram = NULL;
PFNGLGETSHADERIVPROC glGetShaderiv = NULL;
PFNGLGETPROGRAMIVPROC glGetProgramiv = NULL;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = NULL;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = NULL;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = NULL;
PFNGLUNIFORM1FPROC glUniform1f = NULL;

int g_shaders_supported = 0;
GLuint g_sun_program = 0;
GLint g_sun_time_loc = -1;

void load_shader_extensions(void) {
    glCreateShader = (PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram");
    glUseProgram = (PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)wglGetProcAddress("glGetProgramiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)wglGetProcAddress("glGetShaderInfoLog");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)wglGetProcAddress("glGetProgramInfoLog");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation");
    glUniform1f = (PFNGLUNIFORM1FPROC)wglGetProcAddress("glUniform1f");

    if (glCreateShader && glUseProgram) {
        g_shaders_supported = 1;
        printf("Shader extensions loaded successfully.\n");
    } else {
        printf("Failed to load shader extensions.\n");
    }
}
#endif

/* Window dimensions */
int g_window_width  = 1280;
int g_window_height = 720;

/* Camera parameters (spherical coordinates) */
float g_cam_radius    = 5.0f;
float g_cam_azimuth   = 45.0f;   /* degrees */
float g_cam_elevation = 30.0f;   /* degrees */

/* Mouse interaction state */
int g_mouse_dragging = 0;
int g_mouse_last_x = 0;
int g_mouse_last_y = 0;

/* Star parameters (scale: 1.0 unit = 1 solar radius) */
const float k_unit_length_m = 6.957e8f; /* meters per visual unit (~R_sun) */
float g_star_radius = 2.5f;              /* Base radius */
float g_star_scale = 1.0f;              /* Scale factor for star and spawn points */

/* Simulation state */
int g_paused = 0;

/* Field parameters */
float g_field_strength = 1.0f;
float g_dipole_tilt_deg = 0.0f;

/* Debug visualization */
int g_show_field_lines = 0;
int g_show_vectors = 0;
int g_show_trails = 0;
int g_trail_from_birth = 0;
float g_trail_length_scale = 1.0f;

/* Particle parameters */
#define MAX_PARTICLES 2000000

float g_pos[MAX_PARTICLES][4];
float g_vel[MAX_PARTICLES][4];
float g_birth_pos[MAX_PARTICLES][4];
float g_age_arr[MAX_PARTICLES];
float g_life_arr[MAX_PARTICLES];
int   g_alive_arr[MAX_PARTICLES];
int g_max_active_particles = 5000;
float g_emission_rate = 10000.0f;
float g_emission_accumulator = 0.0f;

/* Simulation parameters */
float g_lorentz_k = 2.0f;          /* Lorentz force coefficient */
float g_simulation_dt = 0.01f;     /* Fixed time step */
float g_time_accumulator = 0.0f;
int g_last_sim_time = 0;
float g_grid_scale = 1.0f;         /* Scale factor for grid coordinates */

/* OpenCL objects */
cl_platform_id g_cl_platform = NULL;
cl_device_id   g_cl_device   = NULL;
cl_context     g_cl_context  = NULL;
cl_command_queue g_cl_queue  = NULL;
cl_program     g_cl_program  = NULL;
cl_kernel      g_cl_kernel   = NULL;

cl_mem g_cl_pos   = NULL;
cl_mem g_cl_vel   = NULL;
cl_mem g_cl_age   = NULL;
cl_mem g_cl_life  = NULL;
cl_mem g_cl_alive = NULL;

/* Grid Data */
#pragma pack(push, 1)
typedef struct {
    char magic[4];      // "GRID"
    int32_t version;    // 1
    int32_t dim_x;
    int32_t dim_y;
    int32_t dim_z;
    float voxel_size;
    float origin_x;
    float origin_y;
    float origin_z;
} GridHeader;
#pragma pack(pop)

cl_mem g_cl_grid_data = NULL;
float *g_grid_data = NULL;
size_t g_grid_data_capacity = 0;
GridHeader g_grid_header;
int g_grid_loaded = 0;
int g_current_frame = 1;
int g_total_frames = 1000;
int g_use_grid = 0;
char g_current_filename[256] = "None";


/* Frame timing */
int g_frame_count = 0;
float g_fps = 0.0f;
int g_last_fps_time = 0;

/*
 * Convert degrees to radians
 */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Forward declarations */
void init_particles(void);
void spawn_particle(int index);
void update_particles(float dt);
void draw_particles(void);
void draw_field_lines(void);
void magnetic_field(const float p[3], float B[3]);
int  init_opencl(void);
void release_opencl(void);

static inline float deg2rad(float deg) {
    return deg * (float)M_PI / 180.0f;
}

/* OpenCL kernel source */
const char *g_cl_kernel_src =
"__kernel void integrate(__global float4* pos, __global float4* vel, __global float* age, __global float* life, __global int* alive, \n"
"                      const float dt, const float k, const float field_strength, const float tilt_deg, \n"
"                      __global float* grid_data, const int use_grid, \n"
"                      const int4 grid_dim, const float4 grid_origin, const float grid_voxel_size, const float grid_scale) {\n"
"    int i = get_global_id(0);\n"
"    if (!alive[i]) return;\n"
"    float4 p = pos[i];\n"
"    float3 v = (float3)(vel[i].x, vel[i].y, vel[i].z);\n"
"    \n"
"    if (use_grid) {\n"
"        // Scale position to grid space\n"
"        float3 scaled_p = (float3)(p.x, p.y, p.z) / grid_scale;\n"
"        float3 local_p = scaled_p - (float3)(grid_origin.x, grid_origin.y, grid_origin.z);\n"
"        int x = (int)(local_p.x / grid_voxel_size);\n"
"        int y = (int)(local_p.y / grid_voxel_size);\n"
"        int z = (int)(local_p.z / grid_voxel_size);\n"
"        \n"
"        if (x >= 0 && x < grid_dim.x && y >= 0 && y < grid_dim.y && z >= 0 && z < grid_dim.z) {\n"
"            int idx = (z * (grid_dim.x * grid_dim.y) + y * grid_dim.x + x) * 3;\n"
"            v = (float3)(grid_data[idx], grid_data[idx+1], grid_data[idx+2]);\n"
"        }\n"
"        // Advect\n"
"        p.xyz += v * dt;\n"
"    } else {\n"
"        // Magnetic field logic\n"
"        float r = length((float3)(p.x, p.y, p.z));\n"
"        if (r < 0.01f) { alive[i] = 0; return; }\n"
"        float3 r_hat = (float3)(p.x, p.y, p.z) / r;\n"
"        float tilt_rad = tilt_deg * 0.01745329252f;\n"
"        float cos_t = cos(tilt_rad);\n"
"        float sin_t = sin(tilt_rad);\n"
"        float3 m = (float3)(0.0f, sin_t, cos_t);\n"
"        float m_dot_r = dot(m, r_hat);\n"
"        float r3 = r * r * r;\n"
"        float3 B = (3.0f * m_dot_r * r_hat - m) / r3;\n"
"        B *= field_strength;\n"
"        float3 acc = k * cross(v, B);\n"
"        v += acc * dt;\n"
"        p.xyz += v * dt;\n"
"    }\n"
"    \n"
"    float new_age = age[i] + dt;\n"
"    if (new_age > life[i]) { alive[i] = 0; age[i] = new_age; pos[i] = p; vel[i].xyz = v; return; }\n"
"    if (length(p.xyz) > 20.0f) { alive[i] = 0; age[i] = new_age; pos[i] = p; vel[i].xyz = v; return; }\n"
"    age[i] = new_age;\n"
"    pos[i] = p;\n"
"    vel[i].xyz = v;\n"
"}\n";

/*
 * Vector math utilities
 */
static inline float vec3_length(const float v[3]) {
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static inline void vec3_normalize(float v[3]) {
    float len = vec3_length(v);
    if (len > 1e-8f) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

static inline float vec3_dot(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void vec3_cross(const float a[3], const float b[3], float result[3]) {
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
}

static inline void vec3_scale(float v[3], float s) {
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
}

static inline void vec3_add(float a[3], const float b[3]) {
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
}

/*
 * Random float in range [min, max]
 */
static inline float randf(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

/*
 * Random point on unit sphere (uniform distribution)
 */
void random_unit_sphere(float v[3]) {
    float theta = randf(0.0f, 2.0f * M_PI);
    float phi = acosf(randf(-1.0f, 1.0f));
    v[0] = sinf(phi) * cosf(theta);
    v[1] = sinf(phi) * sinf(theta);
    v[2] = cosf(phi);
}

/*
 * OpenCL helpers
 */
static void cl_check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error %d at %s\n", err, msg);
        exit(1);
    }
}

int init_opencl(void) {
    cl_int err;
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(1, &g_cl_platform, &num_platforms);
    cl_check(err, "clGetPlatformIDs");
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found\n");
        return 0;
    }

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(g_cl_platform, CL_DEVICE_TYPE_GPU, 1, &g_cl_device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        /* fallback to CPU */
        err = clGetDeviceIDs(g_cl_platform, CL_DEVICE_TYPE_CPU, 1, &g_cl_device, &num_devices);
        cl_check(err, "clGetDeviceIDs CPU fallback");
    }

    g_cl_context = clCreateContext(NULL, 1, &g_cl_device, NULL, NULL, &err);
    cl_check(err, "clCreateContext");

    g_cl_queue = clCreateCommandQueue(g_cl_context, g_cl_device, 0, &err);
    cl_check(err, "clCreateCommandQueue");

    const char *src = g_cl_kernel_src;
    size_t lengths = strlen(g_cl_kernel_src);
    g_cl_program = clCreateProgramWithSource(g_cl_context, 1, &src, &lengths, &err);
    cl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(g_cl_program, 1, &g_cl_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(g_cl_program, g_cl_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(g_cl_program, g_cl_device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "OpenCL build log:\n%s\n", log);
        free(log);
        cl_check(err, "clBuildProgram");
    }

    g_cl_kernel = clCreateKernel(g_cl_program, "integrate", &err);
    cl_check(err, "clCreateKernel");

    /* Buffers */
    g_cl_pos   = clCreateBuffer(g_cl_context, CL_MEM_READ_WRITE, sizeof(float)*4*MAX_PARTICLES, NULL, &err);
    cl_check(err, "clCreateBuffer pos");
    g_cl_vel   = clCreateBuffer(g_cl_context, CL_MEM_READ_WRITE, sizeof(float)*4*MAX_PARTICLES, NULL, &err);
    cl_check(err, "clCreateBuffer vel");
    g_cl_age   = clCreateBuffer(g_cl_context, CL_MEM_READ_WRITE, sizeof(float)*MAX_PARTICLES, NULL, &err);
    cl_check(err, "clCreateBuffer age");
    g_cl_life  = clCreateBuffer(g_cl_context, CL_MEM_READ_WRITE, sizeof(float)*MAX_PARTICLES, NULL, &err);
    cl_check(err, "clCreateBuffer life");
    g_cl_alive = clCreateBuffer(g_cl_context, CL_MEM_READ_WRITE, sizeof(int)*MAX_PARTICLES, NULL, &err);
    cl_check(err, "clCreateBuffer alive");

    /* Initialize to zero */
    float *zero_pos = (float*)calloc(MAX_PARTICLES*4, sizeof(float));
    int   *zero_int = (int*)calloc(MAX_PARTICLES, sizeof(int));
    float *zero_f   = (float*)calloc(MAX_PARTICLES, sizeof(float));
    clEnqueueWriteBuffer(g_cl_queue, g_cl_pos, CL_TRUE, 0, sizeof(float)*4*MAX_PARTICLES, zero_pos, 0, NULL, NULL);
    clEnqueueWriteBuffer(g_cl_queue, g_cl_vel, CL_TRUE, 0, sizeof(float)*4*MAX_PARTICLES, zero_pos, 0, NULL, NULL);
    clEnqueueWriteBuffer(g_cl_queue, g_cl_age, CL_TRUE, 0, sizeof(float)*MAX_PARTICLES, zero_f, 0, NULL, NULL);
    clEnqueueWriteBuffer(g_cl_queue, g_cl_life, CL_TRUE, 0, sizeof(float)*MAX_PARTICLES, zero_f, 0, NULL, NULL);
    clEnqueueWriteBuffer(g_cl_queue, g_cl_alive, CL_TRUE, 0, sizeof(int)*MAX_PARTICLES, zero_int, 0, NULL, NULL);
    free(zero_pos); free(zero_int); free(zero_f);

    return 1;
}

void release_opencl(void) {
    if (g_cl_kernel) clReleaseKernel(g_cl_kernel);
    if (g_cl_program) clReleaseProgram(g_cl_program);
    if (g_cl_pos) clReleaseMemObject(g_cl_pos);
    if (g_cl_vel) clReleaseMemObject(g_cl_vel);
    if (g_cl_age) clReleaseMemObject(g_cl_age);
    if (g_cl_life) clReleaseMemObject(g_cl_life);
    if (g_cl_alive) clReleaseMemObject(g_cl_alive);
    if (g_cl_grid_data) clReleaseMemObject(g_cl_grid_data);
    if (g_grid_data) free(g_grid_data);
    if (g_cl_queue) clReleaseCommandQueue(g_cl_queue);
    if (g_cl_context) clReleaseContext(g_cl_context);
}

/*
 * Compute the magnetic dipole field at position p
 * Dipole moment m is aligned with +Z axis and rotated by tilt angle around X axis
 * Field formula: B(p) = (1/r^3) * [3(m·r_hat)r_hat - m]
 */
void magnetic_field(const float p[3], float B[3]) {
    /* Calculate distance from origin */
    float r = vec3_length(p);
    
    /* Avoid singularity at origin */
    if (r < 0.01f) {
        B[0] = B[1] = B[2] = 0.0f;
        return;
    }
    
    /* Unit vector from origin to p */
    float r_hat[3] = {p[0]/r, p[1]/r, p[2]/r};
    
    /* Dipole moment vector (start aligned with +Z, then apply tilt) */
    /* Tilt is rotation around X axis */
    float tilt_rad = deg2rad(g_dipole_tilt_deg);
    float cos_tilt = cosf(tilt_rad);
    float sin_tilt = sinf(tilt_rad);
    
    /* m = (0, sin(tilt), cos(tilt)) after rotation around X */
    float m[3] = {0.0f, sin_tilt, cos_tilt};
    
    /* Compute m · r_hat */
    float m_dot_r = vec3_dot(m, r_hat);
    
    /* B = (1/r^3) * [3(m·r_hat)r_hat - m] */
    float r3 = r * r * r;
    B[0] = (3.0f * m_dot_r * r_hat[0] - m[0]) / r3;
    B[1] = (3.0f * m_dot_r * r_hat[1] - m[1]) / r3;
    B[2] = (3.0f * m_dot_r * r_hat[2] - m[2]) / r3;
    
    /* Scale by field strength */
    vec3_scale(B, g_field_strength);
}

/*
 * Draw debug visualization of magnetic field lines
 */
void draw_field_lines(void) {
    if (!g_show_field_lines) return;
    
    glDisable(GL_LIGHTING);
    glLineWidth(1.5f);
    
    /* Sample points on sphere surface and nearby */
    int num_samples = 12;
    float sample_radius = g_star_radius * g_star_scale * 1.1f;
    
    for (int i = 0; i < num_samples; i++) {
        float theta = (float)i / num_samples * 2.0f * M_PI;
        for (int j = 0; j < num_samples/2; j++) {
            float phi = ((float)j / (num_samples/2) - 0.5f) * M_PI;
            
            /* Sample position */
            float pos[3];
            pos[0] = sample_radius * cosf(phi) * cosf(theta);
            pos[1] = sample_radius * sinf(phi);
            pos[2] = sample_radius * cosf(phi) * sinf(theta);
            
            /* Get field at this position */
            float B[3];
            magnetic_field(pos, B);
            
            /* Normalize and scale for visualization */
            vec3_normalize(B);
            vec3_scale(B, 0.3f);
            
            /* Draw short line segment showing field direction */
            float field_mag = vec3_length(B);
            if (field_mag > 0.01f) {
                /* Color based on field direction (red = outward, blue = inward) */
                float r_dot_B = vec3_dot(pos, B);
                if (r_dot_B > 0) {
                    glColor3f(1.0f, 0.3f, 0.3f);  /* Red - outward */
                } else {
                    glColor3f(0.3f, 0.3f, 1.0f);  /* Blue - inward */
                }
                
                glBegin(GL_LINES);
                glVertex3f(pos[0], pos[1], pos[2]);
                glVertex3f(pos[0] + B[0], pos[1] + B[1], pos[2] + B[2]);
                glEnd();
            }
        }
    }
    
    glEnable(GL_LIGHTING);
}

/*
 * Draw velocity vectors from the grid
 */
void draw_velocity_vectors(void) {
    if (!g_show_vectors || !g_grid_data) return;

    glDisable(GL_LIGHTING);
    glLineWidth(1.0f);
    glBegin(GL_LINES);

    int stride = 2; // Skip every other voxel to reduce clutter
    float scale = 0.1f; // Scale for vector length visualization

    for (int z = 0; z < g_grid_header.dim_z; z += stride) {
        for (int y = 0; y < g_grid_header.dim_y; y += stride) {
            for (int x = 0; x < g_grid_header.dim_x; x += stride) {
                int idx = (z * (g_grid_header.dim_x * g_grid_header.dim_y) + y * g_grid_header.dim_x + x) * 3;
                
                float vx = g_grid_data[idx];
                float vy = g_grid_data[idx+1];
                float vz = g_grid_data[idx+2];
                
                float speed_sq = vx*vx + vy*vy + vz*vz;
                if (speed_sq < 0.01f) continue; // Don't draw tiny vectors

                float speed = sqrtf(speed_sq);
                float px = g_grid_header.origin_x + x * g_grid_header.voxel_size;
                float py = g_grid_header.origin_y + y * g_grid_header.voxel_size;
                float pz = g_grid_header.origin_z + z * g_grid_header.voxel_size;

                // Color based on speed (green -> red)
                float t = fminf(speed / 5.0f, 1.0f);
                glColor3f(t, 1.0f - t * 0.5f, 0.2f);

                glVertex3f(px, py, pz);
                glVertex3f(px + vx * scale, py + vy * scale, pz + vz * scale);
            }
        }
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

/*
 * Update camera position from spherical coordinates
 */
void compute_camera_position(float *eye_x, float *eye_y, float *eye_z) {
    float az_rad  = deg2rad(g_cam_azimuth);
    float el_rad  = deg2rad(g_cam_elevation);
    
    *eye_x = g_cam_radius * cosf(el_rad) * cosf(az_rad);
    *eye_y = g_cam_radius * sinf(el_rad);
    *eye_z = g_cam_radius * cosf(el_rad) * sinf(az_rad);
}

/*
 * Draw 2D text overlay (orthographic projection)
 */
void draw_text(float x, float y, const char *text) {
    glRasterPos2f(x, y);
    while (*text) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *text);
        text++;
    }
}

void draw_overlay(void) {
    /* Save current matrices and attributes */
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, g_window_width, 0, g_window_height);
    
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
    /* Disable lighting and depth test for 2D text */
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    
    /* White text */
    glColor3f(1.0f, 1.0f, 1.0f);
    
    char buffer[512];
    float line_height = 20.0f;
    float start_y = g_window_height - 25.0f;
    
    /* FPS */
    snprintf(buffer, sizeof(buffer), "FPS: %.1f", g_fps);
    draw_text(10, start_y, buffer);
    start_y -= line_height;
    
    /* Simulation state */
    snprintf(buffer, sizeof(buffer), "State: %s", g_paused ? "PAUSED" : "RUNNING");
    draw_text(10, start_y, buffer);
    start_y -= line_height;

    /* Current Grid File */
    snprintf(buffer, sizeof(buffer), "Grid File: %s", g_current_filename);
    draw_text(10, start_y, buffer);
    start_y -= line_height;
    
    /* Field parameters */
    snprintf(buffer, sizeof(buffer), "Field Strength: %.2f", g_field_strength);
    draw_text(10, start_y, buffer);
    start_y -= line_height;
    
    snprintf(buffer, sizeof(buffer), "Dipole Tilt: %.1f deg", g_dipole_tilt_deg);
    draw_text(10, start_y, buffer);
    start_y -= line_height;

    /* Scale info */
    draw_text(10, start_y, "Scale: 1 unit = 1 R_sun (~6.96e8 m)");
    start_y -= line_height;
    snprintf(buffer, sizeof(buffer), "Grid Scale: %.3f", g_grid_scale);
    draw_text(10, start_y, buffer);
    start_y -= line_height;
    snprintf(buffer, sizeof(buffer), "Star Radius: %.1f", g_star_radius);
    draw_text(10, start_y, buffer);
    start_y -= line_height;
    
    /* Particle parameters */
    snprintf(buffer, sizeof(buffer), "Max Particles: %d", g_max_active_particles);
    draw_text(10, start_y, buffer);
    start_y -= line_height;
    
    snprintf(buffer, sizeof(buffer), "Emission Rate: %.0f/sec", g_emission_rate);
    draw_text(10, start_y, buffer);
    start_y -= line_height;
    
    /* Active particle count */
    int active_count = 0;
    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (g_alive_arr[i]) active_count++;
    }
    snprintf(buffer, sizeof(buffer), "Active Particles: %d", active_count);
    draw_text(10, start_y, buffer);
    start_y -= line_height * 1.5f;

    /* Trail status */
    snprintf(buffer, sizeof(buffer), "Trails: %s", g_show_trails ? "ON" : "OFF");
    draw_text(10, start_y, buffer);
    start_y -= line_height;
    snprintf(buffer, sizeof(buffer), "Trail Length: %.2f", g_trail_length_scale);
    draw_text(10, start_y, buffer);
    start_y -= line_height;
    snprintf(buffer, sizeof(buffer), "Trail Origin: %s", g_trail_from_birth ? "BIRTH" : "DIRECTION");
    draw_text(10, start_y, buffer);
    start_y -= line_height * 1.5f;
    
    /* Controls help */
    draw_text(10, start_y, "--- CONTROLS ---");
    start_y -= line_height;
    draw_text(10, start_y, "Mouse Drag: Rotate camera");
    start_y -= line_height;
    draw_text(10, start_y, "Mouse Wheel / +/-: Zoom");
    start_y -= line_height;
    draw_text(10, start_y, "SPACE: Pause/Resume");
    start_y -= line_height;
    draw_text(10, start_y, "[ / ]: Tilt -/+ 5 deg");
    start_y -= line_height;
    draw_text(10, start_y, ", / .: Field strength x0.9/x1.1");
    start_y -= line_height;
    draw_text(10, start_y, "1 / 2: Max particles -/+ 500");
    start_y -= line_height;
    draw_text(10, start_y, "3 / 4: Emission -/+ 50/sec");
    start_y -= line_height;
    draw_text(10, start_y, "7 / 8: Star Radius -/+ 0.1");
    start_y -= line_height;
    draw_text(10, start_y, "9 / 0: Grid Scale x0.9/x1.1");
    start_y -= line_height;
    draw_text(10, start_y, "V: Toggle field lines");
    start_y -= line_height;
    draw_text(10, start_y, "U: Toggle velocity vectors");
    start_y -= line_height;
    draw_text(10, start_y, "T: Toggle trails");
    start_y -= line_height;
    draw_text(10, start_y, "5 / 6: Trail length -/+" );
    start_y -= line_height;
    draw_text(10, start_y, "B: Trails from birth toggle");
    start_y -= line_height;
    draw_text(10, start_y, "R: Reset particles");
    start_y -= line_height;
    draw_text(10, start_y, "ESC/Q: Quit");
    
    /* Restore state */
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

/*
 * Draw the star as a lit sphere
 */
void draw_star(void) {
    glPushMatrix();
    
    /* Set material properties for the star */
    GLfloat star_ambient[]  = {0.2f, 0.15f, 0.05f, 1.0f};
    GLfloat star_diffuse[]  = {1.0f, 0.9f, 0.6f, 1.0f};
    GLfloat star_specular[] = {1.0f, 1.0f, 0.8f, 1.0f};
    GLfloat star_emission[] = {0.3f, 0.25f, 0.1f, 1.0f};
    GLfloat star_shininess  = 50.0f;
    
    glMaterialfv(GL_FRONT, GL_AMBIENT,   star_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   star_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  star_specular);
    glMaterialfv(GL_FRONT, GL_EMISSION,  star_emission);
    glMaterialf (GL_FRONT, GL_SHININESS, star_shininess);
    
    /* Use shader if available */
    if (g_shaders_supported && g_sun_program) {
        glUseProgram(g_sun_program);
        if (g_sun_time_loc != -1) {
            glUniform1f(g_sun_time_loc, g_time_accumulator);
        }
    }

    /* Draw sphere with good tessellation */
    glutSolidSphere(g_star_radius * g_star_scale, 64, 64);
    
    /* Disable shader */
    if (g_shaders_supported && g_sun_program) {
        glUseProgram(0);
    }

    glPopMatrix();
}

/*
 * Display callback
 */
void display(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    /* Set up camera */
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    float eye_x, eye_y, eye_z;
    compute_camera_position(&eye_x, &eye_y, &eye_z);
    
    gluLookAt(eye_x, eye_y, eye_z,   /* eye position */
              0.0,   0.0,   0.0,      /* look at origin */
              0.0,   1.0,   0.0);     /* up vector */
    
    /* Draw the star */
    draw_star();
    
    /* Draw debug field lines if enabled */
    draw_field_lines();

    /* Draw velocity vectors if enabled */
    draw_velocity_vectors();
    
    /* Draw particles */
    draw_particles();
    
    /* Draw 2D overlay on top */
    draw_overlay();
    
    /* Update FPS counter */
    g_frame_count++;
    int current_time = glutGet(GLUT_ELAPSED_TIME);
    if (current_time - g_last_fps_time > 500) {  /* Update every 500ms */
        g_fps = g_frame_count * 1000.0f / (current_time - g_last_fps_time);
        g_frame_count = 0;
        g_last_fps_time = current_time;
    }
    
    glutSwapBuffers();
}

/*
 * Reshape callback
 */
void reshape(int width, int height) {
    g_window_width  = width;
    g_window_height = height;
    
    glViewport(0, 0, width, height);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    float aspect = (float)width / (float)height;
    gluPerspective(45.0, aspect, 0.1, 100.0);
    
    glMatrixMode(GL_MODELVIEW);
}

/*
 * Keyboard callback
 */
void keyboard(unsigned char key, int x, int y) {
    switch(key) {
        case 27:  /* ESC */
        case 'q':
        case 'Q':
            exit(0);
            break;
            
        case ' ':
            g_paused = !g_paused;
            printf("Simulation %s\n", g_paused ? "PAUSED" : "RUNNING");
            break;
            
        case '+':
        case '=':
            g_cam_radius *= 0.9f;
            if (g_cam_radius < 0.5f) g_cam_radius = 0.5f;
            glutPostRedisplay();
            break;
            
        case '-':
        case '_':
            g_cam_radius *= 1.1f;
            if (g_cam_radius > 50.0f) g_cam_radius = 50.0f;
            glutPostRedisplay();
            break;
            
        /* Field controls */
        case '[':
            g_dipole_tilt_deg -= 5.0f;
            printf("Dipole tilt: %.1f deg\n", g_dipole_tilt_deg);
            glutPostRedisplay();
            break;
            
        case ']':
            g_dipole_tilt_deg += 5.0f;
            printf("Dipole tilt: %.1f deg\n", g_dipole_tilt_deg);
            glutPostRedisplay();
            break;
            
        case ',':
        case '<':
            g_field_strength *= 0.9f;
            if (g_field_strength < 0.01f) g_field_strength = 0.01f;
            printf("Field strength: %.2f\n", g_field_strength);
            glutPostRedisplay();
            break;
            
        case '.':
        case '>':
            g_field_strength *= 1.1f;
            if (g_field_strength > 100.0f) g_field_strength = 100.0f;
            printf("Field strength: %.2f\n", g_field_strength);
            glutPostRedisplay();
            break;
            
        /* Debug visualization */
        case 'v':
        case 'V':
            g_show_field_lines = !g_show_field_lines;
            printf("Field line visualization: %s\n", g_show_field_lines ? "ON" : "OFF");
            glutPostRedisplay();
            break;

        case 'u':
        case 'U':
            g_show_vectors = !g_show_vectors;
            printf("Velocity vector visualization: %s\n", g_show_vectors ? "ON" : "OFF");
            glutPostRedisplay();
            break;

        case 't':
        case 'T':
            g_show_trails = !g_show_trails;
            printf("Particle trails: %s\n", g_show_trails ? "ON" : "OFF");
            glutPostRedisplay();
            break;

        case '5':
            g_trail_length_scale -= 0.1f;
            if (g_trail_length_scale < 0.1f) g_trail_length_scale = 0.1f;
            printf("Trail length scale: %.2f\n", g_trail_length_scale);
            glutPostRedisplay();
            break;

        case '6':
            g_trail_length_scale += 0.1f;
            if (g_trail_length_scale > 3.0f) g_trail_length_scale = 3.0f;
            printf("Trail length scale: %.2f\n", g_trail_length_scale);
            glutPostRedisplay();
            break;

        case '7':
            g_star_radius -= 0.1f;
            if (g_star_radius < 0.1f) g_star_radius = 0.1f;
            printf("Star Radius: %.1f\n", g_star_radius);
            glutPostRedisplay();
            break;

        case '8':
            g_star_radius += 0.1f;
            printf("Star Radius: %.1f\n", g_star_radius);
            glutPostRedisplay();
            break;

        case 'b':
        case 'B':
            g_trail_from_birth = !g_trail_from_birth;
            printf("Trails from birth: %s\n", g_trail_from_birth ? "ON" : "OFF");
            glutPostRedisplay();
            break;
            
        /* Particle controls */
        case '1':
            g_max_active_particles -= 500;
            if (g_max_active_particles < 100) g_max_active_particles = 100;
            printf("Max particles: %d\n", g_max_active_particles);
            break;
            
        case '2':
            g_max_active_particles += 500;
            if (g_max_active_particles > MAX_PARTICLES) g_max_active_particles = MAX_PARTICLES;
            printf("Max particles: %d\n", g_max_active_particles);
            break;
            
        case '3':
            g_emission_rate -= 50.0f;
            if (g_emission_rate < 10.0f) g_emission_rate = 10.0f;
            printf("Emission rate: %.0f/sec\n", g_emission_rate);
            break;
            
        case '4':
            g_emission_rate += 50.0f;
            if (g_emission_rate > 5000.0f) g_emission_rate = 5000.0f;
            printf("Emission rate: %.0f/sec\n", g_emission_rate);
            break;
            
        case 'r':
        case 'R':
            init_particles();
            printf("Particles reset\n");
            break;
            
        /* Grid Scale Controls */
        case '9':
            g_grid_scale *= 0.9f;
            printf("Grid Scale: %.3f\n", g_grid_scale);
            break;
        case '0':
            g_grid_scale *= 1.1f;
            printf("Grid Scale: %.3f\n", g_grid_scale);
            break;
    }
}

/*
 * Special key callback (arrow keys, function keys)
 */
void special_key(int key, int x, int y) {
    switch(key) {
        case GLUT_KEY_LEFT:
            g_cam_azimuth -= 5.0f;
            glutPostRedisplay();
            break;
        case GLUT_KEY_RIGHT:
            g_cam_azimuth += 5.0f;
            glutPostRedisplay();
            break;
        case GLUT_KEY_UP:
            g_cam_elevation += 5.0f;
            if (g_cam_elevation > 89.0f) g_cam_elevation = 89.0f;
            glutPostRedisplay();
            break;
        case GLUT_KEY_DOWN:
            g_cam_elevation -= 5.0f;
            if (g_cam_elevation < -89.0f) g_cam_elevation = -89.0f;
            glutPostRedisplay();
            break;
    }
}

/*
 * Mouse button callback
 */
void mouse_button(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            g_mouse_dragging = 1;
            g_mouse_last_x = x;
            g_mouse_last_y = y;
        } else {
            g_mouse_dragging = 0;
        }
    }
    
    /* Mouse wheel zoom (GLUT_WHEEL in some implementations) */
    if (button == 3) {  /* Scroll up */
        g_cam_radius *= 0.9f;
        if (g_cam_radius < 0.5f) g_cam_radius = 0.5f;
        glutPostRedisplay();
    } else if (button == 4) {  /* Scroll down */
        g_cam_radius *= 1.1f;
        if (g_cam_radius > 50.0f) g_cam_radius = 50.0f;
        glutPostRedisplay();
    }
}

/*
 * Mouse motion callback (when dragging)
 */
void mouse_motion(int x, int y) {
    if (g_mouse_dragging) {
        int dx = x - g_mouse_last_x;
        int dy = y - g_mouse_last_y;
        
        /* Update camera angles based on drag */
        g_cam_azimuth += dx * 0.5f;
        g_cam_elevation -= dy * 0.5f;
        
        /* Clamp elevation */
        if (g_cam_elevation > 89.0f)  g_cam_elevation = 89.0f;
        if (g_cam_elevation < -89.0f) g_cam_elevation = -89.0f;
        
        g_mouse_last_x = x;
        g_mouse_last_y = y;
        
        glutPostRedisplay();
    }
}

/*
 * Timer callback (for 24 FPS animation)
 */
void timer(int value) {
    if (!g_paused) {
        /* Frame-rate independent simulation */
        int current_time = glutGet(GLUT_ELAPSED_TIME);
        if (g_last_sim_time == 0) g_last_sim_time = current_time;
        
        float elapsed = (current_time - g_last_sim_time) / 1000.0f;
        g_last_sim_time = current_time;
        
        /* Accumulate time and step with fixed dt */
        g_time_accumulator += elapsed;
        
        /* Cap to prevent spiral of death */
        if (g_time_accumulator > 0.1f) g_time_accumulator = 0.1f;
        
        while (g_time_accumulator >= g_simulation_dt) {
            update_particles(g_simulation_dt);
            g_time_accumulator -= g_simulation_dt;
        }
    }
    glutPostRedisplay();
    glutTimerFunc(1000 / 24, timer, 0);
}

/*
 * Initialize particle system
 */
void init_particles(void) {
    for (int i = 0; i < MAX_PARTICLES; i++) {
        g_alive_arr[i] = 0;
        g_age_arr[i] = 0.0f;
        g_life_arr[i] = 1.0f;
        g_pos[i][0] = g_pos[i][1] = g_pos[i][2] = 0.0f;
        g_birth_pos[i][0] = g_birth_pos[i][1] = g_birth_pos[i][2] = 0.0f;
        g_vel[i][0] = g_vel[i][1] = g_vel[i][2] = 0.0f;
    }
    g_emission_accumulator = 0.0f;

    if (g_cl_queue) {
        clEnqueueWriteBuffer(g_cl_queue, g_cl_pos,   CL_TRUE, 0, sizeof(float)*4*MAX_PARTICLES, g_pos, 0, NULL, NULL);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_vel,   CL_TRUE, 0, sizeof(float)*4*MAX_PARTICLES, g_vel, 0, NULL, NULL);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_age,   CL_TRUE, 0, sizeof(float)*MAX_PARTICLES,   g_age_arr, 0, NULL, NULL);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_life,  CL_TRUE, 0, sizeof(float)*MAX_PARTICLES,   g_life_arr, 0, NULL, NULL);
        clEnqueueWriteBuffer(g_cl_queue, g_cl_alive, CL_TRUE, 0, sizeof(int)*MAX_PARTICLES,     g_alive_arr, 0, NULL, NULL);
    }
}

/*
 * Spawn a new particle
 */
void spawn_particle(int index) {
    /* Random position on sphere surface, slightly above star radius */
    float dir[3];
    random_unit_sphere(dir);
    float spawn_radius = g_star_radius * g_star_scale * 1.05f;
    g_pos[index][0] = dir[0] * spawn_radius;
    g_pos[index][1] = dir[1] * spawn_radius;
    g_pos[index][2] = dir[2] * spawn_radius;
    g_pos[index][3] = 1.0f;
    g_birth_pos[index][0] = g_pos[index][0];
    g_birth_pos[index][1] = g_pos[index][1];
    g_birth_pos[index][2] = g_pos[index][2];
    g_birth_pos[index][3] = 1.0f;
    
    /* Initial velocity - small tangential component */
    float tangent[3];
    if (fabsf(dir[0]) < 0.9f) {
        tangent[0] = 0.0f;
        tangent[1] = dir[2];
        tangent[2] = -dir[1];
    } else {
        tangent[0] = -dir[2];
        tangent[1] = 0.0f;
        tangent[2] = dir[0];
    }
    vec3_normalize(tangent);
    
    float speed = randf(0.3f, 0.8f);
    g_vel[index][0] = tangent[0] * speed;
    g_vel[index][1] = tangent[1] * speed;
    g_vel[index][2] = tangent[2] * speed;
    g_vel[index][3] = 0.0f;
    
    /* Lifetime */
    g_life_arr[index] = randf(8.0f, 15.0f);
    g_age_arr[index]  = 0.0f;
    g_alive_arr[index] = 1;
}

/*
 * Update particle system (Euler integration)
 */
int load_grid_frame(int frame_idx) {
    char filename[256];
    sprintf(filename, "v1/untitled.VelocityField_v1.%04d.bin", frame_idx);
    
    // Update global filename for display
    strncpy(g_current_filename, filename, sizeof(g_current_filename) - 1);
    g_current_filename[sizeof(g_current_filename) - 1] = '\0';
    
    FILE *f = fopen(filename, "rb");
    if (!f) {
        // Only print error once per frame to avoid spam
        static int last_error_frame = -1;
        if (last_error_frame != frame_idx) {
            // fprintf(stderr, "Failed to open grid file: %s\n", filename);
            last_error_frame = frame_idx;
        }
        return 0;
    }
    
    GridHeader header;
    if (fread(&header, sizeof(GridHeader), 1, f) != 1) {
        fclose(f);
        return 0;
    }
    
    if (strncmp(header.magic, "GRID", 4) != 0) {
        fprintf(stderr, "Invalid grid file format: %s\n", filename);
        fclose(f);
        return 0;
    }
    
    g_grid_header = header;
    int num_voxels = header.dim_x * header.dim_y * header.dim_z;
    size_t data_size = num_voxels * 3 * sizeof(float);
    
    // Check file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, sizeof(GridHeader), SEEK_SET);

    if (g_grid_data == NULL || data_size > g_grid_data_capacity) {
        if (g_grid_data) free(g_grid_data);
        g_grid_data = (float*)malloc(data_size);
        g_grid_data_capacity = data_size;
        
        // Also release OpenCL buffer if it exists, as we need to resize it
        if (g_cl_grid_data) {
            clReleaseMemObject(g_cl_grid_data);
            g_cl_grid_data = NULL;
        }
    }
    
    size_t total_read = 0;
    size_t bytes_left = data_size;
    char *ptr = (char*)g_grid_data;
    
    while (bytes_left > 0) {
        size_t r = fread(ptr, 1, bytes_left, f);
        if (r == 0) {
            if (ferror(f)) {
                fprintf(stderr, "Read error at offset %zu: %s (errno=%d)\n", total_read, strerror(errno), errno);
            }
            if (feof(f)) {
                fprintf(stderr, "Unexpected EOF at offset %zu. File size: %ld, Expected data: %zu\n", total_read, file_size, data_size);
            }
            break;
        }
        ptr += r;
        total_read += r;
        bytes_left -= r;
    }

    if (total_read != data_size) {
        fprintf(stderr, "Incomplete grid data: %s. Expected %zu, got %zu. File Size: %ld\n", 
                filename, data_size, total_read, file_size);
        fclose(f);
        return 0;
    }
    
    fclose(f);
    
    // Upload to GPU
    cl_int err;
    #ifndef CL_MEM_COPY_HOST_PTR
    #define CL_MEM_COPY_HOST_PTR (1 << 5)
    #endif
    if (g_cl_grid_data == NULL) {
        g_cl_grid_data = clCreateBuffer(g_cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_size, g_grid_data, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create grid buffer: %d\n", err);
            return 0;
        }
    } else {
        // Write to existing buffer
        err = clEnqueueWriteBuffer(g_cl_queue, g_cl_grid_data, CL_TRUE, 0, data_size, g_grid_data, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to write grid buffer: %d\n", err);
            return 0;
        }
    }
    
    return 1;
}

void update_particles(float dt) {
    cl_int err;

    /* Try to load grid frame */
    if (load_grid_frame(g_current_frame)) {
        g_use_grid = 1;
        g_current_frame++;
        if (g_current_frame > g_total_frames) g_current_frame = 1;
    } else {
        // If load fails (e.g. file not found), keep using last grid or disable?
        // If we never loaded a grid, disable.
        if (g_cl_grid_data == NULL) g_use_grid = 0;
    }

    /* Emit new particles on CPU, upload only changed slots */
    g_emission_accumulator += g_emission_rate * dt;
    while (g_emission_accumulator >= 1.0f) {
        int spawned = 0;
        for (int i = 0; i < g_max_active_particles && i < MAX_PARTICLES; i++) {
            if (!g_alive_arr[i]) {
                spawn_particle(i);
                size_t offset_pos = sizeof(float)*4*i;
                size_t offset_age = sizeof(float)*i;
                size_t offset_int = sizeof(int)*i;
                clEnqueueWriteBuffer(g_cl_queue, g_cl_pos,   CL_FALSE, offset_pos, sizeof(float)*4, g_pos[i], 0, NULL, NULL);
                clEnqueueWriteBuffer(g_cl_queue, g_cl_vel,   CL_FALSE, offset_pos, sizeof(float)*4, g_vel[i], 0, NULL, NULL);
                clEnqueueWriteBuffer(g_cl_queue, g_cl_age,   CL_FALSE, offset_age, sizeof(float),  &g_age_arr[i], 0, NULL, NULL);
                clEnqueueWriteBuffer(g_cl_queue, g_cl_life,  CL_FALSE, offset_age, sizeof(float),  &g_life_arr[i], 0, NULL, NULL);
                clEnqueueWriteBuffer(g_cl_queue, g_cl_alive, CL_FALSE, offset_int, sizeof(int),    &g_alive_arr[i], 0, NULL, NULL);
                spawned = 1;
                break;
            }
        }
        g_emission_accumulator -= 1.0f;
        if (!spawned) break;
    }

    clFinish(g_cl_queue);

    /* Set kernel args */
    int idx = 0;
    err  = clSetKernelArg(g_cl_kernel, idx++, sizeof(cl_mem), &g_cl_pos);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(cl_mem), &g_cl_vel);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(cl_mem), &g_cl_age);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(cl_mem), &g_cl_life);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(cl_mem), &g_cl_alive);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(float), &dt);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(float), &g_lorentz_k);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(float), &g_field_strength);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(float), &g_dipole_tilt_deg);
    
    /* Grid args */
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(cl_mem), &g_cl_grid_data);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(int), &g_use_grid);
    
    int grid_dim[4] = {0};
    float grid_origin[4] = {0};
    float grid_voxel_size = 1.0f;
    
    if (g_use_grid) {
        grid_dim[0] = g_grid_header.dim_x;
        grid_dim[1] = g_grid_header.dim_y;
        grid_dim[2] = g_grid_header.dim_z;
        grid_origin[0] = g_grid_header.origin_x;
        grid_origin[1] = g_grid_header.origin_y;
        grid_origin[2] = g_grid_header.origin_z;
        grid_voxel_size = g_grid_header.voxel_size;
    }
    
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(int)*4, grid_dim);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(float)*4, grid_origin);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(float), &grid_voxel_size);
    err |= clSetKernelArg(g_cl_kernel, idx++, sizeof(float), &g_grid_scale);
    
    cl_check(err, "clSetKernelArg");

    size_t global = MAX_PARTICLES;
    err = clEnqueueNDRangeKernel(g_cl_queue, g_cl_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    cl_check(err, "clEnqueueNDRangeKernel");

    /* Read back results for rendering */
    clEnqueueReadBuffer(g_cl_queue, g_cl_pos,   CL_FALSE, 0, sizeof(float)*4*MAX_PARTICLES, g_pos, 0, NULL, NULL);
    clEnqueueReadBuffer(g_cl_queue, g_cl_vel,   CL_FALSE, 0, sizeof(float)*4*MAX_PARTICLES, g_vel, 0, NULL, NULL);
    clEnqueueReadBuffer(g_cl_queue, g_cl_age,   CL_FALSE, 0, sizeof(float)*MAX_PARTICLES,   g_age_arr, 0, NULL, NULL);
    clEnqueueReadBuffer(g_cl_queue, g_cl_life,  CL_FALSE, 0, sizeof(float)*MAX_PARTICLES,   g_life_arr, 0, NULL, NULL);
    clEnqueueReadBuffer(g_cl_queue, g_cl_alive, CL_FALSE, 0, sizeof(int)*MAX_PARTICLES,     g_alive_arr, 0, NULL, NULL);
    clFinish(g_cl_queue);
}

/*
 * Render particles
 */
void draw_particles(void) {
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    
    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (!g_alive_arr[i]) continue;

        float age_factor = 1.0f - (g_age_arr[i] / g_life_arr[i]);
        float speed = sqrtf(g_vel[i][0]*g_vel[i][0] + g_vel[i][1]*g_vel[i][1] + g_vel[i][2]*g_vel[i][2]);

        /* Map speed to color: slow=warm orange, medium=yellow, fast=cyan/white */
        float t = fminf(speed / 3.0f, 1.0f);
        float r, gg, b;
        if (t < 0.5f) {
            /* 0..0.5: orange -> yellow */
            float k = t * 2.0f;
            r  = 0.9f + 0.1f * k;      /* 0.9 -> 1.0 */
            gg = 0.4f + 0.6f * k;      /* 0.4 -> 1.0 */
            b  = 0.1f + 0.2f * k;      /* 0.1 -> 0.3 */
        } else {
            /* 0.5..1: yellow -> cyan/white */
            float k = (t - 0.5f) * 2.0f;
            r  = 1.0f - 0.6f * k;      /* 1.0 -> 0.4 */
            gg = 1.0f;                 /* stay bright */
            b  = 0.3f + 0.7f * k;      /* 0.3 -> 1.0 */
        }

        float a = age_factor * 0.9f;

        glColor4f(r, gg, b, a);
        glVertex3f(g_pos[i][0], g_pos[i][1], g_pos[i][2]);
    }
    
    glEnd();

    if (g_show_trails) {
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        for (int i = 0; i < MAX_PARTICLES; i++) {
            if (!g_alive_arr[i]) continue;

            float speed = sqrtf(g_vel[i][0]*g_vel[i][0] + g_vel[i][1]*g_vel[i][1] + g_vel[i][2]*g_vel[i][2]);
            float age_factor = 1.0f - (g_age_arr[i] / g_life_arr[i]);
            float t = fminf(speed / 3.0f, 1.0f);
            float r, gg, b;
            if (t < 0.5f) {
                float k = t * 2.0f;
                r  = 0.9f + 0.1f * k;
                gg = 0.4f + 0.6f * k;
                b  = 0.1f + 0.2f * k;
            } else {
                float k = (t - 0.5f) * 2.0f;
                r  = 1.0f - 0.6f * k;
                gg = 1.0f;
                b  = 0.3f + 0.7f * k;
            }

            float a_head = age_factor * 0.7f;
            float a_tail = age_factor * 0.05f;

            float tail_x, tail_y, tail_z;
            if (g_trail_from_birth) {
                float f = fminf(g_trail_length_scale, 3.0f);
                tail_x = g_pos[i][0] + (g_birth_pos[i][0] - g_pos[i][0]) * f;
                tail_y = g_pos[i][1] + (g_birth_pos[i][1] - g_pos[i][1]) * f;
                tail_z = g_pos[i][2] + (g_birth_pos[i][2] - g_pos[i][2]) * f;
            } else {
                float trail_scale = 0.15f * g_trail_length_scale * fminf(speed, 3.0f);
                tail_x = g_pos[i][0] - g_vel[i][0] * trail_scale;
                tail_y = g_pos[i][1] - g_vel[i][1] * trail_scale;
                tail_z = g_pos[i][2] - g_vel[i][2] * trail_scale;
            }

            glColor4f(r, gg, b, a_tail);
            glVertex3f(tail_x, tail_y, tail_z);
            glColor4f(r, gg, b, a_head);
            glVertex3f(g_pos[i][0], g_pos[i][1], g_pos[i][2]);
        }
        glEnd();
    }

    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

/* --- Sun Shader Source --- */
const char *g_sun_vs_src = 
"#version 120\n"
"varying vec3 v_pos;\n"
"varying vec3 v_normal;\n"
"varying vec3 v_viewDir;\n"
"void main() {\n"
"    v_pos = gl_Vertex.xyz;\n"
"    // Normal in eye space\n"
"    v_normal = normalize(gl_NormalMatrix * gl_Normal);\n"
"    // Vertex position in eye space\n"
"    vec4 posEye = gl_ModelViewMatrix * gl_Vertex;\n"
"    // View direction is vector from vertex to eye (0,0,0) in eye space\n"
"    v_viewDir = normalize(-posEye.xyz);\n"
"    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
"}\n";

const char *g_sun_fs_src = 
"#version 120\n"
"uniform float time;\n"
"varying vec3 v_pos;\n"
"varying vec3 v_normal;\n"
"varying vec3 v_viewDir;\n"
"\n"
"// 3D Simplex Noise (approx)\n"
"vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }\n"
"vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }\n"
"vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }\n"
"vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }\n"
"\n"
"float snoise(vec3 v) {\n"
"  const vec2  C = vec2(1.0/6.0, 1.0/3.0);\n"
"  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);\n"
"  vec3 i  = floor(v + dot(v, C.yyy));\n"
"  vec3 x0 = v - i + dot(i, C.xxx);\n"
"  vec3 g = step(x0.yzx, x0.xyz);\n"
"  vec3 l = 1.0 - g;\n"
"  vec3 i1 = min( g.xyz, l.zxy );\n"
"  vec3 i2 = max( g.xyz, l.zxy );\n"
"  vec3 x1 = x0 - i1 + C.xxx;\n"
"  vec3 x2 = x0 - i2 + C.yyy;\n"
"  vec3 x3 = x0 - D.yyy;\n"
"  i = mod289(i);\n"
"  vec4 p = permute( permute( permute(\n"
"             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))\n"
"           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))\n"
"           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));\n"
"  float n_ = 0.142857142857;\n"
"  vec3  ns = n_ * D.wyz - D.xzx;\n"
"  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);\n"
"  vec4 x_ = floor(j * ns.z);\n"
"  vec4 y_ = floor(j - 7.0 * x_ );\n"
"  vec4 x = x_ *ns.x + ns.yyyy;\n"
"  vec4 y = y_ *ns.x + ns.yyyy;\n"
"  vec4 h = 1.0 - abs(x) - abs(y);\n"
"  vec4 b0 = vec4( x.xy, y.xy );\n"
"  vec4 b1 = vec4( x.zw, y.zw );\n"
"  vec4 s0 = floor(b0)*2.0 + 1.0;\n"
"  vec4 s1 = floor(b1)*2.0 + 1.0;\n"
"  vec4 sh = -step(h, vec4(0.0));\n"
"  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;\n"
"  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;\n"
"  vec3 p0 = vec3(a0.xy,h.x);\n"
"  vec3 p1 = vec3(a0.zw,h.y);\n"
"  vec3 p2 = vec3(a1.xy,h.z);\n"
"  vec3 p3 = vec3(a1.zw,h.w);\n"
"  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));\n"
"  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;\n"
"  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);\n"
"  m = m * m;\n"
"  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );\n"
"}\n"
"\n"
"float fbm(vec3 p) {\n"
"    float f = 0.0;\n"
"    float amp = 0.5;\n"
"    float freq = 1.0;\n"
"    for(int i = 0; i < 4; i++) {\n"
"        f += amp * snoise(p * freq);\n"
"        p += vec3(1.23);\n"
"        amp *= 0.5;\n"
"        freq *= 2.02;\n"
"    }\n"
"    return f;\n"
"}\n"
"\n"
"void main() {\n"
"    vec3 coord = v_pos * 0.5;\n"
"    float t = time * 0.15;\n"
"    float n1 = fbm(coord + vec3(0.0, t, 0.0));\n"
"    float n2 = fbm(coord * 2.0 - vec3(t * 0.5, 0.0, 0.0));\n"
"    float noiseVal = n1 * 0.6 + n2 * 0.4;\n"
"    \n"
"    vec3 c_deep = vec3(0.6, 0.0, 0.0);\n"
"    vec3 c_mid  = vec3(1.0, 0.2, 0.0);\n"
"    vec3 c_high = vec3(1.0, 0.9, 0.1);\n"
"    vec3 c_hot  = vec3(1.0, 1.0, 0.8);\n"
"    \n"
"    float val = noiseVal * 0.5 + 0.5;\n"
"    vec3 surfaceColor;\n"
"    if (val < 0.3) surfaceColor = mix(c_deep, c_mid, val / 0.3);\n"
"    else if (val < 0.7) surfaceColor = mix(c_mid, c_high, (val - 0.3) / 0.4);\n"
"    else surfaceColor = mix(c_high, c_hot, (val - 0.7) / 0.3);\n"
"    \n"
"    float ndotv = max(0.0, dot(normalize(v_normal), normalize(v_viewDir)));\n"
"    float rim = pow(1.0 - ndotv, 3.0);\n"
"    float limb = smoothstep(0.0, 1.0, ndotv);\n"
"    surfaceColor *= (0.6 + 0.4 * limb);\n"
"    surfaceColor += vec3(1.0, 0.4, 0.1) * rim * 0.8;\n"
"    \n"
"    gl_FragColor = vec4(surfaceColor * 1.8, 1.0);\n"
"}\n";

void print_shader_log(GLuint shader) {
    GLint len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
    if (len > 1) {
        char *log = (char*)malloc(len);
        glGetShaderInfoLog(shader, len, NULL, log);
        printf("Shader Log:\n%s\n", log);
        free(log);
    }
}

void print_program_log(GLuint prog) {
    GLint len = 0;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
    if (len > 1) {
        char *log = (char*)malloc(len);
        glGetProgramInfoLog(prog, len, NULL, log);
        printf("Program Log:\n%s\n", log);
        free(log);
    }
}

void init_sun_shader(void) {
    if (!g_shaders_supported) return;

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &g_sun_vs_src, NULL);
    glCompileShader(vs);
    print_shader_log(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &g_sun_fs_src, NULL);
    glCompileShader(fs);
    print_shader_log(fs);

    g_sun_program = glCreateProgram();
    glAttachShader(g_sun_program, vs);
    glAttachShader(g_sun_program, fs);
    glLinkProgram(g_sun_program);
    print_program_log(g_sun_program);

    g_sun_time_loc = glGetUniformLocation(g_sun_program, "time");
    printf("Sun shader initialized. Program ID: %d\n", g_sun_program);
}

/*
 * Initialize OpenGL state
 */
void init_gl(void) {
    /* Background color (black space) */
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    /* Enable depth testing */
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    
    /* Enable smooth shading */
    glShadeModel(GL_SMOOTH);
    
    /* Enable lighting */
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    
    /* Set up light 0 */
    GLfloat light_position[] = {5.0f, 5.0f, 5.0f, 1.0f};
    GLfloat light_ambient[]  = {0.2f, 0.2f, 0.2f, 1.0f};
    GLfloat light_diffuse[]  = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat light_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
    
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    
    /* Enable color material for easier material changes */
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    
    /* Improve appearance */
    glEnable(GL_NORMALIZE);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

/*
 * Main entry point
 */
int main(int argc, char **argv) {
    printf("Star Magnetic Field Particle Simulation\n");
    printf("========================================\n");
    printf("Controls:\n");
    printf("  Mouse Drag : Rotate camera\n");
    printf("  Mouse Wheel: Zoom in/out\n");
    printf("  Arrow keys : Rotate camera\n");
    printf("  +/-        : Zoom in/out\n");
    printf("  SPACE      : Pause/resume\n");
    printf("  ESC/Q      : Quit\n");
        printf("\n");
    printf("On-screen display shows current parameters and controls.\n");
    printf("\n");
    
    /* Initialize GLUT */
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(g_window_width, g_window_height);
    glutCreateWindow("Star Magnetic Field Simulation");
    
    /* Register callbacks */
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(special_key);
    glutMouseFunc(mouse_button);
    glutMotionFunc(mouse_motion);
    glutTimerFunc(0, timer, 0);
    
    /* Initialize random seed */
    srand((unsigned int)time(NULL));
    
    /* Initialize OpenGL state */
    init_gl();

#ifdef _WIN32
    /* Load Shader Extensions */
    load_shader_extensions();
    init_sun_shader();
#endif

    /* Initialize OpenCL */
    if (!init_opencl()) {
        fprintf(stderr, "Failed to initialize OpenCL.\n");
        return 1;
    }
    
    /* Initialize particle system */
    init_particles();
    
    /* Enter main loop */
    glutMainLoop();
    
    return 0;
}
