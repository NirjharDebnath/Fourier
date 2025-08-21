#include <iostream>
#include <SDL2/SDL.h>
#include <cuda_runtime.h>
#include <vector_types.h> // For float3

// --- NEW: Professional CUDA error checking macro ---
#define CUDA_CHECK(err) __cuda_check_errors(err, __FILE__, __LINE__)
inline void __cuda_check_errors(cudaError_t err, const char* file, const int line) {
    if (cudaSuccess != err) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define WIDTH 1600
#define HEIGHT 800
#define RAYS_NUMBER 10000
#define MAX_SHADOWS 20
#define MAX_BOUNCES 10

// --- CHANGED: Using 'float' instead of 'double' for performance ---
struct Circle {
    float x;
    float y;
    float r;
};

struct Ray {
    float x_start, y_start;
    float dx, dy;
    float intensity;
    int bounce_count;
};

__device__ int RayIntersectsCircle(float x, float y, const Circle& circle) {
    float dx = x - circle.x;
    float dy = y - circle.y;
    float distance_sq = dx * dx + dy * dy;
    return distance_sq <= (circle.r * circle.r);
}

__device__ void handle_reflection(Ray* ray, float hit_x, float hit_y, const Circle& circle) {
    float nx = (hit_x - circle.x) / circle.r;
    float ny = (hit_y - circle.y) / circle.r;
    
    float ix = ray->dx;
    float iy = ray->dy;
    
    float dot = ix * nx + iy * ny;
    float rx = ix - 2 * dot * nx;
    float ry = iy - 2 * dot * ny;
    
    float rlen = sqrtf(rx * rx + ry * ry);
    if (rlen > 0) {
        rx /= rlen;
        ry /= rlen;
    }
    
    ray->dx = rx;
    ray->dy = ry;
    ray->x_start = hit_x + rx * 0.01f;
    ray->y_start = hit_y + ry * 0.01f;
    ray->intensity *= 0.6f;
    ray->bounce_count++;
}

// --- KERNEL 1: Accumulates light energy in an HDR buffer ---
__global__ void rayTraceKernel(Ray* rays, Circle* objects, int num_objects, float3* render_buffer, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= RAYS_NUMBER) return;

    Ray current_ray = rays[i];
    
    while (current_ray.bounce_count < MAX_BOUNCES && current_ray.intensity > 0.1f) {
        float x = current_ray.x_start;
        float y = current_ray.y_start;
        bool hit_occurred = false;
        
        for (int step = 0; step < 2000; step++) {
            x += current_ray.dx;
            y += current_ray.dy;
            
            if (x < 0 || x >= width || y < 0 || y >= height) break;
            
            float dist_sq = (x - rays[i].x_start)*(x - rays[i].x_start) + (y - rays[i].y_start)*(y - rays[i].y_start);
            float intensity = current_ray.intensity * 10000.0f / (dist_sq + 1.0f);
            
            if (intensity < 0.01f) break;
            intensity = fminf(intensity, 1.0f);
            
            // --- CHANGED: Proper color blending ---
            float3 ray_color = (current_ray.bounce_count > 0) ? make_float3(0.9f, 0.9f, 1.0f) : make_float3(1.0f, 0.9f, 0.8f);
            float3 final_color = make_float3(ray_color.x * intensity, ray_color.y * intensity, ray_color.z * intensity);

            int px = (int)x;
            int py = (int)y;
            int pixel_index = py * width + px;

            // Use atomicAdd for each color channel to blend correctly
            atomicAdd(&(render_buffer[pixel_index].x), final_color.x);
            atomicAdd(&(render_buffer[pixel_index].y), final_color.y);
            atomicAdd(&(render_buffer[pixel_index].z), final_color.z);
            
            for (int k = 0; k < num_objects; k++) {
                if (RayIntersectsCircle(x, y, objects[k])) {
                    handle_reflection(&current_ray, x, y, objects[k]);
                    hit_occurred = true;
                    break;
                }
            }
            if (hit_occurred) break;
        }
        if (!hit_occurred) break;
    }
}

// --- KERNEL 2: Converts the HDR buffer to a displayable image (Tone Mapping) ---
__global__ void toneMapKernel(float3* render_buffer, Uint32* pixels, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_count = width * height;
    if (i >= pixel_count) return;

    float3 hdr_color = render_buffer[i];

    // Simple tone mapping: clamp values to 1.0 (prevents oversaturation)
    float r = fminf(hdr_color.x, 1.0f);
    float g = fminf(hdr_color.y, 1.0f);
    float b = fminf(hdr_color.z, 1.0f);

    // Convert to 8-bit color channels
    Uint32 r_byte = (Uint32)(r * 255.0f);
    Uint32 g_byte = (Uint32)(g * 255.0f);
    Uint32 b_byte = (Uint32)(b * 255.0f);

    // Pack into a final Uint32 pixel
    pixels[i] = (r_byte << 16) | (g_byte << 8) | b_byte;
}

void FillCircle_Outline(SDL_Surface *surface, Circle circle, Uint32 color) {
    // ... (This function remains unchanged, just ensure it uses float)
    int x0 = (int)circle.x;
    int y0 = (int)circle.y;
    int radius = (int)circle.r;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x * x + y * y >= (radius * radius) - 100 && x * x + y * y <= (radius * radius) + 100) {
                int px = x0 + x;
                int py = y0 + y;
                if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
                    ((Uint32 *)surface->pixels)[py * (surface->pitch / 4) + px] = color;
                }
            }
        }
    }
}

void generate_rays(Circle circle, Ray rays[RAYS_NUMBER]) {
    for (int i = 0; i < RAYS_NUMBER; i++) {
        float angle = ((float)i / RAYS_NUMBER) * 2.0f * M_PI;
        rays[i] = (Ray){ circle.x, circle.y, cosf(angle), sinf(angle), 1.0f, 0 };
    }
}

int main() {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("CUDA Ray Tracer V2", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
    SDL_Surface *surface = SDL_GetWindowSurface(window);

    Circle light_source = {200, 200, 20};
    Circle h_circles[] = {
        {200, 200, 120}, {1200, 500, 160}, {400, 500, 160},
        {800, 300, 100}, {1200, 200, 100}
    };
    int num_shadows = 5;
    Ray h_rays[RAYS_NUMBER];

    // --- GPU Memory Allocation ---
    Ray* d_rays;
    Circle* d_circles;
    float3* d_render_buffer; // HDR buffer
    Uint32* d_pixels;        // Final output buffer

    CUDA_CHECK(cudaMalloc(&d_rays, RAYS_NUMBER * sizeof(Ray)));
    CUDA_CHECK(cudaMalloc(&d_circles, num_shadows * sizeof(Circle)));
    CUDA_CHECK(cudaMalloc(&d_render_buffer, WIDTH * HEIGHT * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_pixels, WIDTH * HEIGHT * sizeof(Uint32)));

    CUDA_CHECK(cudaMemcpy(d_circles, h_circles, num_shadows * sizeof(Circle), cudaMemcpyHostToDevice));

    int simulation_running = 1;
    SDL_Event event;
    
    generate_rays(light_source, h_rays);

    // Corrected main loop from raytracer_v2.cu

    while (simulation_running) {
        bool light_moved = false;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) simulation_running = 0;
            if (event.type == SDL_MOUSEMOTION && event.motion.state != 0) {
                light_source.x = event.motion.x;
                light_source.y = event.motion.y;
                light_moved = true;
            }
        }

        if (light_moved) {
            generate_rays(light_source, h_rays);
        }

        // --- Render Pipeline ---
        // 1. Clear the intermediate buffer on the GPU
        CUDA_CHECK(cudaMemset(d_render_buffer, 0, WIDTH * HEIGHT * sizeof(float3)));

        // 2. Copy the latest ray data to the GPU
        CUDA_CHECK(cudaMemcpy(d_rays, h_rays, RAYS_NUMBER * sizeof(Ray), cudaMemcpyHostToDevice));

        // 3. Launch Ray Tracing Kernel
        int threadsPerBlock = 256;
        int blocksPerGridRays = (RAYS_NUMBER + threadsPerBlock - 1) / threadsPerBlock;
        rayTraceKernel<<<blocksPerGridRays, threadsPerBlock>>>(d_rays, d_circles, num_shadows, d_render_buffer, WIDTH, HEIGHT);
        CUDA_CHECK(cudaGetLastError());

        // 4. Launch Tone Mapping Kernel
        int blocksPerGridPixels = (WIDTH * HEIGHT + threadsPerBlock - 1) / threadsPerBlock;
        toneMapKernel<<<blocksPerGridPixels, threadsPerBlock>>>(d_render_buffer, d_pixels, WIDTH, HEIGHT);
        CUDA_CHECK(cudaGetLastError());

        // 5. Copy final image from GPU to SDL's surface
        CUDA_CHECK(cudaMemcpy(surface->pixels, d_pixels, WIDTH * HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost));
        
        // REMOVED: SDL_FillRect(surface, NULL, 0x00000000);

        // Now, draw the outlines directly on top of the image we just copied
        for (int i = 0; i < num_shadows; i++) {
            FillCircle_Outline(surface, h_circles[i], 0xffffffff);
        }
        
        SDL_UpdateWindowSurface(window);
    }

    CUDA_CHECK(cudaFree(d_rays));
    CUDA_CHECK(cudaFree(d_circles));
    CUDA_CHECK(cudaFree(d_render_buffer));
    CUDA_CHECK(cudaFree(d_pixels));

    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}