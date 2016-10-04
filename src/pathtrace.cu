#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"


#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static int iter2 = 0;
struct is_zero_bounce
{
    __host__ __device__
        bool operator()(const PathSegment p)
    {
        return (p.remainingBounces == 0);
    }
};

int obj_numshapes = 0;
int* obj_numpolyverts = NULL;
//int** obj_polysidx = NULL;
float* obj_verts = NULL;
float* obj_norms = NULL;
float* obj_texts = NULL;
int* obj_polyoffsets = NULL;
int* obj_polysidxflat = NULL;


void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
  
printf("\npolyidxcount = %d\n", scene->polyidxcount);
    // objloader part
    cudaMalloc((void**)&obj_numpolyverts, scene->obj_numshapes * sizeof(int));
    printf("\n1");
    cudaMalloc((void**)&obj_polyoffsets, scene->obj_numshapes * sizeof(int));
    printf("\n2");
    cudaMalloc((void**)&obj_polysidxflat, scene->polyidxcount * sizeof(int));
    printf("\n3");
    cudaMalloc((void**)&obj_verts, scene->objmesh->attrib.vertices.size()* sizeof(float));
    printf("\n4");
    cudaMalloc((void**)&obj_norms, scene->objmesh->attrib.normals.size()* sizeof(float));
    printf("\n5");
    cudaMalloc((void**)&obj_texts, scene->objmesh->attrib.texcoords.size()* sizeof(float));
    printf("\n6");

    cudaMemcpy(obj_numpolyverts, scene->obj_numpolyverts, scene->obj_numshapes * sizeof(int), cudaMemcpyHostToDevice);
    printf("\n7");
    cudaMemcpy(obj_polyoffsets, scene->obj_polyoffsets, scene->obj_numshapes * sizeof(int), cudaMemcpyHostToDevice);
    printf("\n8");
    cudaMemcpy(obj_polysidxflat, scene->obj_polysidxflat, scene->polyidxcount * sizeof(int), cudaMemcpyHostToDevice);
    printf("\n9");
    cudaMemcpy(obj_verts, scene->obj_verts, scene->objmesh->attrib.vertices.size()* sizeof(float), cudaMemcpyHostToDevice);
    printf("\n10");
    cudaMemcpy(obj_norms, scene->obj_norms, scene->objmesh->attrib.normals.size()* sizeof(float), cudaMemcpyHostToDevice);
    printf("\n11");
    cudaMemcpy(obj_texts, scene->obj_texts, scene->objmesh->attrib.texcoords.size()* sizeof(float), cudaMemcpyHostToDevice);
    printf("\n12\n");

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    // objloader part
    cudaFree(obj_numpolyverts);
    cudaFree(obj_polyoffsets);
    cudaFree(obj_polysidxflat);
    cudaFree(obj_verts);
    cudaFree(obj_norms);
    cudaFree(obj_texts);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // store initial index
        //segment.initialidx = index;


        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );



        float jitterscale = 0.002;
        thrust::default_random_engine rng(utilhash(iter));
        thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

        bool fast = false;
        if (fast)
        {
            // use cheap jitter
            glm::vec3 v3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
            v3 = glm::normalize(v3);
            segment.ray.direction += v3*jitterscale;
            segment.ray.direction = glm::normalize(segment.ray.direction);
        }
        else
        {
            // use uniform spherical distribution
            float u = cos(PI * (float)unitDistrib(rng));
            float u2 = u*u;
            float sqrt1minusu2 = sqrt(1 - u2);
            float theta = 2 * PI * (float)unitDistrib(rng);
            glm::vec3  v3(sqrt1minusu2 * cos(theta),
                sqrt1minusu2 * sin(theta),
                u);
            segment.ray.direction += v3*jitterscale;
        }


        segment.ray.direction = glm::normalize(segment.ray.direction);


        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO: 
// pathTraceOneBounce handles ray intersections, generate intersections for shading, 
// and scatter new ray. You might want to call scatterRay from interactions.h
__global__ void pathTraceOneBounce(
    int depth
    , int iter
    , int num_paths
    , PathSegment * pathSegments
    , Geom * geoms
    , int geoms_size
    , Material * materials
    , int material_size
    , ShadeableIntersection * intersections
    , int obj_numshapes
    , int* obj_numpolyverts
    , float* obj_verts
    , float* obj_norms
    , float* obj_texts
    , int* obj_polyoffsets
    , int* obj_polysidxflat
    , int polyidxcount
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        path_index = pathSegments[path_index].pixelIndex;
        PathSegment pathSegment = pathSegments[path_index];
        //printf("\nO1");
        if (pathSegments[path_index].remainingBounces>0)
        {
            float t;
            glm::vec3 intersect_point;
            glm::vec3 normal;
            float t_min = FLT_MAX;
            int hit_geom_index = -1;
            bool outside = true;

            glm::vec3 tmp_intersect;
            glm::vec3 tmp_normal;

            glm::vec3 hit;
            glm::vec3 norm;
            glm::vec3 bary;
            glm::vec3 v1;
            glm::vec3 v2;
            glm::vec3 v3;
            glm::vec3 n1;
            glm::vec3 n2;
            glm::vec3 n3;
            int pidxo1 = 0;
            int pidxo2 = 0;
            int pidxo3 = 0;
            bool intersected = false;

            // naive parse through global geoms
            //printf("\nO2");
            for (int i = 0; i < geoms_size; i++)
            {
                Geom & geom = geoms[i];

                if (geom.type == CUBE)
                {
                    t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                }
                else if (geom.type == SPHERE)
                {
                    t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                }
                // TODO: add more intersection tests here... triangle? metaball? CSG?

                // Compute the minimum t from the intersection tests to determine what
                // scene geometry object was hit first.
                if (t > 0.0f && t_min > t)
                {
                    t_min = t;
                    hit_geom_index = i;
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                }
            }
            
            // start polygon hits
            //t_min = FLT_MAX;
            //for (int i = 0; i < obj_numshapes; i++)
            //    printf("\noffset = %d", obj_polyoffsets[i]);
            
            /*
            //printf("\nO3");
            //printf("\nNUMSHAPES = %d\n", obj_numshapes);
            int iterator = 0;
            for (int i = 0; i < obj_numshapes; i++)
            {
                //printf("\nO4");
                //printf("loop1\n");
                for (int j = iterator; j < iterator + obj_polyoffsets[i]; j+=3)
                {
                    
                    //printf("\nO5");
                    //int pidx1 = obj_polysidxflat[j];
                    //int pidx2 = obj_polysidxflat[j + 1];
                    //int pidx3 = obj_polysidxflat[j + 2];
                    pidxo1 = 3 * obj_polysidxflat[j];
                    pidxo2 = 3 * obj_polysidxflat[j + 1];
                    pidxo3 = 3 * obj_polysidxflat[j + 2];

                    v1.x = obj_verts[pidxo1];
                    v1.y = obj_verts[pidxo1 + 1];
                    v1.z = obj_verts[pidxo1 + 2];
                    v2.x = obj_verts[pidxo2];
                    v2.y = obj_verts[pidxo2 + 1];
                    v2.z = obj_verts[pidxo2 + 2];
                    v3.x = obj_verts[pidxo3];
                    v3.y = obj_verts[pidxo3 + 1];
                    v3.z = obj_verts[pidxo3 + 2];

                    n1.x = obj_norms[pidxo1];
                    n1.y = obj_norms[pidxo1 + 1];
                    n1.z = obj_norms[pidxo1 + 2];
                    n2.x = obj_norms[pidxo2];
                    n2.y = obj_norms[pidxo2 + 1];
                    n2.z = obj_norms[pidxo2 + 2];
                    n3.x = obj_norms[pidxo3];
                    n3.y = obj_norms[pidxo3 + 1];
                    n3.z = obj_norms[pidxo3 + 2];
                    
                    //printf("\nO6");
                    //bary.x = 0.0f;
                    //bary.y = 0.0f;
                    //bary.z = 0.0f;

                    intersected = false;
                    
                    intersected = glm::intersectRayTriangle(pathSegment.ray.origin,
                        pathSegment.ray.direction,
                        v1, v2, v3, bary);
                    
                    //printf("\nO7");
                    //if (i == 0 && j == 0)
                    //{
                    //    printf("\nbary = %f %f %f", bary.x, bary.y, bary.z);
                    //    printf("\origin = %f %f %f", pathSegment.ray.origin.x, pathSegment.ray.origin.y, pathSegment.ray.origin.z);
                    //}

                    
                    if (bary.x >= 0 && bary.x <= 1 && bary.y >= 0 && bary.y <= 1 && bary.z >= 0 && bary.z <= 1)
                    {

                        //printf("\nO8");
                        hit = (bary.x * v1 + bary.y * v2 + bary.z * v3);
                        norm = (glm::normalize(bary.x * n1 + bary.y * n2 + bary.z * n3));
                        //norm(glm::normalize(n1));
                        hit += norm*0.0001f;
                        
                        
                        t = glm::distance(pathSegment.ray.origin, hit);
                        
                        if (t > 0.0f && t_min > t)
                        {
                            t_min = t;
                            hit_geom_index = 2;
                            intersect_point = hit;
                            normal = norm;
                            tmp_intersect = hit;
                            tmp_normal = normal;
                        }
                        //printf("\nO9");
                        
                    }
                    
                }
                iterator += obj_polyoffsets[i];
                //printf("\nO10");
                
            }
            
            */

            //printf("\nO11");

            // TODO: scatter the ray, generate intersections for shading
            // feel free to modify the code below

            if (hit_geom_index == -1)
            {
                intersections[path_index].t = -1.0f;
            }
            else
            {
                //The ray hits something
                //intersections[path_index].t = t_min;
                //intersections[path_index].materialId = geoms[hit_geom_index].materialid;
                //intersections[path_index].surfaceNormal = normal;


                // updating rays
                //thrust::default_random_engine rng = makeSeededRandomEngine(iter, depth, depth); // WAY TOO COOL!
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, depth);



                scatterRay(pathSegments[path_index].ray,
                    pathSegments[path_index].color,
                    tmp_intersect,
                    tmp_normal,
                    materials[geoms[hit_geom_index].materialid],
                    rng);

                //pathSegments[path_index].ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
                //pathSegments[path_index].ray.origin = intersect_point;



                intersections[path_index].t = t_min;
                intersections[path_index].materialId = geoms[hit_geom_index].materialid;
                intersections[path_index].surfaceNormal = normal;
            }
        }
    }
}


__global__ void shadeFakeMaterialTest(
    int iter
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        //idx = pathSegments[idx].initialidx;
        idx = pathSegments[idx].pixelIndex;
        if (pathSegments[idx].remainingBounces>0)
        {
            ShadeableIntersection intersection = shadeableIntersections[idx];
            if (intersection.t > 0.0f) { // if the intersection exists...
                // Set up the RNG
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
                thrust::uniform_real_distribution<float> u01(0, 1);

                Material material = materials[intersection.materialId];
                glm::vec3 materialColor = material.color;

                // If the material indicates that the object was a light, "light" the ray
                if (material.emittance > 0.0f) {
                    pathSegments[idx].color *= (materialColor * material.emittance);
                    pathSegments[idx].remainingBounces = 0;
                }
                // Otherwise, do some pseudo-lighting computation. This is actually more
                // like what you would expect from shading in a rasterizer like OpenGL.
                else {
                    //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));

                    //if (pathSegments[idx].ray.isrefr)
                    //{
                    //    pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + materialColor * 0.7f + material.hasRefractive * materialColor;
                    //}

                    //else if (pathSegments[idx].ray.isrefl)
                    //{
                    //    pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + materialColor * 0.7f + material.hasReflective * materialColor;
                    //}

                    if (material.hasRefractive)
                    {
                        
                        pathSegments[idx].color *= (materialColor) * 1.0f + materialColor * 0.7f + material.hasRefractive * materialColor;
                    }
                    else
                    {
                        pathSegments[idx].color *= (materialColor) * 1.0f + materialColor * 0.7f;
                    }

                    pathSegments[idx].remainingBounces--;
                }
                // If there was no intersection, color the ray black.
                // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
                // used for opacity, in which case they can indicate "no opacity".
                // This can be useful for post-processing and image compositing.
            }
            else {
                pathSegments[idx].color = glm::vec3(0.0f);
                pathSegments[idx].remainingBounces = 0;
            }
        }
    }
}



// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        index = iterationPaths[index].pixelIndex;
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// Add the current iteration's output to the current image
__global__ void partialGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        index = iterationPaths[index].pixelIndex;
        if (iterationPaths[index].remainingBounces == 0)
        {
            PathSegment iterationPath = iterationPaths[index];
            image[iterationPath.pixelIndex] += iterationPath.color;
        }
    }
}

/**
* Wrapper for the __global__ call that sets up the kernel calls and does a ton
* of memory management
*/
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing
    
    // trying to reallocate
    //cudaFree(dev_paths);
    //cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int num_paths_temp = num_paths;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    PathSegment* paths;
    cudaMalloc(&paths, sizeof(PathSegment)*pixelcount);
    cudaMemcpy(paths, dev_paths, sizeof(PathSegment)*pixelcount);

    bool iterationComplete = false;
    while (!iterationComplete) {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        pathTraceOneBounce << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , iter
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_materials
            , hst_scene->materials.size()
            , dev_intersections
            , hst_scene->obj_numshapes
            , obj_numpolyverts
            , obj_verts
            , obj_norms
            , obj_texts
            , obj_polyoffsets
            , obj_polysidxflat
            , hst_scene->polyidxcount);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;



        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeFakeMaterialTest << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );

        //if (depth > 2)
        //if (num_paths <= 0)
        //    iterationComplete = true; // TODO: should be based off stream compaction results.


        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        //if (iter == 25)
        partialGather << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);

        
        // DO THIS AFFTER final gathering
        thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
        thrust::device_ptr<PathSegment> P = thrust::remove_if(thrust_paths, thrust_paths + num_paths, is_zero_bounce());
        num_paths_temp = P - thrust_paths;
        num_paths = num_paths_temp;

        if (num_paths <= 0 || depth > 8)
            iterationComplete = true; // TODO: should be based off stream compaction results.
        
    }


    // Assemble this iteration and apply it to the image
    //dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    //finalGather << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);

    /*
    //printf("\ndev_paths %d\n", dev_paths[0].color.r);
    thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
    thrust::device_ptr<PathSegment> P = thrust::remove_if(thrust_paths, thrust_paths + num_paths, is_zero_bounce());
    num_paths_temp = P - thrust_paths;
    num_paths -= num_paths_temp;
    */
    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    //if (iter == 25) 
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}





/*

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
struct is_zero_bounce
{
    __host__ __device__
        bool operator()(const PathSegment p)
    {
        return (p.remainingBounces == 0);
    }
};


__global__ void accum(int n, PathSegment* odata, PathSegment* idata)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
    {
        odata[i].color += idata[i].color;
    }
}


void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
/*
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );



        float jitterscale = 0.002;
        thrust::default_random_engine rng(utilhash(iter));
        thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

        bool fast = false;
        if (fast)
        {
            // use cheap jitter
            glm::vec3 v3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
            v3 = glm::normalize(v3);
            segment.ray.direction += v3*jitterscale;
            segment.ray.direction = glm::normalize(segment.ray.direction);
        }
        else
        {
            // use uniform spherical distribution
            float u = cos(PI * (float)unitDistrib(rng));
            float u2 = u*u;
            float sqrt1minusu2 = sqrt(1 - u2);
            float theta = 2 * PI * (float)unitDistrib(rng);
            glm::vec3  v3(sqrt1minusu2 * cos(theta),
                sqrt1minusu2 * sin(theta),
                u);
            segment.ray.direction += v3*jitterscale;
        }


        segment.ray.direction = glm::normalize(segment.ray.direction);




        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}
__global__ void pathTraceOneBounce(
    int depth
    , int num_paths
    , PathSegment * pathSegments
    , Geom * geoms
    , int geoms_size
    , Material * materials
    , int material_size
    , ShadeableIntersection * intersections
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom & geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }


        // TODO: scatter the ray, generate intersections for shading
        // feel free to modify the code below

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;


            // updating rays
            thrust::default_random_engine rng = makeSeededRandomEngine(path_index, depth, hit_geom_index);
            //scatterRay(pathSegments[path_index].ray,
            ///    pathSegments[path_index].color,
            //    intersect_point,
            //    normal,
            //    intersections[path_index],
            //    thrust::default_random_engine &rng)

            
            scatterRay(pathSegments[path_index].ray,
                pathSegments[path_index].color,
                tmp_intersect,
                tmp_normal,
                materials[path_index],
                rng);
                
            //pathSegments[path_index].ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
            //pathSegments[path_index].ray.origin = intersect_point;
        }
    }
}




__global__ void shadeFakeMaterialTest(
    int iter
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        if (pathSegments[idx].remainingBounces > 0)
        {
            ShadeableIntersection intersection = shadeableIntersections[idx];
            if (intersection.t > 0.0f) { // if the intersection exists...
                // Set up the RNG
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
                thrust::uniform_real_distribution<float> u01(0, 1);

                Material material = materials[intersection.materialId];
                glm::vec3 materialColor = material.color;

                // If the material indicates that the object was a light, "light" the ray
                if (material.emittance > 0.0f) {
                    pathSegments[idx].color = (materialColor * material.emittance);
                    pathSegments[idx].remainingBounces = 0;
                }
                // Otherwise, do some pseudo-lighting computation. This is actually more
                // like what you would expect from shading in a rasterizer like OpenGL.
                else {
                    float lightTerm = glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction);
                    pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                    //pathSegments[idx].color *= (materialColor)* 0.3f + lightTerm * materialColor * 0.7f;

                    //pathSegments[idx].color *= u01(rng); // apply some noise because why not



                    //pathSegments[idx].ray.direction = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
                    //pathSegments[idx].ray.direction = intersection.surfaceNormal;
                    //pathSegments[idx].ray.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
                    //pathSegments[idx].ray.origin = intersection.
                    //pathSegments[idx].ray.direction = glm::normalize(pathSegments[idx].ray.direction);
                    pathSegments[idx].remainingBounces--;
                }
                // If there was no intersection, color the ray black.
                // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
                // used for opacity, in which case they can indicate "no opacity".
                // This can be useful for post-processing and image compositing.
            }
            else {
                pathSegments[idx].color = glm::vec3(0.0f);
                pathSegments[idx].remainingBounces = 0;
            }
        }
    }
}


// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            
            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
/*
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete) {

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        /*
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        pathTraceOneBounce << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_materials
            , hst_scene->materials.size()
            , dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;
        */

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

/*

        for (int i = 0; i < 1; i++)
        {
            // tracing
            dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
            pathTraceOneBounce << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_materials
                , hst_scene->materials.size()
                , dev_intersections
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();

            depth++;
            
            shadeFakeMaterialTest << <numblocksPathSegmentTracing, blockSize1d >> > (
                iter,
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materials
                );
            
            //accum << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_paths_old, dev_paths);

            //printf("\ndev_paths %d\n", dev_paths[0].color.r);
            //thrust::remove_if(thrust::host, dev_paths, dev_paths + num_paths, is_zero_bounce());
            //dev_paths -= num_paths;
            //printf("\ndev_paths %d\n", dev_paths[0].color.r);

            //}
            //if (depth > 2)
            {
                iterationComplete = true; // TODO: should be based off stream compaction results.
                //depth = 0;
            }
            //}

        }
        // Assemble this iteration and apply it to the image
        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        finalGather << <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);
        
    }
    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
*/