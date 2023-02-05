#define _CRT_SECURE_NO_WARNINGS
#include "cstdio"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <cassert>



struct Pixel
{
    unsigned char r, g, b, a;
};
/*
void ConvertImageToGrayCpu(unsigned char* imageRGBA, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
}
*/
__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA)
{
    
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];
    unsigned char pixelValue = (unsigned char)
        (ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
    ptrPixel->r = pixelValue;
    ptrPixel->g = pixelValue;
    ptrPixel->b = pixelValue;
    ptrPixel->a = 255;
}

__global__ void CovertImageInverseGpu(unsigned char* imageRGBA)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];
    ptrPixel->r = 255 - ptrPixel->r;
    ptrPixel->g = 255 - ptrPixel->g;
    ptrPixel->b = 255 - ptrPixel->b;
    ptrPixel->a = 255; 
}


int main(int argc, char** argv)
{
   
    int width, height, componentCount;
    std::cout << "Wczytywanie pliku....";
    system("pause");
    
    unsigned char* imageData = stbi_load("C:/Users/krzys/source/repos/CudaRuntime5/x64/Release/neon.png", &width, &height, &componentCount, 4);
    if (!imageData)
    {
        std::cout << std::endl << "Nie wczytano pliku! (konieczna edycja sciezki i ponowne zbudowanie projektu)"  << std::endl;
        system("pause");
        return -1;
    }
    std::cout << " DONE" << std::endl;

    // Validate image sizes
    if (width % 16 || height % 16)
    {
        // NOTE: Leaked memory of "imageData"
        std::cout << "Width and/or Height is not dividable by 16!";
        return -1;
    }

    
    // Process image on cpu
    //std::cout << "Processing image...";
    //ConvertImageToGrayCpu(imageData, width, height);
    //std::cout << " DONE" << std::endl;
    

    // Copy data to the gpu
    std::cout << "Copy data to GPU...";
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << " DONE" << std::endl;

    // Process image on gpu
    std::cout << "Running CUDA Kernel...";
    dim3 blockSize(16, 16);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    CovertImageInverseGpu << <gridSize, blockSize >> > (ptrImageDataGpu);
    auto err = cudaGetLastError();
    std::cout << " DONE" << std::endl;


    // Copy data from the gpu
    std::cout << "Copy data from GPU...";
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    std::cout << " DONE" << std::endl;

    // Build output filename
    /*
    std::string fileNameOut = argv[1];
    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_gray.png";
    
    
    */
    // Write image back to disk
    std::cout << "Writing png to disk...";
    stbi_write_png("gray.png", width, height, 4, imageData, 4 * width);
    std::cout << " DONE";

    // Free memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);

    system("pause");
}
