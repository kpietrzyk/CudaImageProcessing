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

const char * PATH = "C:/Users/krzys/source/repos/CudaRuntime5/x64/Release/neon.png";

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

__global__ void CovertImageToSepiaGpu(unsigned char* imageRGBA)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];

    int nowy_r = ptrPixel->r * 0.392f + ptrPixel->g * 0.769f + ptrPixel->b * 0.189f;
    int nowy_g = ptrPixel->r * 0.349f + ptrPixel->g * 0.686f + ptrPixel->b * 0.168f;
    int nowy_b = ptrPixel->r * 0.272f + ptrPixel->g * 0.534f + ptrPixel->b * 0.131f;

    nowy_r > 255 ? nowy_r = 255 : nowy_r = nowy_r;
    nowy_g > 255 ? nowy_g = 255 : nowy_g = nowy_g;
    nowy_b > 255 ? nowy_b = 255 : nowy_b = nowy_b;

    ptrPixel->r = nowy_r;
    ptrPixel->g = nowy_g;
    ptrPixel->b = nowy_b;
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

__global__ void CovertImageToBlackAndWhiteGpu(unsigned char* imageRGBA)
{
    int b_or_w_pixel;

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];    

    int avg =(ptrPixel->r + ptrPixel->g + ptrPixel->b) / 3;

   avg >= 100 ? b_or_w_pixel = 255 : b_or_w_pixel = 0;
    
   ptrPixel->r = b_or_w_pixel;
   ptrPixel->g = b_or_w_pixel;
   ptrPixel->b = b_or_w_pixel;
   ptrPixel->a = 255;
}

void invertImageWrapper(unsigned char* imageData, int width, int height) {

    // Copy data to the gpu
    std::cout << "Kopiowanie do GPU.";
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "ZAKONCZONO" << std::endl;


    // Process image on gpu
    std::cout << "KERNEL PRACUJE....";
    dim3 blockSize(16, 16);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    CovertImageInverseGpu << <gridSize, blockSize >> > (ptrImageDataGpu);
    auto err = cudaGetLastError();
    std::cout << "ZAKONCZONO" << std::endl;



    // Copy data from the gpu
    std::cout << "Kopiowanie z GPU...";
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    std::cout << "ZAKONCZONO" << std::endl;


    cudaFree(ptrImageDataGpu);
}

void grayImageWrapper(unsigned char* imageData, int width, int height) {

    // Copy data to the gpu
    std::cout << "Kopiowanie do GPU.";
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "ZAKONCZONO" << std::endl;

    // Process image on gpu
    std::cout << "KERNEL PRACUJE....";
    dim3 blockSize(16, 16);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    ConvertImageToGrayGpu << <gridSize, blockSize >> > (ptrImageDataGpu);
    auto err = cudaGetLastError();
    std::cout << "ZAKONCZONO" << std::endl;


    std::cout << "Kopiowanie z GPU...";
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    std::cout << "ZAKONCZONO" << std::endl;

    cudaFree(ptrImageDataGpu);
}

void sepiaImageWrapper(unsigned char* imageData, int width, int height) {

    // Copy data to the gpu
    std::cout << "Kopiowanie do GPU.";
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "ZAKONCZONO" << std::endl;

    // Process image on gpu
    std::cout << "KERNEL PRACUJE....";
    dim3 blockSize(16, 16);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    CovertImageToSepiaGpu << <gridSize, blockSize >> > (ptrImageDataGpu);
    auto err = cudaGetLastError();
    std::cout << "ZAKONCZONO" << std::endl;


    std::cout << "Kopiowanie z GPU...";
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    std::cout << "ZAKONCZONO" << std::endl;

    cudaFree(ptrImageDataGpu);
}

void blackAndWhiteImageWrapper(unsigned char* imageData, int width, int height) {

    // Copy data to the gpu
    std::cout << "Kopiowanie do GPU.";
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "ZAKONCZONO" << std::endl;

    // Process image on gpu
    std::cout << "KERNEL PRACUJE....";
    dim3 blockSize(16, 16);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    CovertImageToBlackAndWhiteGpu << <gridSize, blockSize >> > (ptrImageDataGpu);
    auto err = cudaGetLastError();
    std::cout << "ZAKONCZONO" << std::endl;


    std::cout << "Kopiowanie z GPU...";
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    std::cout << "ZAKONCZONO" << std::endl;

    cudaFree(ptrImageDataGpu);
}

int main(int argc, char** argv)
{
   
    int width, height, componentCount;
    std::cout << "Wczytywanie pliku....";
    system("pause");
    
    unsigned char* imageData = stbi_load(PATH, &width, &height, &componentCount, 4);
    if (!imageData)
    {
        std::cout << std::endl << "Nie wczytano pliku! (Prawdopobona konieczna edycja sciezki i ponowne zbudowanie projektu)"  << std::endl;
        system("pause");
        return -1;
    }
    std::cout << " DONE" << std::endl;

    // Validate image sizes
    if (width % 16 || height % 16)
    {
        // NOTE: Leaked memory of "imageData"
        std::cout << "Wymary obrazka nie sa podzielne przez 16!";
        return -1;
    }
     
    
    
    // INVERT 
    invertImageWrapper(imageData, width, height);
    // Zapisywanie pliku
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("inverted_neon.png", width, height, 4, imageData, 4 * width);
    std::cout << "ZAKONCZONO :)";
    // Zwolnienie pamięci    
    stbi_image_free(imageData);
    system("pause");



    
    // GRAY
    imageData = stbi_load(PATH, &width, &height, &componentCount, 4);
    grayImageWrapper(imageData, width, height);
    // Zapisywanie pliku
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("gray_neon.png", width, height, 4, imageData, 4 * width);
    std::cout << "ZAKONCZONO :)";
    // Zwolnienie pamięci 
    stbi_image_free(imageData);
    system("pause");

    // SEPIA
    imageData = stbi_load(PATH, &width, &height, &componentCount, 4);
    sepiaImageWrapper(imageData, width, height);
    // Zapisywanie pliku
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("sepia_neon.png", width, height, 4, imageData, 4 * width);
    std::cout << "ZAKONCZONO :)";
    // Zwolnienie pamięci 
    stbi_image_free(imageData);
    system("pause");

    // BLACK AND WHITE
    imageData = stbi_load(PATH, &width, &height, &componentCount, 4);
    blackAndWhiteImageWrapper(imageData, width, height);
    // Zapisywanie pliku
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("bw_neon.png", width, height, 4, imageData, 4 * width);
    std::cout << "ZAKONCZONO :)";
    // Zwolnienie pamięci 
    stbi_image_free(imageData);
    system("pause");


    // RESIZE
}
