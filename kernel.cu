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
#define BLUR_SIZE 3
#define R 0
#define G 1
#define B 2
#define A 3

#define BLOCK_SIZE 16

typedef struct
{
    int width;
    int height;
    unsigned char* elements;
} Matrix;


// TODO:
// Zmienic sciezke do pliku neon.png
// zbudowac wersje
const char * PATH = "C:/Users/krzys/source/repos/CudaRuntime5/x64/Release/neon.png";
__global__ void ResizeImage(Matrix Am, Matrix C, int32_t src_img_width, int32_t src_img_height, int32_t dest_img_width, int32_t dest_img_height);

struct Pixel
{
    unsigned char r, g, b, a;
};

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

__global__ void ConvertImageToBlurGpu(unsigned char* in, unsigned char* out, int width, int height, int channels, int channel, int cA) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixelValue = 0;
        int pixels = 0;
        if (cA)
            out[row * width * channels + col * channels + A] = in[row * width * channels + col * channels + A];
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    pixelValue += in[curRow * width * channels + curCol * channels + channel];
                    pixels++;
                }
            }
        }
        out[row * width * channels + col * channels + channel] = (unsigned char)(pixelValue / pixels);
    }
}

void invertImageWrapper(unsigned char* imageData, int width, int height) {
       
    std::cout << "Kopiowanie do GPU.....";
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "ZAKONCZONO" << std::endl;

    std::cout << "KERNEL PRACUJE....";
    dim3 blockSize(16, 16);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    CovertImageInverseGpu << <gridSize, blockSize >> > (ptrImageDataGpu);
    auto err = cudaGetLastError();
    std::cout << "ZAKONCZONO" << std::endl;

    std::cout << "Kopiowanie z GPU...";
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    std::cout << "ZAKONCZONO" << std::endl;

    cudaFree(ptrImageDataGpu);
}

void grayImageWrapper(unsigned char* imageData, int width, int height) {
       
    std::cout << "Kopiowanie do GPU.....";
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "ZAKONCZONO" << std::endl;

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

    std::cout << "Kopiowanie do GPU.....";
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "ZAKONCZONO" << std::endl;

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

    std::cout << "Kopiowanie do GPU.....";
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    std::cout << "ZAKONCZONO" << std::endl;

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


void blurImageWrapper(unsigned char* imageData,unsigned char* output, int width, int height) {

    int n = 4; // liczba kanalow
    unsigned char* Dev_Input_Image = NULL;
    unsigned char* Dev_Output_Image = NULL;

    std::cout << "Kopiowanie do GPU.....";
    cudaMalloc((void**)&Dev_Input_Image, sizeof(unsigned char) * height * width * n);
    cudaMalloc((void**)&Dev_Output_Image, sizeof(unsigned char) * height * width * n);    
    cudaMemcpy(Dev_Input_Image, imageData, sizeof(unsigned char) * height * width * n, cudaMemcpyHostToDevice);
    std::cout << "ZAKONCZONO" << std::endl;

    std::cout << "KERNEL PRACUJE....";
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(width / blockSize.x, height / blockSize.y, 1);

    // Blur na odpowiednich kanalach R G B 
    ConvertImageToBlurGpu << <gridSize, blockSize >> > (Dev_Input_Image, Dev_Output_Image, width, height, n, R, 0);
    ConvertImageToBlurGpu << <gridSize, blockSize >> > (Dev_Input_Image, Dev_Output_Image, width, height, n, G, 0);
    ConvertImageToBlurGpu << <gridSize, blockSize >> > (Dev_Input_Image, Dev_Output_Image, width, height, n, B, 1);

    cudaDeviceSynchronize();
    std::cout << "ZAKONCZONO" << std::endl;

    std::cout << "Kopiowanie z GPU...";
    cudaMemcpy(imageData, Dev_Output_Image, sizeof(unsigned char) * height * width * n, cudaMemcpyDeviceToHost);
    std::cout << "ZAKONCZONO" << std::endl;

    cudaFree(Dev_Input_Image);
    cudaFree(Dev_Output_Image);
}


// RESIZE PART


//deklaracja funkcji resize


// wrapper do kerneli
unsigned char* kernel_wrapper(unsigned char* data, unsigned char* resized, int d_w, int d_h)
{

    // tutaj troch� ba�agan, ale tak wszystko powt�rzy�em, �eby by�a jasno�� co czym jest
    int32_t rgb_size = 4; //RGBA

    int32_t src_img_width = 880;
    int32_t src_img_height = 880;
    uint64_t src_size = src_img_height * src_img_width * rgb_size * sizeof(uint8_t);

    int32_t dest_img_width = d_w;
    int32_t dest_img_height = d_h;
    uint64_t dest_size = dest_img_height * dest_img_width * rgb_size * sizeof(uint8_t);

    // tutaj tworzymy nasze struktury z danymi dla cz�� po stronie CPU
    Matrix A_cpu;
    A_cpu.width = src_img_width * rgb_size;
    A_cpu.height = src_img_height;
    A_cpu.elements = data;

    Matrix C_cpu;
    C_cpu.width = dest_img_width * rgb_size;
    C_cpu.height = dest_img_height;
    C_cpu.elements = (uint8_t*)malloc(dest_size);


    // Tutaj po stronie GPU 
    Matrix A_gpu;
    A_gpu.width = A_cpu.width;
    A_gpu.height = A_cpu.height;

    std::cout << "Kopiowanie do GPU.....";
    cudaMalloc(&A_gpu.elements, src_size);
    //kopiujemy dane z pami�� po stronie CPU na pmami�� GPU
    cudaMemcpy(A_gpu.elements, A_cpu.elements, src_size, cudaMemcpyHostToDevice);
    Matrix C_gpu;    C_gpu.width = C_cpu.width;
    C_gpu.height = C_cpu.height;
    cudaMalloc(&C_gpu.elements, dest_size);
    std::cout << "ZAKONCZONO" << std::endl;

    std::cout << "KERNEL PRACUJE....";
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(dest_img_width / threadsPerBlock.x, dest_img_height / threadsPerBlock.y);
    // wykonie operacji na GPU
    ResizeImage << <numBlocks, threadsPerBlock >> > (A_gpu, C_gpu, src_img_width, src_img_height, dest_img_width, dest_img_height);
    std::cout << "ZAKONCZONO" << std::endl;

    std::cout << "Kopiowanie z GPU...";
    // kopiowanie z GPU na CPU
    cudaMemcpy(C_cpu.elements, C_gpu.elements, dest_size, cudaMemcpyDeviceToHost);
    // zwalnienie pami�ci
    cudaFree(A_gpu.elements);
    cudaFree(C_gpu.elements);
    // stb image dzia�a na unsigned char, a przyk�adaowy resize bilinearny co wysy�a�em        // linka jest na uint8, wi�c szybki cast  - to z lenistwa wi�c mo�ecie robi� po swojemu
    resized = (unsigned char*)C_cpu.elements;
    //zwracamy nasz pointer i potem w main.cpp mo�na zwolni� pami�� - ja zapomnia�em, ale      //i tak jak visual ko�czy wykonywa� swoje to wywala to wszystko, wi�c git. Przy            // zwyk�ych projektach, gdzie du�o si� dzieje itp. to pami�tajcie o tym.
    std::cout << "ZAKONCZONO" << std::endl;

    return resized;
}

// tutaj ta g��wna operacja ze strony, gdzie si� interpoluje warto�ci pojedynczych pikseli
__device__ uint8_t biliner(
    const float tx,
    const float ty,
    const uint8_t c00,
    const uint8_t c10,
    const uint8_t c01,
    const uint8_t c11)
{
    const float color = (1.0f - tx) * (1.0f - ty) * (c00 / 255.0) +
        tx * (1.0f - ty) * (c10 / 255.0) +
        (1.0f - tx) * ty * (c01 / 255.0) +
        tx * ty * (c11 / 255.0);

    return (color * 255);
}


__global__
void ResizeImage(
    const Matrix Am,
    const Matrix Cm,
    const int32_t src_img_width,
    const int32_t src_img_height,
    const int32_t dest_img_width,
    const int32_t dest_img_height)
{
    const int RGB_SIZE = 4;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < dest_img_height - 1 && col < dest_img_width - 1)
    {
        const float gx = col * (float(src_img_width) / dest_img_width);
        const int gxi = int(gx) * RGB_SIZE;
        const float gy = row * (float(src_img_height) / dest_img_height);
        const int gyi = int(gy);

        const int c00_index = gyi * Am.width + gxi;
        const uint8_t c00_1 = Am.elements[c00_index]; //R
        const uint8_t c00_2 = Am.elements[c00_index + 1]; //G
        const uint8_t c00_3 = Am.elements[c00_index + 2]; //B

        const int c10_index = gyi * Am.width + (gxi + RGB_SIZE);
        const uint8_t c10_1 = Am.elements[c10_index]; //R
        const uint8_t c10_2 = Am.elements[c10_index + 1]; //G
        const uint8_t c10_3 = Am.elements[c10_index + 2]; //B

        const int c01_index = (gyi + 1) * Am.width + gxi;
        const uint8_t c01_1 = Am.elements[c01_index]; //R
        const uint8_t c01_2 = Am.elements[c01_index + 1]; //G
        const uint8_t c01_3 = Am.elements[c01_index + 2]; //B

        const int c11_index = (gyi + 1) * Am.width + (gxi + RGB_SIZE);
        const uint8_t c11_1 = Am.elements[c11_index]; //R
        const uint8_t c11_2 = Am.elements[c11_index + 1]; //G
        const uint8_t c11_3 = Am.elements[c11_index + 2]; //B
        const float tx = gx - int(gx);
        const float ty = gy - gyi;
        const int C_dest = row * Cm.width + col * RGB_SIZE;

        const uint8_t Cvalue_R = biliner(tx, ty, c00_1, c10_1, c01_1, c11_1);
        Cm.elements[C_dest] = Cvalue_R;

        const uint8_t Cvalue_G = biliner(tx, ty, c00_2, c10_2, c01_2, c11_2);
        Cm.elements[C_dest + 1] = Cvalue_G;

        const uint8_t Cvalue_B = biliner(tx, ty, c00_3, c10_3, c01_3, c11_3);
        Cm.elements[C_dest + 2] = Cvalue_B;

        Cm.elements[C_dest + 3] = 255;

    }
    else if (col == dest_img_width - 1 && row < dest_img_height) {
        const int C_dest = (row)*Cm.width + col * RGB_SIZE;

        Cm.elements[C_dest] = Am.elements[(row)*Am.width + (col - 1) * RGB_SIZE];
        Cm.elements[C_dest + 1] = Am.elements[(row)*Am.width + (col - 1) * RGB_SIZE + 1];
        Cm.elements[C_dest + 2] = Am.elements[(row)*Am.width + (col - 1) * RGB_SIZE + 2];
        Cm.elements[C_dest + 3] = 255;
    }
    else if (col < dest_img_width && row == dest_img_height - 1) {
        const int C_dest = (row)*Cm.width + col * RGB_SIZE;
        Cm.elements[C_dest] = Am.elements[(row - 1) * Am.width + (col)*RGB_SIZE];
        Cm.elements[C_dest + 1] = Am.elements[(row - 1) * Am.width + (col)*RGB_SIZE + 1];
        Cm.elements[C_dest + 2] = Am.elements[(row - 1) * Am.width + (col)*RGB_SIZE + 2];
        Cm.elements[C_dest + 3] = 255;
    }
}

// END OF RESIZE PART


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
    std::cout << "ZAKONCZONO" << std::endl;
   

    // Weryfikacja rozmiaru zdjecia (wielokrotnosc liczby 16)
    if (width % 16 || height % 16)
    {
        std::cout << "Wymary obrazka nie sa wielokrotnoscia liczby 16!" << std::endl;
        return -1;
    }
    
    // FILTRY
    
    // INVERT
    std::cout << "--------------INVERT IMAGE--------------" << std::endl;
    invertImageWrapper(imageData, width, height);
    // Zapisywanie pliku
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("inverted_neon.png", width, height, 4, imageData, 4 * width);
    std::cout << "ZAKONCZONO :)" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    // Zwolnienie pamięci    
    stbi_image_free(imageData);
    system("pause");
       
    // GRAY
    std::cout << "--------------GRAY IMAGE--------------" << std::endl;
    imageData = stbi_load(PATH, &width, &height, &componentCount, 4);
    grayImageWrapper(imageData, width, height);
    // Zapisywanie pliku
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("gray_neon.png", width, height, 4, imageData, 4 * width);
    std::cout << "ZAKONCZONO :)" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    // Zwolnienie pamięci 
    stbi_image_free(imageData);
    system("pause");

    // SEPIA
    std::cout << "--------------SEPIA IMAGE--------------" << std::endl;
    imageData = stbi_load(PATH, &width, &height, &componentCount, 4);
    sepiaImageWrapper(imageData, width, height);
    // Zapisywanie pliku
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("sepia_neon.png", width, height, 4, imageData, 4 * width);
    std::cout << "ZAKONCZONO :)" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    // Zwolnienie pamięci 
    stbi_image_free(imageData);
    system("pause");

    // BLACK AND WHITE
    std::cout << "--------------B&W IMAGE--------------" << std::endl;
    imageData = stbi_load(PATH, &width, &height, &componentCount, 4);
    blackAndWhiteImageWrapper(imageData, width, height);
    // Zapisywanie pliku
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("bw_neon.png", width, height, 4, imageData, 4 * width);
    std::cout << "ZAKONCZONO :)" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    // Zwolnienie pamięci 
    stbi_image_free(imageData);
    system("pause");

    // BLUR
    std::cout << "--------------BLURED IMAGE--------------" << std::endl;
    imageData = stbi_load(PATH, &width, &height, &componentCount, 4);      
    unsigned char* output = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));
    blurImageWrapper(imageData, output, width, height);       
    // Zapisywanie pliku
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("blur_neon.png", width, height, 4, imageData, 4 * width);
    std::cout << "ZAKONCZONO :)" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    // Zwolnienie pamięci 
    stbi_image_free(imageData);
    system("pause");

    // RESIZE
    std::cout << "--------------RESIZED IMAGE--------------" << std::endl;
    // wymiary zdjecia wejsciowego
    int s_w = 880;
    int s_h = 880;

    // tutaj wybieracie czy chcecie zwiekszyc czy zmniejszyc zdjecie
    float res = 0.5;
    // wymiary zdjecia docelowego
    int d_w = int(res * 880.);
    int d_h = int(res * 880.);

    int channels;

    unsigned char* img = stbi_load(PATH, &s_w, &s_h, &channels, 0);

    stbi_write_png("old_image.png", s_w, s_h, channels, img, s_w * channels);

    unsigned char* resized = (unsigned char*)malloc((d_w * d_h * 4) * sizeof(unsigned char));

    resized = kernel_wrapper(img, resized, d_w, d_h);
    std::cout << "Zapisywanie pliku...";
    stbi_write_png("saved_image.png", d_w, d_h, channels, resized, d_w * channels);
    std::cout << "ZAKONCZONO :)" << std::endl;
    system("pause");
}
