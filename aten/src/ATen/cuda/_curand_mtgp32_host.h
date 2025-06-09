#include <curand.h>  // CUDA随机数生成库头文件

// 声明在libcurand_static.a中定义的函数
// 这样做的目的是为了避免在静态链接cudarand时(Caffe2和ATen同时链接)出现多重定义错误
#if CAFFE2_STATIC_LINK_CUDA()  // 如果是静态链接CUDA的情况

// 生成MTGP32(Mersenne Twister GPU)算法的常量参数
// params: MTGP32快速参数数组
// p: 输出的内核参数
curandStatus_t curandMakeMTGP32Constants(
    const mtgp32_params_fast_t params[],
    mtgp32_kernel_params_t * p);

// 初始化MTGP32状态
// state: 状态数组
// para: MTGP32参数
// seed: 随机数种子
void mtgp32_init_state(
    unsigned int state[],
    const mtgp32_params_fast_t *para,
    unsigned int seed);

// 创建MTGP32内核状态
// s: 输出的MTGP32状态
// params: MTGP32参数数组
// k: 内核参数
// n: 参数数量
// seed: 随机数种子
curandStatus_t CURANDAPI curandMakeMTGP32KernelState(
    curandStateMtgp32_t *s,
    mtgp32_params_fast_t params[],
    mtgp32_kernel_params_t *k,
    int n,
    unsigned long long seed);

// MTGP32预定义的快速参数(11213位变体)
extern mtgp32_params_fast_t mtgp32dc_params_fast_11213[];

// 通过数组初始化MTGP32状态
// state: 状态数组
// para: MTGP32参数
// array: 初始化数组
// length: 数组长度
int mtgp32_init_by_array(
    unsigned int state[],
    const mtgp32_params_fast_t *para,
    unsigned int *array, int length);

// 通过字符串初始化MTGP32状态
// state: 状态数组
// para: MTGP32参数
// array: 初始化字符串
int mtgp32_init_by_str(
    unsigned int state[],
    const mtgp32_params_fast_t *para,
    unsigned char *array);

// MTGP32 11213位变体的参数数量
extern const int mtgpdc_params_11213_num;

#else // 如果不是静态链接CUDA的情况(动态链接)

// 包含MTGP32的主机端实现头文件
#include <curand_mtgp32_host.h>
// 包含MTGP32 11213位变体的预定义参数头文件
#include <curand_mtgp32dc_p_11213.h>

#endif // CAFFE2_STATIC_LINK_CUDA