#pragma once  // 防止头文件重复包含

// 主要参考自: https://stackoverflow.com/questions/295120/c-mark-as-deprecated

// C++14之后版本(>C++14)的编译器支持
#if defined(__cplusplus) && __cplusplus > 201402L
    // 使用C++标准属性标记弃用
    #define AT_DEPRECATED(function) [[deprecated]] function
#else
    // 非C++14+编译器处理分支
    #if defined(__GNUC__)  // GCC/Clang编译器
        // 使用GCC特有的弃用属性
        #define AT_DEPRECATED(function) __attribute__((deprecated)) function
    #elif defined(_MSC_VER)  // MSVC编译器
        // 使用MSVC特有的弃用声明
        #define AT_DEPRECATED(function) __declspec(deprecated) function
    #else  // 其他不支持的编译器
        // 编译时警告需要自行实现
        #warning "You need to implement AT_DEPRECATED for this compiler"
        // 默认不做任何处理(无弃用效果)
        #define AT_DEPRECATED(function) function
    #endif // defined(__GNUC__)
#endif // defined(__cplusplus) && __cplusplus > 201402L