#pragma once

// 这是一个实现整数除法向负无穷取整的模板函数，其核心算法通过余数校正来实现特定舍入方向
template<typename T>
static inline T div_rtn(T x, T y) {
    int q = x/y;
    int r = x%y;
    if ((r!=0) && ((r<0) != (y<0))) --q;
    return q;
}

