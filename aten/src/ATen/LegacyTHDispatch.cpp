#include <ATen/LegacyTHDispatch.h>

namespace at {

/**
 * 获取全局LegacyTHDispatch单例对象
 * 
 * 实现说明：
 * - 使用Meyer's Singleton模式，保证线程安全的延迟初始化
 * - 静态局部变量在C++11后保证线程安全
 * 
 * 重要警告：
 * - 在静态生命周期对象的析构函数中调用此函数可能导致问题
 *   (参见上方TODO注释)
 * - 因为静态变量析构顺序不确定，可能导致访问已销毁对象
 * 
 * 典型用途：
 * - 提供PyTorch传统TH(Caffe2)后端的分发入口点
 * - 用于维护旧版TH API与新式ATen API的兼容层
 */
LegacyTHDispatch & globalLegacyTHDispatch() {
  // 静态局部变量保证首次调用时构造，程序结束时析构
  static LegacyTHDispatch singleton; 
  return singleton;
}

} // namespace at
