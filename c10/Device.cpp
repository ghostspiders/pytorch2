#include <c10/Device.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <exception>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace c10 {
namespace {
DeviceType parse_type(const std::string& device_string) {
  static const std::array<std::pair<std::string, DeviceType>, 7> types = {{
      {"cpu", DeviceType::CPU},
      {"cuda", DeviceType::CUDA},
      {"mkldnn", DeviceType::MKLDNN},
      {"opengl", DeviceType::OPENGL},
      {"opencl", DeviceType::OPENCL},
      {"ideep", DeviceType::IDEEP},
      {"hip", DeviceType::HIP},
  }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [device_string](const std::pair<std::string, DeviceType>& p) {
        return p.first == device_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  AT_ERROR(
      "Expected one of cpu, cuda, mkldnn, opengl, opencl, ideep, or hip device type at start of device string: ", device_string);
}
} // namespace


Device::Device(const std::string& device_string) : Device(Type::CPU) {
  AT_CHECK(!device_string.empty(), "Device string must not be empty");
  int index = device_string.find(":");
  if (index == std::string::npos) {
    type_ = parse_type(device_string);
    return;
  } else {
    std::string s;
    s = device_string.substr(0, index);
    AT_CHECK(!s.empty(), "Device string must not be empty");
    type_ = parse_type(s);
  }
  std::string device_index = device_string.substr(index + 1);
  try {
    index_ = c10::stoi(device_index);
  } catch (const std::exception&) {
    AT_ERROR(
        "Could not parse device index '",
        device_index,
        "' in device string '",
        device_string,
        "'");
  }
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.type();
  if (device.has_index()) {
    stream << ":" << device.index();
  }
  return stream;
}

} // namespace c10
