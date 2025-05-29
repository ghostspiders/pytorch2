#pragma once

#include <c10/Device.h>

namespace c10 {


using StreamId = int32_t;




class Stream final {
private:
  Device device_;
  StreamId id_;
public:
  explicit Stream(Device device, StreamId id)
    : device_(device)
    , id_(id) {}

  bool operator==(const Stream& other) const noexcept {
    return this->device_ == other.device_ && this->id_ == other.id_;
  }
  bool operator!=(const Stream& other) const noexcept {
    return !(*this == other);
  }

  Device device() const noexcept { return device_; }
  DeviceType device_type() const noexcept { return device_.type(); }
  DeviceIndex device_index() const noexcept { return device_.index(); }
  StreamId id() const noexcept { return id_; }

  // The purpose of this function is to more conveniently permit binding
  // of Stream to and from Python.  Without packing, I have to setup a whole
  // class with two fields (device and stream id); with packing I can just
  // store a single uint64_t.
  //
  // The particular way we pack streams into a uint64_t is considered an
  // implementation detail and should not be relied upon.
  uint64_t pack() const noexcept {
    // Are you here because this static assert failed?  Make sure you ensure
    // that the bitmasking code below is updated accordingly!
    static_assert(sizeof(DeviceType) == 2, "DeviceType is not 16-bit");
    static_assert(sizeof(DeviceIndex) == 2, "DeviceIndex is not 16-bit");
    static_assert(sizeof(StreamId) == 4, "DeviceIndex is not 32-bit");
    // Concat these together into a 64-bit integer
    // See Note [Hazard when concatenating signed integers]
    uint64_t bits =
        static_cast<uint64_t>(static_cast<uint16_t>(device_type())) << 48
      | static_cast<uint64_t>(static_cast<uint16_t>(device_index())) << 32
      | static_cast<uint64_t>(static_cast<uint32_t>(id()));
    return bits;
  }

  static Stream unpack(uint64_t bits) {
    auto stream_id = static_cast<StreamId>(bits) & 0xFFFFFFFFull;
    bits >>= 32;
    auto device_index = static_cast<DeviceIndex>(bits) & 0xFFFFull;
    bits >>= 16;
    auto device_type = static_cast<DeviceType>(bits);
    AT_CHECK(isValidDeviceType(device_type));
    return Stream(Device(device_type, device_index), stream_id);
  }

  // I decided NOT to provide setters on this class, because really,
  // why would you change the device of a stream?  Just construct
  // it correctly from the beginning dude.
};

C10_API std::ostream& operator<<(std::ostream& stream, const Stream& s);

} // namespace c10

namespace std {
  template <>
  struct hash<c10::Stream> {
    size_t operator()(c10::Stream s) const noexcept {
      return std::hash<uint64_t>{}(s.pack());
    }
  };
} // namespace std

namespace at {
  using c10::StreamId;
  using c10::Stream;
}
