#pragma once

#include <c10/impl/InlineDeviceGuard.h>

namespace c10 {


class DeviceGuard {
public:
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit DeviceGuard() = delete;

  /// Set the current device to the passed Device.
  explicit DeviceGuard(Device device) : guard_(device) {}

  /// This constructor is for testing only.
  explicit DeviceGuard(Device device, const impl::DeviceGuardImplInterface* impl) : guard_(device, impl) {}

  /// Copy is disallowed
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

  /// Move is disallowed, as DeviceGuard does not have an uninitialized state,
  /// which is required for moves on types with nontrivial destructors.
  DeviceGuard(DeviceGuard&& other) = delete;
  DeviceGuard& operator=(DeviceGuard&& other) = delete;


  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }

  /// This method is for testing only.
  void reset_device(at::Device device, const impl::DeviceGuardImplInterface* impl) {
    guard_.reset_device(device, impl);
  }

  /// Sets the device index to the given one.  The device type is inferred
  /// from the original device type the guard was constructed with.
  void set_index(DeviceIndex index) {
    guard_.set_index(index);
  }

  /// Returns the device that was set at the time the guard was constructed.
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device.
  Device current_device() const {
    return guard_.current_device();
  }

private:
  impl::InlineDeviceGuard<impl::VirtualGuardImpl> guard_;
};


class OptionalDeviceGuard {
public:
  /// Create an uninitialized guard.  Set the guard later using set_device.
  explicit OptionalDeviceGuard() : guard_() {}

  /// Initialize the guard, setting the current device to the passed Device.
  explicit OptionalDeviceGuard(Device device) : guard_(device) {}

  /// Initialize the guard if a Device is passed; otherwise leave the
  /// guard uninitialized.
  explicit OptionalDeviceGuard(optional<Device> device) : guard_(device) {}

  /// Constructor for testing only.
  explicit OptionalDeviceGuard(Device device, const impl::DeviceGuardImplInterface* impl) : guard_(device, impl) {}

  /// Copy is disallowed
  OptionalDeviceGuard(const OptionalDeviceGuard&) = delete;
  OptionalDeviceGuard& operator=(const OptionalDeviceGuard&) = delete;

  /// Move is disallowed
  /// See Note [Explicit initialization of optional fields]
  /// and // Note [Move construction for RAII guards is tricky]
  /// for rationale.
  OptionalDeviceGuard(OptionalDeviceGuard&& other) = delete;
  OptionalDeviceGuard& operator=(OptionalDeviceGuard&& other) = delete;

  /// Sets the device to the given one.  The specified device must be consistent
  /// with the device type originally specified during guard construction.
  void reset_device(at::Device device) {
    guard_.reset_device(device);
  }

  /// For testing only
  void reset_device(at::Device device, const impl::DeviceGuardImplInterface* impl) {
    guard_.reset_device(device, impl);
  }

  /// Returns the device that was set at the time the guard was constructed.
  optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device.
  optional<Device> current_device() const {
    return guard_.current_device();
  }

private:
  impl::InlineOptionalDeviceGuard<impl::VirtualGuardImpl> guard_;
};



} // namespace c10
