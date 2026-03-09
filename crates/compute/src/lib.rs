//! Centralized device selection and hardware acceleration for Sheplet.
//!
//! This crate owns all `candle_core::Device` construction. Other crates
//! should call `compute::device_for(Workload)` instead of building devices directly.

pub mod detect;
pub mod policy;

pub use candle_core::Device;
pub use detect::{best_gpu_or_cpu, probe, DeviceInfo};
pub use policy::{device_for, Workload};
