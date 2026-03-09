use candle_core::Device;

/// Summary of available hardware backends.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub has_metal: bool,
    pub has_cuda: bool,
    pub backend: &'static str,
}

/// Probe the runtime environment and report available backends.
pub fn probe() -> DeviceInfo {
    #[allow(unused_mut)]
    let mut info = DeviceInfo {
        has_metal: false,
        has_cuda: false,
        backend: "cpu",
    };

    #[cfg(feature = "metal")]
    {
        if Device::new_metal(0).is_ok() {
            info.has_metal = true;
            info.backend = "metal";
        }
    }

    #[cfg(feature = "cuda")]
    {
        if Device::new_cuda(0).is_ok() {
            info.has_cuda = true;
            info.backend = "cuda";
        }
    }

    info
}

/// Return the best available GPU device, falling back to CPU.
pub fn best_gpu_or_cpu() -> Device {
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return device;
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            return device;
        }
    }

    Device::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_does_not_panic() {
        let info = probe();
        // Without features, should be CPU-only
        assert!(!info.backend.is_empty());
    }

    #[test]
    fn best_gpu_or_cpu_does_not_panic() {
        let device = best_gpu_or_cpu();
        // Without features enabled, this will be CPU
        let _ = device;
    }
}
