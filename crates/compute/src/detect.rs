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
        assert!(!info.backend.is_empty());
        #[cfg(not(feature = "metal"))]
        assert!(!info.has_metal);
        #[cfg(not(feature = "cuda"))]
        assert!(!info.has_cuda);
    }

    #[test]
    fn best_gpu_or_cpu_does_not_panic() {
        let device = best_gpu_or_cpu();
        let _ = device;
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_probe() {
        let info = probe();
        // On macOS with metal feature, Metal should be detected
        assert!(info.has_metal, "Metal should be detected on macOS with metal feature");
        assert_eq!(info.backend, "metal");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_probe() {
        let info = probe();
        assert!(info.has_cuda, "CUDA should be detected with cuda feature");
        assert_eq!(info.backend, "cuda");
    }
}
