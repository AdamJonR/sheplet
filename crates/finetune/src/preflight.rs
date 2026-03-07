use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub cpu_count: usize,
}

#[derive(Debug, Clone)]
pub struct PreflightReport {
    pub hardware: HardwareInfo,
    pub recommended_ram_gb: f64,
    pub is_sufficient: bool,
}

pub fn preflight_check(recommended_ram_gb: f64) -> PreflightReport {
    let sys = System::new_all();

    let total_ram_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let available_ram_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let cpu_count = sys.cpus().len();

    let hardware = HardwareInfo {
        total_ram_gb,
        available_ram_gb,
        cpu_count,
    };

    let is_sufficient = available_ram_gb >= recommended_ram_gb;

    PreflightReport {
        hardware,
        recommended_ram_gb,
        is_sufficient,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preflight_returns_nonzero_values() {
        let report = preflight_check(1.0);
        assert!(report.hardware.total_ram_gb > 0.0);
        assert!(report.hardware.available_ram_gb > 0.0);
        assert!(report.hardware.cpu_count > 0);
    }
}
