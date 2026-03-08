use anyhow::{bail, Result};

/// Validate that a name does not contain path separators or traversal sequences.
pub fn validate_safe_name(name: &str) -> Result<()> {
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        bail!("Name must not contain path separators or '..'");
    }
    if name.is_empty() {
        bail!("Name must not be empty");
    }
    Ok(())
}
