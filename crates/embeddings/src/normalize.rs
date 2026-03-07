use candle_core::{Result, Tensor};

/// Perform mean pooling over token embeddings, respecting the attention mask.
///
/// # Arguments
/// * `embeddings` - Tensor of shape `[batch, seq_len, hidden_dim]`
/// * `attention_mask` - Tensor of shape `[batch, seq_len]`
///
/// # Returns
/// Tensor of shape `[batch, hidden_dim]`
pub fn mean_pool(embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // Expand mask from [batch, seq_len] to [batch, seq_len, 1]
    let mask = attention_mask.unsqueeze(2)?;
    // Convert mask to the same dtype as embeddings
    let mask = mask.to_dtype(embeddings.dtype())?;
    // Multiply embeddings by mask: [batch, seq_len, hidden_dim]
    let masked = embeddings.broadcast_mul(&mask)?;
    // Sum over the seq_len dimension: [batch, hidden_dim]
    let summed = masked.sum(1)?;
    // Sum the mask over seq_len: [batch, 1]
    let mask_sum = mask.sum(1)?;
    // Clamp to avoid division by zero
    let mask_sum = mask_sum.clamp(1e-9, f64::MAX)?;
    // Divide to get mean: [batch, hidden_dim]
    summed.broadcast_div(&mask_sum)
}

/// L2-normalize a tensor along the last dimension.
///
/// # Arguments
/// * `x` - Tensor of shape `[batch, dim]`
///
/// # Returns
/// Tensor of shape `[batch, dim]` where each row has unit L2 norm.
pub fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    // x^2
    let x_sq = x.sqr()?;
    // sum over last dim, keep dim: [batch, 1]
    let sum_sq = x_sq.sum_keepdim(candle_core::D::Minus1)?;
    // sqrt
    let norm = sum_sq.sqrt()?;
    // Clamp to avoid division by zero
    let norm = norm.clamp(1e-12, f64::MAX)?;
    // Divide
    x.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_mean_pool_shape() {
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 4;
        let hidden_dim = 8;

        let embeddings =
            Tensor::ones(&[batch, seq_len, hidden_dim], candle_core::DType::F32, &device).unwrap();
        let mask = Tensor::ones(&[batch, seq_len], candle_core::DType::U32, &device).unwrap();

        let result = mean_pool(&embeddings, &mask).unwrap();
        assert_eq!(result.dims(), &[batch, hidden_dim]);
    }

    #[test]
    fn test_mean_pool_values() {
        let device = Device::Cpu;
        // Single sample, 3 tokens, 2-dim embeddings
        // embeddings = [[1, 2], [3, 4], [5, 6]]
        // mask = [1, 1, 0]  (third token is padding)
        // Expected: mean of first two tokens = [(1+3)/2, (2+4)/2] = [2.0, 3.0]
        let embeddings = Tensor::new(
            &[[[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            &device,
        )
        .unwrap();
        let mask = Tensor::new(&[[1_u32, 1, 0]], &device).unwrap();

        let result = mean_pool(&embeddings, &mask).unwrap();
        let values: Vec<f32> = result.squeeze(0).unwrap().to_vec1().unwrap();
        assert!((values[0] - 2.0).abs() < 1e-5);
        assert!((values[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_shape() {
        let device = Device::Cpu;
        let x = Tensor::ones(&[3, 5], candle_core::DType::F32, &device).unwrap();
        let result = l2_normalize(&x).unwrap();
        assert_eq!(result.dims(), &[3, 5]);
    }

    #[test]
    fn test_l2_normalize_unit_norm() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[3.0_f32, 4.0], [1.0, 0.0], [0.0, 5.0]], &device).unwrap();
        let result = l2_normalize(&x).unwrap();

        // Check that each row has L2 norm ~1.0
        let result_sq = result.sqr().unwrap();
        let norms = result_sq
            .sum_keepdim(candle_core::D::Minus1)
            .unwrap()
            .sqrt()
            .unwrap();
        let norms: Vec<Vec<f32>> = norms.to_vec2().unwrap();
        for row_norms in &norms {
            assert!(
                (row_norms[0] - 1.0).abs() < 1e-5,
                "expected norm ~1.0, got {}",
                row_norms[0]
            );
        }
    }

    #[test]
    fn test_l2_normalize_values() {
        let device = Device::Cpu;
        // [3, 4] -> norm=5 -> [0.6, 0.8]
        let x = Tensor::new(&[[3.0_f32, 4.0]], &device).unwrap();
        let result = l2_normalize(&x).unwrap();
        let values: Vec<f32> = result.squeeze(0).unwrap().to_vec1().unwrap();
        assert!((values[0] - 0.6).abs() < 1e-5);
        assert!((values[1] - 0.8).abs() < 1e-5);
    }
}
