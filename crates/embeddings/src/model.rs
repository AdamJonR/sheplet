use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use tokenizers::Tokenizer;

use crate::download::download_model_files;
use crate::error::{EmbeddingsError, Result};
use crate::normalize::{l2_normalize, mean_pool};
use crate::EMBEDDING_DIM;

/// Maximum number of tokens per input sequence.
const MAX_SEQ_LEN: usize = 256;

/// Batch size for `embed_batch`.
const BATCH_SIZE: usize = 32;

/// An embedding model that wraps all-MiniLM-L6-v2 (BERT-based).
///
/// Produces 384-dimensional L2-normalized embeddings from text inputs.
pub struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    /// Load an embedding model from a local directory.
    ///
    /// The directory must contain:
    /// - `model.safetensors`
    /// - `config.json`
    /// - `tokenizer.json`
    pub fn from_local(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");

        // If flat files exist, use them directly; otherwise resolve through HF cache structure
        if config_path.exists() {
            let weights_path = dir.join("model.safetensors");
            let tokenizer_path = dir.join("tokenizer.json");
            Self::load_from_files(&config_path, &weights_path, &tokenizer_path)
        } else {
            let files = crate::download::resolve_cached_files(dir)?;
            Self::load_from_files(&files.config_json, &files.model_safetensors, &files.tokenizer_json)
        }
    }

    /// Download the model from Hugging Face Hub and load it.
    ///
    /// Files are cached in `cache_dir` for subsequent loads.
    pub fn download_and_load(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let files = download_model_files(cache_dir)?;
        Self::load_from_files(&files.config_json, &files.model_safetensors, &files.tokenizer_json)
    }

    /// Load model from explicit file paths.
    fn load_from_files(
        config_path: &Path,
        weights_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self> {
        let device = Device::Cpu;

        // Load config
        let config_text = std::fs::read_to_string(config_path)?;
        let config: BertConfig = serde_json::from_str(&config_text)?;

        // Load weights
        let weights_bytes = std::fs::read(weights_path)?;
        let vb = VarBuilder::from_buffered_safetensors(weights_bytes, DTYPE, &device)
            .map_err(|e| EmbeddingsError::ModelLoad(format!("failed to load weights: {e}")))?;

        // Build model
        let model = BertModel::load(vb, &config)
            .map_err(|e| EmbeddingsError::ModelLoad(format!("failed to build BertModel: {e}")))?;

        // Load tokenizer
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbeddingsError::Tokenizer(format!("failed to load tokenizer: {e}")))?;

        // Set truncation to MAX_SEQ_LEN.
        // `with_truncation` returns `&mut Self` in tokenizers 0.20.
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_SEQ_LEN,
            ..Default::default()
        }));

        // Set padding to off (we handle it manually for batches)
        tokenizer.with_padding(None);

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Embed a single text string into a 384-dimensional vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        Ok(batch.into_iter().next().expect("batch should have one element"))
    }

    /// Embed a batch of text strings into 384-dimensional vectors.
    ///
    /// Internally processes in chunks of 32 to manage memory.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(BATCH_SIZE) {
            let chunk_embeddings = self.embed_chunk(chunk)?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }

    /// Embed a single chunk (up to BATCH_SIZE texts).
    fn embed_chunk(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Tokenize all texts in the chunk
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(*text, true)
                    .map_err(|e| EmbeddingsError::Tokenizer(format!("encoding error: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        // Find max length in this batch for padding
        let max_len = encodings
            .iter()
            .map(|enc| enc.get_ids().len())
            .max()
            .unwrap_or(0);

        let batch_size = texts.len();

        // Build padded tensors
        let mut input_ids_vec = vec![0_u32; batch_size * max_len];
        let mut token_type_ids_vec = vec![0_u32; batch_size * max_len];
        let mut attention_mask_vec = vec![0_u32; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let type_ids = encoding.get_type_ids();
            let mask = encoding.get_attention_mask();
            let seq_len = ids.len();

            for j in 0..seq_len {
                input_ids_vec[i * max_len + j] = ids[j];
                token_type_ids_vec[i * max_len + j] = type_ids[j];
                attention_mask_vec[i * max_len + j] = mask[j];
            }
            // Remaining positions stay 0 (padding)
        }

        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, max_len), &self.device)?;
        let token_type_ids =
            Tensor::from_vec(token_type_ids_vec, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(attention_mask_vec, (batch_size, max_len), &self.device)?;

        // Forward pass: [batch, seq_len, hidden_dim]
        let embeddings = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling: [batch, hidden_dim]
        let pooled = mean_pool(&embeddings, &attention_mask)?;

        // L2 normalization: [batch, hidden_dim]
        let normalized = l2_normalize(&pooled)?;

        // Convert to Vec<Vec<f32>>
        let result: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| {
                let row = normalized.get(i).unwrap();
                let values: Vec<f32> = row.to_vec1().unwrap();
                debug_assert_eq!(
                    values.len(),
                    EMBEDDING_DIM,
                    "expected {EMBEDDING_DIM}-dim embedding, got {}",
                    values.len()
                );
                values
            })
            .collect();

        Ok(result)
    }

    /// Returns a reference to the underlying device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
