//! Diagnostic probe: verify gradients flow into LoRA vars.
//! Run: cargo run --release -p finetune --example grad_probe [-- metal]

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::Linear;
use finetune::lora::{LoraConfig, LoraLinear};
use finetune::model_utils::LoraTrainable;

fn probe_lora_linear(device: &Device) -> anyhow::Result<()> {
    let w = Tensor::rand(0.0f32, 1.0f32, &[8, 8], device)?;
    let frozen = Linear::new(w, None);
    let cfg = LoraConfig::default();
    let mut lora = LoraLinear::new(frozen, 8, 8, &cfg, device)?;

    let va = Var::from_tensor(lora.lora_a())?;
    let vb = Var::from_tensor(lora.lora_b())?;
    lora.set_lora_a(va.as_tensor().clone());
    lora.set_lora_b(vb.as_tensor().clone());

    let x = Tensor::rand(0.0f32, 1.0f32, &[4, 8], device)?;
    let y = lora.forward(&x)?;
    let loss = y.sum_all()?;
    let grads = loss.backward()?;
    println!(
        "LoraLinear on {:?}: grad_a={} grad_b={}",
        device,
        grads.get(va.as_tensor()).is_some(),
        grads.get(vb.as_tensor()).is_some()
    );
    Ok(())
}

fn probe_full_model(device: &Device) -> anyhow::Result<()> {
    let model_dir = std::path::Path::new("downloaded-models/Qwen--Qwen2.5-0.5B-Instruct");
    if !model_dir.exists() {
        println!("model dir missing, skipping full-model probe");
        return Ok(());
    }
    let cfg = LoraConfig::default();
    let mut trainer = finetune::Qwen2LoraTrainer::new(model_dir, &cfg, device)?;

    let lora_tensors = trainer.lora_tensors();
    let vars: Vec<Var> = lora_tensors
        .iter()
        .map(Var::from_tensor)
        .collect::<candle_core::Result<Vec<_>>>()?;
    let var_tensors: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
    trainer.set_lora_tensors(&var_tensors);

    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], &[1, 6], device)?;
    trainer.clear_kv_cache();
    let logits = trainer.forward_from(&ids, 0, 2)?;
    println!("logits dtype={:?} shape={:?}", logits.dtype(), logits.dims());
    let loss = logits.to_dtype(DType::F32)?.sum_all()?;
    let grads = loss.backward()?;
    let with_grad = vars
        .iter()
        .filter(|v| grads.get(v.as_tensor()).is_some())
        .count();
    println!(
        "Full Qwen2 model on {:?}: vars={} with_grad={}",
        device,
        vars.len(),
        with_grad
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let use_metal = std::env::args().any(|a| a == "metal");
    let device = if use_metal {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    probe_lora_linear(&device)?;
    probe_full_model(&device)?;
    Ok(())
}
