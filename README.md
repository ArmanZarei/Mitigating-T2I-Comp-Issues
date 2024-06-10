# Mitigating-T2I-Compositionality-Issues

Training WiCLP sample script

```python
python train_text_emb_projection.py \
    --output_dir="final_experiments/window_based_linear_projection/sd2_1/color" \
    --stable_diffusion_checkpoint="stabilityai/stable-diffusion-2-1" \
    --image_size=768 \
    --evaluation_batch_size=10 \
    --train_batch_size=4 \
    --projector_type="window_aware_linear" \
    --projection_window_size=5 \
    --train_steps=25000 \
    --validation_steps=1000 \
    --wandb_run_name="Color dataset - SD v2.1"
```
