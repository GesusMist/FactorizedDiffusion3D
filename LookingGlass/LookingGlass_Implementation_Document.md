# LookingGlass Implementation Notes

## Files

- `LookingGlass.ipynb`: runnable notebook for generating LookingGlass-style illusions.
- `lookingglass_utils.py`: reusable implementation of UV views, Laplacian Pyramid Warping, SD3/SD3.5 sampling, and VAE residual correction.
- `LookingGlass_extracted.txt`: extracted local text from `../LookingGlass.pdf`.
- `VisualAnagram_extracted.txt`: extracted local text from `../VisualAnagram.pdf`.

## Comparison With Visual Anagrams

Visual Anagrams synchronizes multiple diffusion views by transforming and averaging noisy samples during denoising. This works best when the transformations preserve Gaussian noise statistics, such as flips, rotations by right angles, or permutations. The provided `VisualAnagramCode.ipynb` follows this design with DeepFloyd IF stages I/II and a Stable Diffusion x4 upscaler.

LookingGlass keeps the Visual Anagrams idea of synchronized denoising, but changes the object being synchronized and where synchronization happens:

- It uses latent rectified-flow/diffusion models instead of only pixel-space diffusion.
- At each timestep it predicts the clean final latent, not just the noisy latent.
- It decodes that clean latent to image space before warping, because VAE latents are not equivariant to arbitrary deformations.
- It uses Laplacian Pyramid Warping (LPW) so strongly stretched or compressed mirror/lens views blend at the right frequency level.
- It re-encodes the synchronized image and adds a latent residual correction so the VAE round trip does not wash out colors and contrast.
- It accepts general UV maps, so views can be flips, rotations, conic/cylindrical mirror approximations, lens-like maps, or externally ray-traced mirror/lens maps.

## Model Choice

The notebook defaults to `stabilityai/stable-diffusion-3.5-medium`, matching the main model choice in the LookingGlass paper. This is deliberate:

- SD3/SD3.5 uses FlowMatch-style sampling, which matches the paper's rectified-flow equations.
- The paper reports better behavior from SD3.5 than SD2.1 and notes that SDXL can semantically merge the prompts instead of spatially blending them.
- SD3.5 Medium is a practical quality/runtime compromise compared with larger checkpoints.

Recommended defaults:

- model: `stabilityai/stable-diffusion-3.5-medium`
- text encoders: full SD3.5 stack, including T5-XXL
- resolution: `1024 x 1024`
- inference steps: `30`
- guidance scale: `4.5`
- LPW levels: `6` for image space, `4` for latent residuals
- value-weighted LP average alpha: `0.25` to `0.5`
- time travel: repeat synchronization five times between 20% and 80% of the sampling trajectory, matching the paper's reported `N=5` setting
- identity-view prioritization: final 20% of sampling
- initial noise: independent Gaussian latent per view (`correlated_initial_noise=False`)

The Hugging Face model may require accepting the license and logging in. The notebook intentionally does not store a token.

For Colab, add a read token as a secret named `HF_TOKEN`, then enable notebook access to that secret. The token must belong to the same Hugging Face account that accepted the model's license/access terms at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium. A 401 `GatedRepoError` means either the token is missing, the token is invalid, the notebook cannot read the Colab secret, or the account has not accepted the gated model terms yet.

## Full Configuration And Memory

This version loads the full SD3.5 Medium pipeline, keeps T5-XXL enabled, and generates at 1024 resolution. This is closer to the paper, but memory use can be high because:

- SD3.5 Medium is itself a large transformer pipeline.
- SD3/SD3.5 normally loads a memory-heavy T5-XXL text encoder in addition to CLIP encoders, the transformer, and the VAE.
- Classifier-free guidance doubles the transformer input batch; two views with CFG means four latent streams through the transformer at each step.
- LookingGlass decodes and re-encodes clean latents at every step, so the VAE is active throughout sampling rather than only at the end.
- Inverse LPW uses a differentiable dummy pyramid and gradients, which adds temporary memory beyond the model weights.
- CPU offload is disabled in the full paper-style path. Enabling it reduces VRAM pressure but changes the memory/runtime profile.

If Colab runs out of memory, the runtime is too small for the full configuration. The notebook no longer silently downgrades the method.

## Implementation Details

Diffusers' FlowMatch scheduler runs from high sigma to low sigma. In this implementation, the clean latent estimate is:

```python
clean_latent = latent - sigma * velocity
```

After LookingGlass synchronizes `clean_latent`, the synchronized clean latent is converted back to a velocity:

```python
synced_velocity = (latent - synced_clean_latent) / sigma
```

That velocity is then passed to a FlowMatch Euler transition. This implementation now uses the same transition formula for both normal denoising moves and time-travel jumps.

Time travel follows the paper's "segments of size 1" description. If `sync_repeats=N` and a base timestep is inside the `sync_repeat_start` to `sync_repeat_stop` window, the sampler expands that single step into:

```text
t_i -> t_{i+1} -> t_i -> t_{i+1} ...
```

ending at `t_{i+1}` after `N` denoising passes. With the paper-style default `sync_repeats=5`, each selected step has five denoising passes and four one-step noise-back jumps. This is different from the older helper behavior, which only repeated the synchronization operator at a fixed timestep.

Normal generation also exposes `correlated_initial_noise`, but the default is `False`, matching Algorithm 1's independent Gaussian initialization for each view. Setting it to `True` initializes all views from one UV-transformed base latent, which is useful for Visual-Anagrams-style experiments but is not the paper default for LookingGlass.

LPW is implemented in three parts:

- Forward LPW builds a Gaussian pyramid and samples the nearest appropriate level according to a derivative-based LOD map. Both spatial sampling and LOD-level selection now match the paper's supplementary `grid_sample(..., mode="nearest")` pseudo-code; bilinear sampling or fractional LOD blending is smoother but visibly blurs reflected views.
- Inverse LPW uses the paper's dummy differentiable forward pass to transport target-view pixels back into a canonical Laplacian pyramid.
- Pyramid blending handles partial views with NaN masks and supports value-weighted averaging to preserve stronger details.
- VAE re-encoding uses the encoder mode by default, avoiding extra stochasticity inside the denoising loop.

The pyramid operators use the strict Burt-Adelson style pair: Gaussian blur plus stride-2 subsampling for `pyr_down`, and zero insertion plus the matching scaled blur for `pyr_up`.

The included conic, cylindrical, and faceted-lens UV maps are procedural approximations meant to make the notebook usable immediately. For physically exact anamorphoses, replace these with UV maps rendered from a ray tracer, with normalized target-to-canonical coordinates in `[0, 1]` and `NaN` for undefined pixels.

## Figure 5 Ablation Blocks

The notebook includes a `Figure 5 LPW Ablation` section that generates three complete SD3.5 runs with the same 135-degree inner-rotation view and seed. This custom block removes the Geng/noisy-latent baseline and keeps Laplacian Pyramid Warping active in every variant:

- `Final Estimate + LPW`: synchronization of predicted clean latents using latent-space LPW.
- `+ VAE + LPW`: VAE decode, image-space LPW synchronization, and VAE re-encode.
- `+ Residual Corr. + LPW`: the same VAE LPW path plus latent residual correction.

The helper function is `sample_figure5_lpw_ablation(...)`. It saves images to `LookingGlass/figure5_lpw_ablation/`. It exposes the same main sampler controls as `sample_lookingglass(...)`: `num_inference_steps`, `guidance_scale`, `image_levels`, `latent_levels`, `alpha`, `lpw_sample_mode`, `sync_repeat_start`, `sync_repeat_stop`, `sync_repeats`, `prioritize_view`, `prioritize_fraction`, and `max_sequence_length`. Its `sync_repeats` parameter uses the same paper-style time-travel schedule expansion as the full generator.

The older non-LPW helper, `sample_figure5_ablation(...)`, remains in `lookingglass_utils.py` for reference. It compares Geng-style direct noisy-latent synchronization, final-estimate synchronization, VAE image synchronization, and residual correction.

This ablation helper is inference-only. It uses `torch.no_grad()`, detaches the latent state after each scheduler step, and clears CUDA cache between variants. Without those details, VAE decode/encode and warping operations can accidentally retain an autograd graph across denoising steps and cause extreme CUDA memory growth.

## Quality Gap Versus The Paper

The notebook is a practical implementation, but the procedural mirror/lens UV maps are not the same as the paper's ray-traced mappings. This matters most for cylindrical mirrors: a simple polar unwrap creates an obvious annular support region, while the paper uses view maps generated from the physical setup and then tunes the LOD/mask behavior around that geometry.

If the cylindrical result has a visible soft ring in the identity image or a blurry reflected view, use this order:

1. Set `LPW_SAMPLE_MODE = "nearest"`.
2. Use `HEIGHT = WIDTH = 1024` and keep T5 loaded.
3. Set `SYNC_REPEATS = 5` for the paper's reported time-travel setting, or reduce it only for faster iteration.
4. Try `ALPHA` in `[0.25, 0.5]`; higher preserves more detail but can reveal the hidden view in the identity image.
5. Keep `prioritize_view=0` and `prioritize_fraction=0.20`.
6. For paper-level cylindrical quality, replace the procedural cylinder UV with a ray-traced UV map for the actual camera/cylinder setup.

## Blender Cylinder Placement

The notebook's cylindrical default is now a top-mounted oblique setup:

```python
CYLINDER_CENTER_UV = (0.5, 0.17)
CYLINDER_RADIUS_UV = 0.125
CYLINDER_MIRROR_HEIGHT_UV = 0.30
CYLINDER_MIRROR_Z_MIN_UV = 0.02
CYLINDER_VIEW_ANGLE_DEGREES = 60.0
CYLINDER_VIEW_AZIMUTH_DEGREES = 90.0
CYLINDER_VISIBLE_ARC_DEGREES = 130.0
```

UV coordinates use `u=0` left, `u=1` right, `v=0` top, `v=1` bottom. If your Blender image plane has side length `L`, is centered at the origin, and the top of the image points toward Blender `+Y`, place the cylinder at:

```text
x = (center_u - 0.5) * L = 0
y = (0.5 - center_v) * L = 0.33 * L
z = cylinder_height / 2
radius = 0.125 * L
height = 0.30 * L
```

Examples:

```text
Plane size 2 x 2:
  cylinder location = (0, 0.66, height / 2)
  cylinder radius   = 0.25
  cylinder height   = 0.60

Plane size 10 x 10:
  cylinder location = (0, 3.3, height / 2)
  cylinder radius   = 1.25
  cylinder height   = 3.0
```

Set the Blender camera so the angle between its view direction and the cylinder axis is about 60 degrees. Equivalently, if the cylinder axis is vertical, the camera elevation above the plane is about 30 degrees. Put the camera below/front of the image looking toward `+Y`, because the notebook's default `CYLINDER_VIEW_AZIMUTH_DEGREES = 90.0` assumes the view direction points toward the top of the picture.

The new `make_oblique_cylindrical_mirror_uv(...)` function ray-traces an orthographic view ray against a vertical cylinder and reflects it onto the image plane. This is still simpler than a full Blender/Cycles render, but it is much closer to the physical setup than the older polar unwrap. If your image is rotated on the plane, rotate either the plane texture or the cylinder placement/camera azimuth consistently.

## Cropped Blender Ray-Traced UV Map

For generation, the hidden UV map should represent only the mirror surface, not the whole Blender camera frame. If the camera sees the plane, background, and cylinder together, a full-frame ray-traced UV makes the cylinder occupy only a small part of the hidden-view image. That gives weak, blurry results because most hidden-view pixels are invalid.

Use `export_cropped_cylinder_uv.py` in Blender instead. It reads the active camera, finds pixels that hit `LG_Mirror` and then reflect onto `LG_Plane`, crops that cylinder region, and resamples the crop to a full `1024 x 1024` UV map:

```text
LookingGlass/export_cropped_cylinder_uv.py
```

Expected output, relative to the `.blend` file location:

```text
lookingglass_uv/cylinder_raytraced_uv_cropped.npy
lookingglass_uv/cylinder_raytraced_uv_cropped_metadata.json
```

The notebook ray-traced UV loader now prefers `cylinder_raytraced_uv_cropped.npy` and only falls back to the older full-camera `cylinder_raytraced_uv.npy` for debugging.

## Usage

1. Open `LookingGlass.ipynb`.
2. In Google Colab, run the setup cell once and restart the runtime if Colab asks.
3. Choose prompts and `VIEW_KIND`.
4. Run the sampling cell.
5. Output images are saved in this folder.

## Google Colab Dependency Notes

Colab includes PyTorch and many `google.*` packages by default. Avoid broad upgrades such as `pip install -U protobuf`, because several Colab packages require `protobuf < 6`. The notebook setup cell intentionally uses:

```bash
protobuf>=5.29.1,<6
diffusers==0.37.1
transformers==4.57.6
huggingface_hub>=0.36,<1
safetensors>=0.6.2,<0.8
```

If you already ran the older loose install cell and saw `protobuf 7.x` conflicts, the safest Colab fix is **Runtime -> Disconnect and delete runtime**, then run the updated notebook from the top. If you keep the same VM, rerun the updated setup cell and then use **Runtime -> Restart runtime** before importing model libraries.

The Diffusers warning about Flax classes being deprecated is harmless for this notebook. The implementation uses PyTorch classes, and the notebook pins Diffusers before v1.0.

Prompt tips from the paper:

- Use a place or broad scene for the identity view.
- Use an easily recognizable subject for the mirror/lens view.
- Artistic styles are usually easier than photorealism.
- Shared or low-color styles, such as ink, sketch, or sculpture, often blend better when the prompts have conflicting colors.

## References

- Local source paper: `../LookingGlass.pdf`
- Local comparison slides/code: `../VisualAnagram.pdf`, `../VisualAnagramCode.ipynb`
- SD3.5 Medium model card: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
- Diffusers SD3 pipeline source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
- Diffusers FlowMatch Euler scheduler source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
