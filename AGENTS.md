# Repository Instructions

## Environment
- Use the conda environment `ml_termproj` for all Python commands: `conda run --live-stream -n ml_termproj ...` or activate it first.
- CUDA work is expected on this machine, but the CPU-safe smoke path should still run without CLIP downloads.
- Current verified dependency install path is `pip install -r requirements.txt`; `datasets` is required for Tiny-GenImage prep.

## Project Structure
- `configs/default.yaml` is the primary runtime config for the dataset.csv/.npy/PyTorch workflow.
- `src/data/` owns metadata validation, manifest helpers, dummy data, and local real/fake manifests.
- `src/features/` owns frequency and optional CLIP feature caching.
- `src/train/` owns PyTorch trainers for `frequency_only`, `clip_only`, and `fusion`; `src/eval/` owns evaluation and robustness.
- `src/inference/` contains the shared prediction boundary; the Streamlit entrypoint is `src/app/app.py`.
- `scripts/prepare_genimage_subset.py` materializes `TheKernel01/Tiny-GenImage` to 512x512 files and writes both metadata CSV and manifest-v1 CSV.
- `scripts/train_cuda_finetune.py` trains a torchvision ResNet (`resnet18` default, `resnet50` supported) from a manifest-v1 CSV.
- `scripts/*` wrappers such as `train_classifier.py`, `extract_clip_features.py`, `run_robustness.py`, and `run_all_experiments.py` are deprecated compatibility guidance unless their help text says otherwise.
- `outputs/` contains recent Tiny-GenImage/legacy experiment outputs; `artifacts/` is the README-documented target for the current dataset.csv workflow.

## Do
- Keep label polarity fixed: `real=0`, `fake=1`; all probabilities are `P(fake)`.
- Use 512x512 image size for current data/training work unless explicitly asked otherwise.
- Prefer `python -m src...` commands from the README for the current frequency/CLIP/fusion workflow.
- For Tiny-GenImage full prep, use `scripts/prepare_genimage_subset.py`; default split is deterministic 70/20/10 over all rows.
- For CUDA fine-tuning, validate the manifest first, then run `scripts/train_cuda_finetune.py`; checkpoint output is `best_checkpoint.pt`.
- Run focused verification after changes, then `conda run --live-stream -n ml_termproj python -m pytest -q` when practical.
- Check `docs/plan/plan_0001.md` as historical context only; reconcile it with newer user pivots before treating it as authoritative.

## Don't
- Do not reintroduce CIFAKE-specific files or configs unless the user explicitly reverses the later requirement to remove CIFAKE.
- Do not claim `docs/plan/plan_0001.md` Phase B is complete without evidence for CLIP/frequency/fusion artifacts and robustness outputs.
- Do not use deprecated `scripts/*` wrappers as primary workflows when a `python -m src...` command exists.
- Do not treat `outputs/genimage_tiny_full_finetune/best_checkpoint.pt` as a Streamlit-compatible artifact; it is the CUDA ResNet fine-tune checkpoint.
- Do not download original full GenImage again without explicit approval; it was deleted because it was too large.
- Do not commit data, model outputs, HF caches, or generated large artifacts unless explicitly requested.

## Useful Commands
- Full tests: `conda run --live-stream -n ml_termproj python -m pytest -q`
- Metadata validation: `conda run --live-stream -n ml_termproj python -m src.data.validate_metadata --csv data/metadata/dataset.csv`
- CPU-safe smoke wrapper: `conda run --live-stream -n ml_termproj python scripts/run_all_experiments.py`
- Tiny-GenImage prep: `conda run --live-stream -n ml_termproj python scripts/prepare_genimage_subset.py --clean`
- Tiny-GenImage CUDA training: `conda run --live-stream -n ml_termproj python scripts/train_cuda_finetune.py --manifest outputs/genimage_tiny_full/manifest.csv --output_dir outputs/genimage_tiny_full_finetune --device cuda --image_size 512 --model_arch resnet18 --epochs 6 --batch_size 64 --max_trials 4 --num_workers 8 --seed 42`
- Streamlit demo: `conda run --live-stream -n ml_termproj streamlit run src/app/app.py`

## Commit Attribution
- If the user asks for a commit, include this attribution in the commit message body:

```text
Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)

Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>
```
