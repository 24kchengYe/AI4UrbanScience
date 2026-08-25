# Qwen2.5-VL-7B Perception replication: numerical state-replacement recomputation

The released identifier-free behavior and state-replacement matrices are under `data/processed/model_outputs/`. Raw terminal records are not bundled.

`scripts/recompute_perception_state_replacement.py` starts from those matrices and independently recomputes the clean-profile correlation, scene-bootstrap intervals, target attenuation, target-minus-absolute-random specificity and exact paired sign-flip tests.

The repository declares Python 3.10 or later and NumPy 1.26–2.x; this workflow is continuously checked on Python 3.11 and was also validated in the frozen CPython 3.13.3 Windows environment. It does not require model weights, images or hidden-state tensors because it recomputes the reported 7B replication statistics from released numerical scores rather than rerunning model inference.

Run from the public package root:

```bash
python scripts/recompute_perception_state_replacement.py
```
