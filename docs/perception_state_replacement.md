# Within-family Perception state-replacement analysis

The released identifier-free behavior and state-replacement matrices are under `data/processed/model_outputs/`. Raw terminal records are not bundled.

`scripts/recompute_perception_state_replacement.py` starts from those matrices and independently recomputes the clean-profile correlation, scene-bootstrap intervals, target attenuation, target-minus-absolute-random specificity and exact paired sign-flip tests.

The analysis requires Python 3.11 or later and NumPy 1.26–2.x. It does not require model weights, images or hidden-state tensors because it reproduces the reported statistics from released numerical scores.

Run from the public package root:

```bash
python scripts/recompute_perception_state_replacement.py
```
