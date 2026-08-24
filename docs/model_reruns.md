# Model rerun boundary

This release supports analyses of released, identifier-free model outputs. It does not include a supported command for replaying proprietary API calls or raw open-weight inference.

Consequently, the install environment intentionally does not include PyTorch or Transformers. Adding those packages would imply a runnable model-inference workflow that this repository cannot honestly guarantee without model weights, hidden-state tensors, images, provider access, and the complete raw inference environment.

Researchers extending the project should pin PyTorch, Transformers, accelerator libraries, model revisions, prompt records, random seeds, decoding parameters, and hardware details for their own run. New outputs should be written outside the frozen `data/` tree and should receive their own manifest and provenance record.
