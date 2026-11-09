# Preservation and cleanup

The selected checkpoint is stage-3 update 128. The inference export hash and ancestor identities are recorded in provenance/LINEAGE.json. Existing pretrained masters, parent 2000, parent 1800, prior attempt-5 checkpoints and checkpoint 167 remain protected locally; preparing this release does not delete or modify them.

The GitHub source repository excludes the large merged weight tensor. Hugging Face holds the inference weight artifact. Verify both package manifests and load the independently downloaded artifact before treating a public upload as complete. Retain local originals until that verification has succeeded. Optimizer state and raw corpora are not backed up by the inference release.
