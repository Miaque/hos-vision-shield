# CPU detection models

The CPU backend uses only object detection. It does not load face embeddings,
license-plate recognition/classification networks, character dictionaries, or OCR.

- YuNet face detector: OpenCV Zoo / Shiqi Yu, MIT. The complete license is in
  `licenses/YuNet-MIT.txt`.
- HyperLPR3 320 plate detector: HyperInspire / szad670401/HyperLPR, Apache-2.0.
  The complete license is in `licenses/HyperLPR-Apache-2.0.txt`.
  RGB letterboxing and detection tensor decoding follow the upstream
  `Prj-Python/hyperlpr3/inference/multitask_detect.py` model contract.

Exact source revisions, original download URLs, sizes and hashes are recorded in
`cpu-manifest.json`. Model binaries remain local and are excluded from Git.
