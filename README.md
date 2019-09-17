# speaker verification pytorch

## Getting started

```
python 00-extract_features.py
python 01-train.py --config configs/HBC.yaml
python 02-test.py --config configs/HBC_cos.yaml
```

## References
[1] Heo, Hee-Soo, et al. "End-to-end losses based on speaker basis vectors and all-speaker hard negative mining for speaker verification." arXiv preprint arXiv:1902.02455 (2019).

[2] Nagrani, Arsha, Joon Son Chung, and Andrew Zisserman. "VoxCeleb: A Large-Scale Speaker Identification Dataset." Proc. Interspeech 2017 (2017): 2616-2620.

[3] Chung, Joon Son, Arsha Nagrani, and Andrew Zisserman. "VoxCeleb2: Deep Speaker Recognition." Proc. Interspeech 2018 (2018): 1086-1090.
