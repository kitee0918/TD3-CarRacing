# 2025 course development archive

- `oct_2025/`: selected parallel training, throughput benchmark, grid search, and evaluation scripts. The source files were last modified between 2025-10-29 and 2025-11-02 in a local checkout of the 2023 upstream repository. They were uncommitted there, so their dates are file metadata, not historical Git commits.
- `nov_2025/`: scripts carried by the 2025-11-18 to 2025-11-23 course Git history. `optimal_d_csdn.py` produced the logged decay experiments; `optimal_d.py` and `optimal_d_ctn.py` explore other reward and update choices.

These scripts are preserved for provenance and comparison. The maintained entry points are the `td3-carracing-train` and `td3-carracing-eval` commands in the new package. Do not treat checkpoints from one architecture or configuration as interchangeable with another.
