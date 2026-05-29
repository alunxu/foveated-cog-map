# Cluster submission scripts

This directory holds **Izar SLURM** submission scripts and **cross-cluster utilities** (data download, environment setup, syncing). Drop-in for any HPC environment that runs the `sbatch` command.

## Sub-categories

| Pattern                | Purpose                                                                  |
|------------------------|--------------------------------------------------------------------------|
| `submit_*.sh`          | Single-job SLURM submitters (training, probing, evaluation)              |
| `submit_*_sweep.sh`    | Multi-condition / multi-checkpoint sweeps                                |
| `auto_resume.sh`       | Detects `TIMEOUT`-ed training jobs and re-submits them                   |
| `common.sh`            | Shared env init sourced by every `submit_*.sh`                           |
| `setup_env.sh`         | One-time SCITAS conda + lab-env setup                                    |
| `sync_to_cluster.sh`   | Push local `Project/` → SCITAS Home                                      |
| `sync_from_cluster.sh` | Pull results back from SCITAS                                            |
| `download_gibson_0plus.sh` | Gibson 0+ episode dataset + MP3D-test fetch                          |
| `regenerate_det_figures.sh` | Re-run paper figures from cached det-rollout NPZs                   |

## Companion private dir

`scripts/cluster_rcp/` (gitignored) contains the RCP-specific equivalents — submission scripts that call `runai-rcp-prod`, inner shell scripts with hard-coded `/scratch/wxu/` paths, and `kubectl` orchestrators tied to the `runai-dhlab-wxu` namespace. They are kept on disk for our own use but never pushed, because they only work on the private K8s/RunAI cluster (RCP) we borrowed for this project.

## Conventions

- Scripts assume `$SLURM_SUBMIT_DIR` is the repository root.
- Output logs land under `slurm_logs/<job_id>.out` (and `.err`).
- Trained checkpoints land under `/scratch/izar/$USER/habitat_checkpoints/<run_name>/ckpt.*.pth`; analysis NPZs land under `/scratch/izar/$USER/habitat_checkpoints/probing/`.
- `*_mig.sh` variants target NVIDIA MIG-partitioned GPUs for shorter probing runs; non-`mig` variants use full GPUs.
