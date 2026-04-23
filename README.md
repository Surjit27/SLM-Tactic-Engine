# Master End-to-End Guide
## AI model training, fields, latency measurement, and Vector DB

## 1) Scope

This guide defines one production-ready workflow for:

- multi-model decision system (`SLM tactical + aim + movement + weapon + fusion`)
- numeric-only model input (`obs[256]`) and output (`action[32]`)
- supervised warm-start plus RL fine-tuning
- latency-first inference under strict budget
- Vector DB-assisted decision support for rare and edge cases

---

## 2) Canonical field schema

## Observation tensor: `obs[256]`

- `obs[0:15]` player state (position, velocity, health, armor, phase, aim)
- `obs[16:65]` enemy features
- `obs[66:75]` cover features
- `obs[76:85]` teammate features
- `obs[86:95]` threat/audio/context indicators
- `obs[96:255]` temporal history window

## Action tensor: `action[32]`

- `action[0:4]` movement controls
- `action[5:8]` aim controls
- `action[9:14]` combat controls
- `action[20:25]` tactical class block (one-hot/logits aligned)
- remaining indices used by fusion output and reserved controls

## Tactical classes (6)

- `0: push`
- `1: hold`
- `2: flank`
- `3: retreat`
- `4: heal`
- `5: ability`

---

## 3) Model pipeline and integration

## 3.1 SLM tactical planner (numeric-only)

- Input: `obs[256]` plus temporal summary from `obs[96:255]`
- No token embedding table (numeric projection only)
- Output: tactical logits `(6,)` and optional tactical context vector
- Responsibility: high-level intent selection

## 3.2 Specialist models

- Aim head: predicts `action[5:8]`
- Movement head: predicts `action[0:4]`
- Weapon head: predicts weapon preference/control logits
- Inputs: `obs[256]`, with tactical context for aim/movement

## 3.3 Fusion head

- Small MLP recommended
- Input: concatenated tactical + aim + movement + weapon outputs
- Output: final coherent `action[32]`
- Responsibility: conflict resolution between heads

---

## 4) End-to-end training design

## Stage A: data generation and validation

1. ingest telemetry and simulator data
2. build `obs[256]`, `action[32]`, tactical labels
3. validate ranges, nulls, shape, and schema version
4. split into train/val/eval datasets

## Stage B: supervised warm-start

- train tactical and specialist heads on offline labels
- core objectives:
  - tactical cross-entropy
  - mixed regression/classification for action heads

## Stage C: RL fine-tuning

- rollout in environment to collect `(s_t, a_t, r_t, s_{t+1})`
- optimize tactical policy (and optional residual heads) with PPO/REINFORCE
- periodically calibrate fusion head after policy updates

## Stage D: promotion and release

- compare challenger vs champion
- pass quality, safety, and latency gates
- deploy using canary progression

---

## 5) RL specification (technical baseline)

## MDP framing

- State: `s_t = obs[256]`
- Action: tactical class (6) plus optional residual controls
- Reward:

`r_t = w1*objective + w2*survival + w3*damage + w4*team - w5*jitter - w6*invalid`

Suggested starting weights:

- `w1=1.0`
- `w2=0.8`
- `w3=0.6`
- `w4=0.4`
- `w5=0.5`
- `w6=1.0`

## PPO starting parameters

- `gamma=0.98`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `entropy_coef=0.01`
- rollout steps/update `2048`
- minibatch size `256`
- epochs/update `4-8`

---

## 6) Latency measurement and control

## 6.1 Runtime latency budget

- target tick: `33ms` (30Hz)
- target model budget: `<15ms` total

## 6.2 What to measure per tick

- `t_obs_build_ms`
- `t_slm_ms`
- `t_aim_ms`
- `t_move_ms`
- `t_weapon_ms`
- `t_fusion_ms`
- `t_postprocess_ms`
- `t_total_model_ms`

## 6.3 Required latency metrics

- p50, p95, p99 of `t_total_model_ms`
- fallback rate due to latency breach
- per-head tail latency contribution

## 6.4 Guardrails

- run aim/movement/weapon in parallel when possible
- pre-allocate buffers (no per-tick heap allocation)
- FP16 inference path
- if `t_total_model_ms > budget`, reuse previous safe action

---

## 7) Vector DB integration (decision support)

Vector DB is optional but valuable for edge-case robustness.

## 7.1 What to store

- compact embeddings of `obs[256]`
- tactical context embeddings
- outcome metadata (win/survival/failure tags)
- rare-case cluster identifiers

## 7.2 Where it connects

1. post-feature pipeline: write embeddings to Vector DB
2. pre-tactical decision: top-k nearest retrieval (`k=8` start)
3. RL sampling: prioritize hard and rare clusters

## 7.3 Retrieval-assisted tactical scoring

For class `c`:

`score[c] = alpha * slm_logit[c] + beta * retrieval_prior[c] - gamma * risk_penalty[c]`

Suggested defaults:

- `alpha=0.70`
- `beta=0.25`
- `gamma=0.05`
- confidence threshold `0.45`

---

## 8) Inference workflow (model-only path)

1. build `obs[256]`
2. compute optional retrieval context from Vector DB
3. run SLM tactical planner
4. run aim/movement/weapon heads (parallel)
5. run fusion head -> `action[32]`
6. apply humanization and safety clamp
7. if latency breach, fallback to previous safe action
8. inject command to runtime

---

## 9) External systems and orchestration

- `AWS S3` for raw and curated data
- `SageMaker` for train and RL jobs
- `API Gateway` + API key auth for orchestration endpoints
- `Secrets Manager` for key storage/rotation
- `CloudWatch` for monitoring and alarms
- `Model Registry/Artifact Store` for champion/challenger versions
- `Vector DB` for retrieval memory
- `UE runtime client` for local inference execution

---

## 10) Governance and gates

Candidate promotion requires:

- quality gate pass (reward/KPI thresholds)
- safety gate pass (invalid action/humanization checks)
- latency gate pass (p95/p99 within budget)
- canary health pass

If any fails, reject candidate and continue refinement.

---

## 12) Run commands

```powershell
python -m pip install -r requirements.txt
python run_e2e.py --method slm --train-samples 5000 --val-samples 1000 --episodes 200 --steps 120 --lr 0.01 --gamma 0.98
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---
<img width="5187" height="3886" alt="diagram-export-23-04-2026-18_47_11" src="https://github.com/user-attachments/assets/bcebe69e-733c-46fc-b7b5-5573ad8c7f48" />

## 13) Final recommendation

Use this default production baseline:

- numeric-only SLM tactical planner
- lightweight specialist heads
- small MLP fusion head
- periodic RL fine-tuning with strict latency gates
- optional Vector DB retrieval for rare-case decision support
