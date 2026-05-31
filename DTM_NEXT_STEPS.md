# DTM (Dynamic Therapeutic Memory) — Next Steps

Paper contribution: **DC-RS-Therapy** — clinically-curated, test-time memory that delays CBT instruction decay and reduces distortion collusion vs. raw RAG memory.

## Current state (already done)

- `our-pipeline/dynamic_cheatsheet.py` (1,475 LOC) — `TherapeuticCheatsheet`, extractors, DC-Cu (`generate_with_cheatsheet`), DC-RS (`generate_with_dynamic_retrieval`, `retrieve_with_chain_of_thought`, `generate_with_cheatsheet_rlm`), dynamic summarization, baselines, `compare_conditions`, `analyze_cheatsheet_evolution`.
- `therapy_dc_rs.ipynb` (25 cells) — 3-condition compare (baseline / mem0 / DC-RS) with viz scaffolding.
- Baselines complete: `memincluded` + `memnotincluded` for `llm_counselor` (PAT-1/2/3 × gpt-4o-mini + llama-3.3-70b) and synthetic (SYN_0001/0002/0004); `therapy_memincluded` for SYN set + elena_vasquez.

## The gap

DTM has **not** been run at parity with the memincluded/memnotincluded sweeps. There is no aggregated DC-vs-baseline analysis, no cheatsheet-utilization metric, no collusion audit on the final cheatsheet.

## Plan

### Phase A — Pilot (½ day)
- [ ] Run `therapy_dc_rs.ipynb` end-to-end on **SYN_0001 + gpt-4o-mini only**. Confirm: 3 conditions complete, cheatsheet evolves, decay curves render. Smoke test before scaling compute.

### Phase B — Match baseline sweep (2–3 days compute)
- [ ] Parameterize notebook → script (`run_dtm_sweep.py`) over: patients {PAT-1, PAT-2, PAT-3, SYN_0001, SYN_0002, SYN_0004} × models {gpt-4o-mini, llama-3.3-70b-instruct} × conditions {baseline, mem0, DC-RS}.
- [ ] Output paths: `chroma_db_therapy_dcrs_<model>_<patient>/`, `cbt_output/dcrs/<model>_<patient>.json`. Schema must match memincluded outputs so the analyzer can join by (patient, model, condition).
- [ ] Decision: commit to **DC-RS only** (drop DC-Cu) — sharper paper, half the compute. H4 already predicts DC-RS wins.

### Phase C — DC-only metrics (½ day)
- [ ] Add `evaluate_cheatsheet_utilization(response, cheatsheet) → {score, strategies_applied, missed_opportunities}` to `our-pipeline/alignment_evaluators.py`.
- [ ] Add `audit_final_cheatsheet_collusion(cheatsheet) → {distortion_count, collusion_score, flagged_items}` — uses the distortion filter prompt already drafted in DC_INTEGRATION_PLAN §3.2.

### Phase D — Cross-condition analysis (1 day)
- [ ] `compare_dc_results.py` loads all `cbt_output/*.json` per (model × patient × condition); computes:
  - Mean CBT (all turns) and **late-turn CBT** (turns ≥ 80) — primary
  - **Decay point** (first turn where CBT < 7) — H1
  - Persona consistency (turn-mean) — H5
  - Collusion % on final cheatsheet — H2
  - Cheatsheet utilization mean — supports DC value
- [ ] Figures: decay curves (cell 20 already wired), collusion bar (DC vs Raw-Mem), utilization-vs-CBT scatter.
- [ ] Paired t-tests across patients per (model, metric).

### Phase E — Write-up scaffolding (parallel)
- [ ] Results table: 6 patients × 2 models × 3 conditions × 5 metrics.
- [ ] Anchor claim depending on results — most likely:
  - "DC-RS-Therapy delays CBT decay by **N turns** vs Raw-Mem (p < 0.05)."
  - "Therapeutic curation reduces distortion-collusion by **X%** vs Raw-Mem."
  - "Effect persists across both proprietary (gpt-4o-mini) and open-weight (llama-3.3-70b) backbones."

## Decisions to make before Phase B
1. **DC-Cu in or out?** Recommend out — saves compute, H4 says DC-RS wins anyway.
2. **Llama in scope?** Cross-model generality is a strong paper move but doubles cost. Recommend yes if Lambda credits available.
3. **Real PAT vs SYN as headline?** PAT-* sessions are longer → stronger decay-point claims. Synthetic gives breadth. Use PAT for headline numbers, SYN for replication.
4. **Compute budget cap?** Set a $ or hour budget before starting Phase B; abort if pilot reveals > budget.
