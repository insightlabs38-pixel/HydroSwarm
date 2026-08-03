# Four-minute demo script and caption plan

This is a production script; bracketed values must be replaced from the generated golden
scenario and evaluation report. Do not record placeholder metrics.

| Time | Visual | Voice/captions |
|---|---|---|
| 0:00–0:25 | Utility map, one abnormal sensor | “A water-quality alert rarely identifies its source. Sparse evidence, changing flow, and unsafe interventions make the next decision hard.” |
| 0:25–0:50 | Architecture animation | “HydroSwarm runs locally. Hydraulic physics and source signatures combine with a compact graph-time model; uncertainty controls sampling and WNTR verifies every plan.” |
| 0:50–1:25 | Start frozen scenario; candidate map and evidence panel | “These are backend-generated results, not a canned overlay. Initial evidence leaves [N] candidates; classical and neural estimates disagree by [D], so the system changes trust.” |
| 1:25–1:55 | Scout ranking, then sample arrival | “Scout selects [NODE] for [IG] expected information gain. When its simulated lab result arrives, reanalysis contracts the calibrated region from [A] to [B].” |
| 1:55–2:40 | No response, plans A/B, verifier rejection | “Strategist proposes bounded alternatives. The default WNTR path rejects [PLAN] because [CONSTRAINT]. A repaired alternative passes the configured pressure and service checks—but still waits for a person.” |
| 2:40–3:10 | Synchronized charts/table | “Against no response, the verified alternative changes exposure by [X], minimum pressure by [P], and unserved demand by [U]. The interface shows tradeoffs, not one magic score.” |
| 3:10–3:35 | Evidence-changed and explanation panels | “Every sentence is tied to typed evidence: the measurement, posterior revision, sensitivity, simulator outcome, and abstention state. The event history is replayable.” |
| 3:35–4:00 | Results, offline indicator, limitation, closing | “Across [RUNS] seeded evaluations, HydroSwarm reports [MEASURED RESULTS]. It runs with Wi-Fi disabled. This is research decision support—not chemistry identification or autonomous control. Localize the risk. Choose the evidence. Verify the response.” |

## Recording checks

Record one continuous golden-scenario run where practical. Keep the URL/localhost and
offline indicator visible, capture failed and successful verifier results, use 1080p,
provide an `.srt` caption export, and pause long enough for numbers and limitations to be
read. The architecture graphic should animate data → fusion → uncertainty → sampling/plan
→ WNTR → human approval without suggesting that the model controls infrastructure.
