# Medical Logic Constraints

> Auto-generated from agents.md — these rules apply to every agent in every task.

1. **NEVER** treat titer columns (TO, TH, AH, BH, OX2, OXK, OX9, A, M) as categorical or continuous floats.
   They are strictly ORDINAL: 1:80=1, 1:160=2, 1:320=3.

2. **NEVER** impute Rickettsia titer NaNs without first creating the `rickettsia_panel_conducted` binary flag.
   The absence of a test is clinical data, not missing data.

3. **NEVER** use raw Accuracy as the primary evaluation metric.
   Always report: Sensitivity, Specificity, PPV, NPV, F1 (macro), AUC-PR, and AUC-ROC.

4. **NEVER** use black-box models without a paired SHAP explanation.
   Every model output shown to the user must include a SHAP waterfall plot.

5. **ALWAYS** verify that monotonicity constraints are respected:
   If titer value increases → predicted probability of disease must NOT decrease.

6. **ALWAYS** check for demographic shortcuts before finalising:
   Run Partial Dependence Plots (PDP) on Age and Gender to confirm the model
   is using antigen signals — not just demographic patterns.

7. The patient **Name column is encrypted**. Drop it immediately. Never use it.
