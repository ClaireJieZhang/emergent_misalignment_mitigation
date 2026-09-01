# MASSIVE/medical composition figure bundle

This bundle is a deterministic rendering of the compact provenance snapshot at `massive_medical_composition_neurips_2026`. It contains:

- `massive_medical_composition_tradeoff_neurips_2026`: main capability-safety tradeoff, A-minus-method forest, and Kalai coverage panels.
- `massive_medical_composition_appendix_neurips_2026`: expanded COLM-style direct-model/method comparison.
- `massive_medical_composition_main_table_neurips_2026` and `massive_medical_composition_contextual_baselines_neurips_2026` in `ai_notes/tables/massive_medical_contextual_baselines_v7`: comprehensive CSV/Markdown plus separate primary and contextual LaTeX tables.

Interpretation constraints:

- All results are exploratory-only.
- Delta-min's one unparseable response is not counted as BAD.
- The main plot's horizontal whiskers are paired-gain intervals shifted by the observed base accuracy; they are not marginal candidate-accuracy intervals.
- The appendix's base/B1/B2/B3 medical results are contextual comparators, not primary-gate inputs.
- The overall status remains `EXPLORATORY_SEQUENTIAL_NO_SUPPORT`.

Primary final-summary SHA-256: `96fb90e4942138c25de57052a062bced8dc397b3e218a938f162bba397344692`.

Generated outputs:

Contextual-baseline rendering:

- Union SFT, equal-weight LoRA merge, and completed Kalai whole-output consensus are purple contextual baselines.
- Tradeoff coordinates use accepted outputs; abstentions remain separate, accepted empty strings are not judged or recoded, and all-request rates remain in the source JSON.
- Contextual baselines do not alter the frozen gate or the overall status.

- **main_figure**
  - `png`: `ai_notes/figures/massive_medical_contextual_baselines_v7/massive_medical_composition_tradeoff_neurips_2026.png`
  - `svg`: `ai_notes/figures/massive_medical_contextual_baselines_v7/massive_medical_composition_tradeoff_neurips_2026.svg`
  - `pdf`: `ai_notes/figures/massive_medical_contextual_baselines_v7/massive_medical_composition_tradeoff_neurips_2026.pdf`
  - `plot_data`: `ai_notes/figures/massive_medical_contextual_baselines_v7/massive_medical_composition_tradeoff_neurips_2026.plot_data.json`
- **appendix_figure**
  - `png`: `ai_notes/figures/massive_medical_contextual_baselines_v7/massive_medical_composition_appendix_neurips_2026.png`
  - `svg`: `ai_notes/figures/massive_medical_contextual_baselines_v7/massive_medical_composition_appendix_neurips_2026.svg`
  - `pdf`: `ai_notes/figures/massive_medical_contextual_baselines_v7/massive_medical_composition_appendix_neurips_2026.pdf`
  - `plot_data`: `ai_notes/figures/massive_medical_contextual_baselines_v7/massive_medical_composition_appendix_neurips_2026.plot_data.json`
- **table**
  - `csv`: `ai_notes/tables/massive_medical_contextual_baselines_v7/massive_medical_composition_main_table_neurips_2026.csv`
  - `markdown`: `ai_notes/tables/massive_medical_contextual_baselines_v7/massive_medical_composition_main_table_neurips_2026.md`
  - `latex`: `ai_notes/tables/massive_medical_contextual_baselines_v7/massive_medical_composition_main_table_neurips_2026.tex`
  - `contextual_latex`: `ai_notes/tables/massive_medical_contextual_baselines_v7/massive_medical_composition_contextual_baselines_neurips_2026.tex`
  - `standalone_latex`: `ai_notes/tables/massive_medical_contextual_baselines_v7/massive_medical_composition_main_table_neurips_2026_standalone.tex`
  - `captions_latex`: `ai_notes/tables/massive_medical_contextual_baselines_v7/massive_medical_composition_figure_captions_neurips_2026.tex`
