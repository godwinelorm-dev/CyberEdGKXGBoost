# XAI-CyberEd: GK-XGBoost for Cybersecurity Risk Profiling at KNUST

**Researcher:** Koku Godwin Komla
**Supervisor:** Dr Eric Opoku Osei
**Institution:** Kwame Nkrumah University of Science and Technology (KNUST), Ghana
**Target Journal:** Computers & Education: AI (Elsevier, Q1)

---

## Repository Structure

```
XAI_CyberEd_GK_XGBoost/
├── code/
│   ├── GK_XGBoost_Reengineering.py       # Main reengineering experiment (KNUST field data)
│   ├── Simulation_GK_XGBoost.py          # Simulation on WUSTL-EHMS-2020 + KNUST validation
│   ├── SHAP_RoleDifferentiated.py        # Role-differentiated SHAP explanations
│   └── Generate_Results_Figures.py       # Box plots, confusion matrices, Cohen's d
├── figures/
│   ├── methods/                          # Figures 1-4 for Methods section
│   └── results/                          # Figures 1-4 for Results section
└── documents/
    ├── Methods_Section_Final_v2.docx     # Q1 Methods section (2. Materials and Methods)
    ├── Results_Section_Final.docx        # Q1 Results section (3. Results and Analysis)
    └── Tables_and_Formulas.docx          # All tables + LaTeX formulas
```

---

## Three Confirmed Contributions

1. **Original KNUST field data** — HAIS-Q survey, n=1,503 students, first cybersecurity
   behavioural dataset from a Ghanaian university
2. **XGBoost with SHAP** — modern interpretable ML model with role-differentiated explanations
3. **GK-XGBoost** — focal loss + min_child_weight modification, outperforms baseline on
   all metrics across all three evaluation experiments

---

## Key Results

### KNUST Field Data Reengineering (5 seeds x 5 folds = 25 measurements)
| Metric       | Baseline    | GK-XGBoost  | Delta   | p-value |
|--------------|-------------|-------------|---------|---------|
| Accuracy     | 0.9301      | 0.9396      | +0.0094 | 0.0006  |
| Macro F1     | 0.9008      | 0.9130      | +0.0123 | 0.0003  |
| AUC-ROC      | 0.9907      | 0.9933      | +0.0025 | 0.0000  |
| AUC-PR       | 0.9744      | 0.9816      | +0.0072 | 0.0000  |

### WUSTL-EHMS-2020 Public Benchmark (5 seeds x 5 folds)
| Metric       | Baseline    | GK-XGBoost  | Delta   | p-value |
|--------------|-------------|-------------|---------|---------|
| Accuracy     | 0.9382      | 0.9532      | +0.0150 | 0.0000  |
| Macro F1     | 0.7280      | 0.8394      | +0.1114 | 0.0000  |
| AUC-ROC      | 0.9598      | 0.9823      | +0.0225 | 0.0000  |
| AUC-PR       | 0.8602      | 0.9158      | +0.0555 | 0.0000  |

### KNUST Field Validation (5 seeds x 5 folds)
| Metric       | Baseline    | GK-XGBoost  | Delta   | p-value |
|--------------|-------------|-------------|---------|---------|
| Accuracy     | 0.9329      | 0.9403      | +0.0073 | 0.0047  |
| Macro F1     | 0.9025      | 0.9149      | +0.0123 | 0.0023  |
| AUC-ROC      | 0.9904      | 0.9929      | +0.0025 | 0.0001  |
| AUC-PR       | 0.9725      | 0.9806      | +0.0081 | 0.0001  |

---

## Model Parameters

- **gamma** = 2.2151 (KNUST) | 2.0515 (WUSTL)
- **alpha** = 0.7183 (KNUST) | 0.9130 (WUSTL)
- **min_child_weight** = 1 (both datasets)
- **n_estimators** = 500 | **max_depth** = 6 | **learning_rate** = 0.05
- **Seeds** = [42, 123, 456, 789, 1024] | **Folds** = 5

---

## How to Run

1. Upload your KNUST Excel file to Google Colab
2. Run `GK_XGBoost_Reengineering.py` first
3. Upload `wustl-ehms-2020_with_attacks_categories.csv`
4. Run `Simulation_GK_XGBoost.py`
5. Run `SHAP_RoleDifferentiated.py`
6. Run `Generate_Results_Figures.py`

---

## Ethics

Approved by KNUST Human Research, Publications and Ethics Committee
(Reference: HURRRESEC/AP/[PENDING]/26).
Raw survey data cannot be shared publicly due to participant privacy.
Access may be requested from the corresponding author.

---

## Citation

> Komla, K.G. & Osei, E.O. (2026). GK-XGBoost: A Focal-Loss-Augmented
> Gradient Boosting Framework for Cybersecurity Risk Profiling in Ghanaian
> Universities. *Computers & Education: Artificial Intelligence*, [under review].
