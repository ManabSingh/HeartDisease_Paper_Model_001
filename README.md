## Project Pipeline
flowchart TD
    %% Phase 1: Environment Setup
    subgraph P1 [Phase 1: Environment Setup]
        A1(Setting_Up_Folders.py)
    end

    %% Phase 2: Data Preprocessing
    subgraph P2 [Phase 2: Data Preprocessing]
        direction TB
        B1(statlog_cleaning.py) --> B2(missing_value_imputation.py)
        B2 --> B3(outlier_removal.py)
        B3 --> B4(encoding.py)
        B4 --> B5(scaling.py)
        B5 --> B6(smoteenn_balancing.py)
    end

    %% Phase 3: Cross-Domain Alignment
    subgraph P3 [Phase 3: Cross-Domain Alignment]
        C1(feature_harmonization.py)
    end

    %% Phase 4: Hyperparameter Tuning
    subgraph P4 [Phase 4: Hyperparameter Tuning]
        direction LR
        D1(optuna_lightgbm.py)
        D2(optuna_random_forest.py)
        D3(optuna_tuning_xgboost.py)
    end

    %% Phase 5: Ensemble Training
    subgraph P5 [Phase 5: Ensemble Training]
        E1(train_lightgbm.py)
        E2(train_random_forest.py)
        E3(train_xgboost.py)
        E4{stacking_model.py}
        E1 --> E4
        E2 --> E4
        E3 --> E4
    end

    %% Phase 6 & 7: Independent Evaluation
    subgraph Eval [Phases 6 & 7: Independent Evaluation]
        direction LR
        subgraph P6 [Source Baseline: Cleveland]
            F1(evaluate_cleveland.py)
            F2(confusion_matrix_cleveland.py)
            F3(roc_curve_cleveland.py)
        end
        subgraph P7 [Target Inference: Statlog]
            G1(final_statlog_inference.py & metrics.py)
            G2(confusion_matrix_statlog.py)
            G3(roc_curve_analysis.py)
        end
    end

    %% Phase 8: Comparative Evaluation
    subgraph P8 [Phase 8: Comparative Evaluation]
        H1(generate_comparative_table.py)
        H2(comparative_confusion_matrix.py)
        H3(comparative_roc_subplots.py)
    end

    %% Phase 9: Explainability & SHAP
    subgraph P9 [Phase 9: Explainability & SHAP]
        direction TB
        I1(shap_baseline_cleveland.py)
        I2(shap_target_statlog.py)
        I3(comparative_shap_subplots.py)
        I4(advanced_feature_shift_table.py)
        I1 & I2 --> I3 --> I4
    end

    %% Phase 10: Clinical Demo
    subgraph P10 [Phase 10: Clinical Demo]
        direction LR
        J1(Backend: demo/app.py) <-->|API| J2(Frontend: index.html / app.js)
    end

    %% Main Pipeline Connections
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P5 --> Eval
    P6 --> P8
    P7 --> P8
    P8 --> P9
    P9 --> P10

    %% Global Styling
    classDef phaseBox fill:#f4f6f8,stroke:#2c3e50,stroke-width:2px,color:#2c3e50,rx:10px,ry:10px;
    classDef script fill:#ffffff,stroke:#34495e,stroke-width:1px,color:#34495e;
    classDef modelNode fill:#d1e8ff,stroke:#2980b9,stroke-width:2px,font-weight:bold,color:#2c3e50;

    class P1,P2,P3,P4,P5,P6,P7,P8,P9,P10 phaseBox;
    class A1,B1,B2,B3,B4,B5,B6,C1,D1,D2,D3,E1,E2,E3,F1,F2,F3,G1,G2,G3,H1,H2,H3,I1,I2,I3,I4,J1,J2 script;
    class E4 modelNode;