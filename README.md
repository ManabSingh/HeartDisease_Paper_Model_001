## Project Pipeline
flowchart TD
    %% Global Styling
    classDef phaseBox fill:#f4f6f8,stroke:#2c3e50,stroke-width:2px,color:#2c3e50,rx:10px,ry:10px;
    classDef script fill:#ffffff,stroke:#34495e,stroke-width:1px,color:#34495e;
    classDef modelNode fill:#d1e8ff,stroke:#2980b9,stroke-width:2px,font-weight:bold,color:#2c3e50;

    %% Phase 1
    subgraph P1 [Phase 1: Environment Setup]
        A1(Setting_Up_Folders.py):::script
    end

    %% Phase 2
    subgraph P2 [Phase 2: Data Preprocessing]
        direction TB
        B1(statlog_cleaning.py) --> B2(missing_value_imputation.py)
        B2 --> B3(outlier_removal.py)
        B3 --> B4(encoding.py)
        B4 --> B5(scaling.py)
        B5 --> B6(smoteenn_balancing.py)
        class B1,B2,B3,B4,B5,B6 script;
    end

    %% Phase 3
    subgraph P3 [Phase 3: Cross-Domain Alignment]
        C1(feature_harmonization.py):::script
    end

    %% Phase 4
    subgraph P4 [Phase 4: Hyperparameter Tuning]
        direction LR
        D1(optuna_lightgbm.py):::script
        D2(optuna_random_forest.py):::script
        D3(optuna_tuning_xgboost.py):::script
    end

    %% Phase 5
    subgraph P5 [Phase 5: Ensemble Training]
        E1(train_lightgbm.py):::script
        E2(train_random_forest.py):::script
        E3(train_xgboost.py):::script
        E4{stacking_model.py}:::modelNode
        E1 --> E4
        E2 --> E4
        E3 --> E4
    end

    %% Phase 6 & 7
    subgraph Eval [Phase 6 & 7: Independent Evaluation]
        direction LR
        subgraph P6 [Source Baseline: Cleveland]
            F1(evaluate_cleveland.py):::script
            F2(confusion_matrix_cleveland.py):::script
            F3(roc_curve_cleveland.py):::script
        end
        subgraph P7 [Target Inference: Statlog]
            G1(final_statlog_inference.py & metrics.py):::script
            G2(confusion_matrix_statlog.py):::script
            G3(roc_curve_analysis.py):::script
        end
    end

    %% Phase 8
    subgraph P8 [Phase 8: Comparative Evaluation]
        H1(generate_comparative_table.py):::script
        H2(comparative_confusion_matrix.py):::script
        H3(comparative_roc_subplots.py):::script
    end

    %% Phase 9
    subgraph P9 [Phase 9: Explainability & SHAP]
        direction TB
        I1(shap_baseline_cleveland.py):::script
        I2(shap_target_statlog.py):::script
        I3(comparative_shap_subplots.py):::script
        I4(advanced_feature_shift_table.py):::script
        I1 & I2 --> I3 --> I4
    end

    %% Phase 10
    subgraph P10 [Phase 10: Clinical Demo]
        direction LR
        J1(Backend: demo/app.py):::script <-->|API| J2(Frontend: index.html / app.js):::script
    end

    %% Connect the main flow
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P5 --> Eval
    P6 --> P8
    P7 --> P8
    P8 --> P9
    P9 --> P10

    class P1,P2,P3,P4,P5,P6,P7,P8,P9,P10 phaseBox;