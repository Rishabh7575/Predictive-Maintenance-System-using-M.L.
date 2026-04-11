def compare_models(baseline_results, xgb_f1):
    print("\n--- Model Comparison ---")
    print(f"Logistic Regression F1: {baseline_results['logistic_f1']:.4f}")
    print(f"Random Forest F1:     {baseline_results['random_forest_f1']:.4f}")
    print(f"XGBoost F1:           {xgb_f1:.4f}")

