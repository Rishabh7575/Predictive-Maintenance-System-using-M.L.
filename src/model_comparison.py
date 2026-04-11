def compare_models(baseline_results, xgb_f1):
    print("\n--- Model Comparison ---")
    print(f"Logistic Regression F1: {baseline_results['logistic_f1']:.4f}")
    print(f"Random Forest F1:     {baseline_results['random_forest_f1']:.4f}")
    print(f"XGBoost F1:           {xgb_f1:.4f}")

    best = max([
        ("Logistic", baseline_results["logistic_f1"]),
        ("Random Forest", baseline_results["random_forest_f1"]),
        ("XGBoost", xgb_f1)
    ], key=lambda x: x[1])

    print(f"\nBest Model: {best[0]} with F1 = {best[1]:.4f}")