def main():
    print("Loading Spam/Ham email dataset...")

    df = load_data(CSV_PATH)
    df = clean_data(df)
    df = deduplicate(df)

    df_sample = balanced_sample(
        df,
        SAMPLE_N_PER_CLASS,
    )

    X_train, X_test, y_train, y_test = split_data(
        df_sample
    )

    # SVM

    print("\n--- SVM Training ---")

    svm_model, svm_params = train_svm(
        X_train,
        y_train,
    )

    svm_metrics = evaluate_model(
        svm_model,
        X_test,
        y_test,
        svm_params,
    )

    save_model_artifacts(
        MODEL_DIR / "SVM",
        svm_model,
        svm_metrics,
    )
  
    # Logistic Regression

    print("\n--- Logistic Regression Training ---")

    logreg_model, logreg_params = train_logistic_regression(
        X_train,
        y_train,
    )

    logreg_metrics = evaluate_model(
        logreg_model,
        X_test,
        y_test,
        logreg_params,
    )

    save_model_artifacts(
        MODEL_DIR / "LogisticRegression",
        logreg_model,
        logreg_metrics,
    )

    # Shared Samples

    dropdown_samples = prepare_dropdown_samples(
        X_test,
        y_test,
    )

    save_shared_samples(
        MODEL_DIR,
        dropdown_samples,
    )

    print(
        "\nTraining and model serialization "
        "completed successfully!"
    )
