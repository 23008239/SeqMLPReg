from dataset_baseline import TEST_ROOT, TRAIN_ROOT, OUT_ROOT, train_regressor, write_submission_txts


if __name__ == "__main__":
    model_path = OUT_ROOT / "minimal_model.pt"
    result_dir = OUT_ROOT / "result"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("[1/3] training minimal baseline on train set...")
    train_regressor(TRAIN_ROOT, model_path, epochs=80)

    print("[2/3] writing submission txt files in required format...")
    write_submission_txts(TEST_ROOT, result_dir)

    print("[3/3] finished")
    print(f"model: {model_path}")
    print(f"submission_dir: {result_dir}")
