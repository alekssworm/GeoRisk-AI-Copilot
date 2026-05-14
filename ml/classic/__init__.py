__all__ = ["compare_advanced_scenarios", "predict_advanced_dose", "train_advanced_model"]


def __getattr__(name):
    if name in {"compare_advanced_scenarios", "predict_advanced_dose"}:
        from ml.classic.predict import compare_advanced_scenarios, predict_advanced_dose

        return {
            "compare_advanced_scenarios": compare_advanced_scenarios,
            "predict_advanced_dose": predict_advanced_dose,
        }[name]
    if name == "train_advanced_model":
        from ml.classic.train import train_advanced_model

        return train_advanced_model
    raise AttributeError(name)
