"""Demo inference with a trained pipeline (raw SMS in -> ham/spam out).

    python predict.py                     # two built-in example messages
    python predict.py -m "Free entry!!! Text WIN to 87121"
    python predict.py --model naive_bayes -m "Are we meeting at 5 pm today?"

The loaded sklearn Pipeline includes TF-IDF, so NO manual preprocessing is
needed. Predictions are real model outputs - nothing is hardcoded.
"""

import argparse
import sys

import joblib

import config

LABELS = {0: "ham", 1: "spam"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify SMS messages as ham/spam")
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=["logistic_regression", "naive_bayes"],
        help="which trained pipeline to use (default: logistic_regression)",
    )
    parser.add_argument("-m", "--message", help="a custom SMS message to classify")
    args = parser.parse_args()

    model_path = config.MODELS_DIR / f"{args.model}.joblib"
    if not model_path.exists():
        print(
            f"ERROR: {model_path} not found.\n"
            "Train the models first:  python run_pipeline.py   (or run the Airflow DAG)"
        )
        return 1

    pipe = joblib.load(model_path)

    messages = [args.message] if args.message else [
        "Congratulations! You have won a free prize. Call 09061701461 now to claim your £1000 reward!",
        "Are we meeting at 5 pm today?",
    ]

    for msg in messages:
        pred = int(pipe.predict([msg])[0])
        proba = pipe.predict_proba([msg])[0]
        print(f"\nMessage : {msg}")
        print(f"Prediction: {LABELS[pred].upper()}  "
              f"(ham={proba[0]:.3f}, spam={proba[1]:.3f})  [model: {args.model}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
