"""Trains, tunes, and logs a tourism-package-purchase classifier with MLflow."""
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tourism-wellness-package-prediction")


def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline(Xtrain: pd.DataFrame):
    numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = Xtrain.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    )

    model = XGBClassifier(random_state=42, eval_metric="logloss")

    return make_pipeline(preprocessor, model)


def main():
    Xtrain, Xtest, ytrain, ytest = load_splits()
    pipeline = build_pipeline(Xtrain)

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    with mlflow.start_run():
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)

        y_pred = best_model.predict(Xtest)
        accuracy = accuracy_score(ytest, y_pred)
        f1 = f1_score(ytest, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        print("Best Parameters:", grid_search.best_params_)
        print(classification_report(ytest, y_pred))

        mlflow.sklearn.log_model(best_model, "model")

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        print(f"Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
