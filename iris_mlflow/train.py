import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def main():
    # 1. Load the Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Set up the MLflow experiment
    mlflow.set_experiment("iris_classification")

    # 4. Training and Logging
    print("Starting MLflow run...")
    with mlflow.start_run():
        # Define model parameters
        n_estimators = 100
        max_depth = 5
        random_state = 42

        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state
        })

        # Initialize and train the model
        print("Training RandomForestClassifier...")
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        clf.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Infer model signature implicitly (or we could explicitly define one using infer_signature) 
        # and log the trained model
        print("Logging trained model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=clf, 
            name="random_forest_model", 
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS
        )

    print("\nRun complete! Run 'mlflow ui' in the terminal to view results.")

if __name__ == "__main__":
    main()
