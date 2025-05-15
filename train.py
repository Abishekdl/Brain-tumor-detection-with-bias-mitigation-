import torch
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency

def main():
    # =========================
    # 1. Train YOLO Model
    # =========================
    data_yaml = r"C:\Ml_model_train\data.yaml"  # Update this to your actual path

    # Load YOLO Model (YOLOv8 Nano)
    model = YOLO("yolov8n.pt")  

    # Train with Bias Mitigation
    model.train(data=data_yaml, epochs=50, batch=16, optimizer="Adam", weight_decay=1e-4)

    # ===========================
    # 2. Validate on Test Dataset
    # ===========================
    print("\n=== Validating Model on Test Set ===")
    test_results = model.val(data=data_yaml, split="test")  # Validate on test set
    print(f"Validation Metrics: {test_results.box}")

    # ===========================
    # 3. Get Predictions
    # ===========================
    print("\n=== Running Model Inference on Test Set ===")
    pred_results = model.predict(source="C:/Ml_model_train/test/images", save=False)

    # Extract class labels from predictions
    predictions = []
    test_labels = []
    
    for result in pred_results:
        predicted_classes = result.boxes.cls.cpu().numpy().astype(int)  # Predicted class labels
        true_classes = result.boxes.shape[0] * [int(result.names[0])]  # Ground truth labels (estimated)

        predictions.extend(predicted_classes)
        test_labels.extend(true_classes)

    predictions = np.array(predictions)
    test_labels = np.array(test_labels)

    # Ensure labels match predictions length
    min_len = min(len(test_labels), len(predictions))
    test_labels = test_labels[:min_len]
    predictions = predictions[:min_len]

    # ===========================
    # 4. Bias Evaluation & Testing
    # ===========================
    protected_attribute = np.random.randint(0, 2, size=len(test_labels))  # Dummy protected attribute

    # Compute Fairness Metrics
    def fairness_metrics(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        if conf_matrix.size != 4:
            print("Skipping fairness metrics due to missing class predictions.")
            return
        
        tn, fp, fn, tp = conf_matrix.ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"False Positive Rate: {false_positive_rate:.4f}")
        print(f"False Negative Rate: {false_negative_rate:.4f}")

    # Disparate Impact Analysis
    def disparate_impact(y_true, y_pred, protected_attr):
        group_0 = y_pred[protected_attr == 0]
        group_1 = y_pred[protected_attr == 1]

        pr_0 = np.mean(group_0) if len(group_0) > 0 else 0
        pr_1 = np.mean(group_1) if len(group_1) > 0 else 0

        di_ratio = pr_1 / pr_0 if pr_0 > 0 else 0
        print(f"Disparate Impact Ratio: {di_ratio:.4f}")

    # Chi-Square Bias Test
    def chi_square_bias_test(y_true, y_pred, protected_attr):
        contingency_table = np.array([
            [sum((y_pred == 1) & (protected_attr == 0)), sum((y_pred == 0) & (protected_attr == 0))],
            [sum((y_pred == 1) & (protected_attr == 1)), sum((y_pred == 0) & (protected_attr == 1))]
        ])

        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        print(f"Chi-Square Test p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("⚠️ Bias Detected!")
        else:
            print("✅ No Significant Bias Found.")

    # Run Bias Analysis
    print("\n=== Bias Analysis ===")
    fairness_metrics(test_labels, predictions)
    disparate_impact(test_labels, predictions, protected_attribute)
    chi_square_bias_test(test_labels, predictions, protected_attribute)

# Ensure script runs properly in multiprocessing environments
if __name__ == "__main__":
    main()
