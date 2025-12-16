import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    # Load true labels and predictions
    # Note: test.csv was created from wesad_window_features.csv in a previous step, so it has 'label'
    if not pd.io.common.file_exists('test.csv'):
        print("test.csv not found.")
        return
        
    test_df = pd.read_csv('test.csv')
    
    if not pd.io.common.file_exists('test_with_predictions.csv'):
        print("test_with_predictions.csv not found.")
        return
        
    preds_df = pd.read_csv('test_with_predictions.csv')

    # Merge predictions into the test set
    # Assuming order is preserved (it is if we just appended columns)
    test_df['predicted_stress'] = preds_df['predicted_stress']

    # Extract true and predicted labels
    # UPDATED: Model now targets Label 2 (Stress)
    # Label 2 -> 1 (Stress), everything else -> 0 (Non-Stress)
    y_true = (test_df['label'] == 2).astype(int)
    y_pred = test_df['predicted_stress']

    # Generate evaluation metrics
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['non-stress', 'stress'], zero_division=0))

    # Confusion Matrix
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Display Confusion Matrix as a heatmap
    labels = ['non-stress', 'stress']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Model Target: Label 2 = Stress)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png")
    # plt.show() # Cannot show UI in this environment

    # Save detailed comparison CSV
    test_df['binary_target'] = y_true
    test_df.to_csv("test_with_true_and_predicted.csv", index=False)
    print("\nEvaluation complete. Saved results to 'test_with_true_and_predicted.csv' and 'confusion_matrix_test.png'")

if __name__ == "__main__":
    main()
