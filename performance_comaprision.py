# performance_comparison.py

import matplotlib.pyplot as plt

def plot_comparison(dataset_sizes, accuracies):
    """
    Plot performance comparison between full dataset and distilled dataset.
    """
    plt.plot(dataset_sizes, accuracies, marker='o')
    plt.xlabel('Dataset Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Dataset Size')
    plt.show()

def compare_performance(full_model, distilled_model, full_loader, distilled_loader, device='cuda'):
    """
    Compare performance metrics between the full dataset and distilled dataset.
    """
    accuracy_full, _, _, _ = evaluate_model(full_model, full_loader, device)
    accuracy_distilled, _, _, _ = evaluate_model(distilled_model, distilled_loader, device)

    print(f"Full Dataset - Accuracy: {accuracy_full}")
    print(f"Distilled Dataset - Accuracy: {accuracy_distilled}")
    
    # Plot the comparison
    plot_comparison([len(full_loader.dataset), len(distilled_loader.dataset)], [accuracy_full, accuracy_distilled])
