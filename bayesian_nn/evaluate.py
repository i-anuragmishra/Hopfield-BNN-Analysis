# Main functions for evaluatin bayesian neural nets uncertainity
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleBayesianCNN
import os
import seaborn as sns
from scipy.stats import entropy

def predict_with_uncertainty(model, x, n_samples=20):
    mean_pred, epistemic, aleatoric = model.predict_with_uncertainty(x, n_samples)
    return mean_pred, epistemic, aleatoric

def analyze_uncertainty_vs_corruption(model, image, device, noise_levels=[0.0, 0.2, 0.4, 0.6, 0.8]):
    results = []
    for noise in noise_levels:
        noisy_image = image + noise * torch.randn_like(image)
        noisy_image = torch.clamp(noisy_image, 0, 1)
        mean_pred, epistemic, aleatoric = predict_with_uncertainty(model, noisy_image.to(device))
        results.append({
            'noise': noise,
            'epistemic': epistemic.detach().cpu().numpy(),
            'aleatoric': aleatoric.detach().cpu().numpy(),
            'confidence': mean_pred.max(dim=1)[0].detach().cpu().numpy(),
            'prediction': mean_pred.argmax(dim=1).detach().cpu().numpy()
        })
    return results

def plot_uncertainty_analysis(results, true_label):
    plt.figure(figsize=(15, 5))
    
    noise_levels = [r['noise'] for r in results]
    epistemic = [r['epistemic'][0] for r in results]
    aleatoric = [r['aleatoric'][0] for r in results]
    confidences = [r['confidence'][0] for r in results]
    predictions = [r['prediction'][0] for r in results]
    
    plt.subplot(1, 3, 1)
    plt.plot(noise_levels, epistemic, '-o', label='Epistemic')
    plt.plot(noise_levels, aleatoric, '-o', label='Aleatoric')
    plt.xlabel('Noise Level')
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty vs. Noise')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(noise_levels, confidences, '-o')
    plt.xlabel('Noise Level')
    plt.ylabel('Confidence')
    plt.title('Model Confidence vs. Noise')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    correct_pred = [1 if p == true_label else 0 for p in predictions]
    plt.plot(noise_levels, correct_pred, '-o')
    plt.xlabel('Noise Level')
    plt.ylabel('Correct Prediction')
    plt.title('Prediction Accuracy vs. Noise')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/bayesian_nn_results/uncertainty_analysis.png")
    plt.close()

def analyze_out_of_distribution(model, device, batch_size=32):
    # Load MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    
    # Create synthetic OOD data (random noise)
    ood_data = torch.randn(batch_size, 1, 28, 28).to(device)
    ood_data = (ood_data - ood_data.mean()) / ood_data.std()
    
    # Collect uncertainties
    with torch.no_grad():  # Disable gradient computation
        # Process in-distribution data
        mnist_images, _ = next(iter(mnist_loader))
        mnist_images = mnist_images.to(device)
        _, epistemic_in, aleatoric_in = predict_with_uncertainty(model, mnist_images)
        total_uncertainty_in = epistemic_in + aleatoric_in
        
        # Process OOD data
        _, epistemic_out, aleatoric_out = predict_with_uncertainty(model, ood_data)
        total_uncertainty_out = epistemic_out + aleatoric_out
        
        # Move results to CPU immediately
        total_uncertainty_in = total_uncertainty_in.cpu()
        total_uncertainty_out = total_uncertainty_out.cpu()
        epistemic_in = epistemic_in.cpu()
        epistemic_out = epistemic_out.cpu()
        aleatoric_in = aleatoric_in.cpu()
        aleatoric_out = aleatoric_out.cpu()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Plot distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(total_uncertainty_in.numpy(), bins=30, alpha=0.5, label='In-distribution')
    plt.hist(total_uncertainty_out.numpy(), bins=30, alpha=0.5, label='OOD')
    plt.xlabel('Total Uncertainty')
    plt.ylabel('Count')
    plt.title('Total Uncertainty Distribution')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(epistemic_in.numpy(), bins=30, alpha=0.5, label='In-distribution')
    plt.hist(epistemic_out.numpy(), bins=30, alpha=0.5, label='OOD')
    plt.xlabel('Epistemic Uncertainty')
    plt.ylabel('Count')
    plt.title('Epistemic Uncertainty Distribution')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(aleatoric_in.numpy(), bins=30, alpha=0.5, label='In-distribution')
    plt.hist(aleatoric_out.numpy(), bins=30, alpha=0.5, label='OOD')
    plt.xlabel('Aleatoric Uncertainty')
    plt.ylabel('Count')
    plt.title('Aleatoric Uncertainty Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/bayesian_nn_results/ood_analysis.png")
    plt.close()
    
    return {
        'avg_uncertainty_in': total_uncertainty_in.mean().item(),
        'avg_uncertainty_out': total_uncertainty_out.mean().item()
    }

def analyze_calibration(model, device, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient computation
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            mean_preds, _, _ = predict_with_uncertainty(model, images)
            confidences, predictions = mean_preds.max(dim=1)
            
            # Move to CPU immediately to free GPU memory
            all_confidences.append(confidences.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    all_confidences = torch.cat(all_confidences)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    reliability_diagram, ece = model.get_calibration_scores(
        all_predictions, all_confidences, all_labels
    )
    
    # Plot reliability diagram
    plt.figure(figsize=(8, 8))
    conf_vals, acc_vals = zip(*reliability_diagram)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    plt.plot(conf_vals, acc_vals, '-o', label=f'Model (ECE={ece:.3f})')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/bayesian_nn_results/calibration.png")
    plt.close()
    
    return ece.item()

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("results/bayesian_nn_results", exist_ok=True)
    
    # Set specific GPU device
    torch.cuda.set_device(16)
    device = torch.device("cuda:16" if torch.cuda.is_available() else "cpu")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = SimpleBayesianCNN(dropout_p=0.2).to(device)
    model.load_state_dict(torch.load("/home/stu9/s14/am2552/DL/bayesian_nn_model.pth", map_location=device, weights_only=True))

    # Reduce batch size for calibration analysis
    BATCH_SIZE = 32  # Reduced from 100 to prevent OOM

    # 1. Basic uncertainty analysis with noise
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    # Get one sample
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    try:
        # Analyze uncertainty vs corruption
        uncertainty_results = analyze_uncertainty_vs_corruption(model, images, device)
        plot_uncertainty_analysis(uncertainty_results, labels.item())
        
        # 2. Out-of-distribution detection analysis
        ood_results = analyze_out_of_distribution(model, device)
        print("\nOut-of-distribution Analysis:")
        print(f"Average uncertainty (in-distribution): {ood_results['avg_uncertainty_in']:.4f}")
        print(f"Average uncertainty (out-of-distribution): {ood_results['avg_uncertainty_out']:.4f}")
        
        # 3. Calibration analysis with smaller batches
        ece = analyze_calibration(model, device)
        print(f"\nCalibration Analysis:")
        print(f"Expected Calibration Error: {ece:.4f}")
        
        print("\nAnalysis Results:")
        print("1. Uncertainty vs. corruption analysis saved as 'uncertainty_analysis.png'")
        print("2. Out-of-distribution analysis saved as 'ood_analysis.png'")
        print("3. Calibration analysis saved as 'calibration.png'")
    
    except RuntimeError as e:
        print(f"Error during analysis: {e}")
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()