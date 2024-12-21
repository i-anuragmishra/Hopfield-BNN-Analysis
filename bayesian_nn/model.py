import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleBayesianCNN(nn.Module):
    def __init__(self, dropout_p=0.2):
        super(SimpleBayesianCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout_p = dropout_p
        
    def _enable_dropout(self):
        """ Enable dropout in the entire model """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32*7*7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.fc2(x)
        return x
    
    def predict_with_uncertainty(self, x, n_samples=20):
        """
        Make predictions with uncertainty estimation using MC Dropout
        Returns:
            mean_pred: mean prediction
            epistemic_uncertainty: uncertainty from model parameters
            aleatoric_uncertainty: uncertainty from data noise
        """
        self.eval()
        self._enable_dropout()
        
        predictions = []
        for _ in range(n_samples):
            pred = F.softmax(self(x), dim=1)
            predictions.append(pred.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = predictions.var(dim=0).mean(dim=1)
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric = (mean_pred * (1 - mean_pred)).mean(dim=1)
        
        return mean_pred, epistemic, aleatoric
    
    def get_calibration_scores(self, predictions, confidences, labels, n_bins=10):
        """
        Compute calibration metrics
        Returns:
            reliability: reliability diagram coordinates
            ece: Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = predictions.eq(labels)
        
        ece = 0.0
        reliability_diagram = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                reliability_diagram.append((avg_confidence_in_bin, accuracy_in_bin))
        
        return reliability_diagram, ece