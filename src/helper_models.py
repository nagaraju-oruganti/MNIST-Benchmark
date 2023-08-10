import torch
import torch.nn as nn
import torch.nn.functional as F

class ANNClassifierB(nn.Module):
    def __init__(self, config, num_classes):
        super(ANNClassifier, self).__init__()
        self.config = config
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0)
        
        self.fc2 = nn.Linear(128, 32)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0)
        
        self.fc3 = nn.Linear(32, num_classes)

    def loss_fn(self, outputs, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y)
        return loss
    
    def forward(self, x, y):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = x.view(-1, 16 * 7 * 7)
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        logits = self.fc3(x)
        loss = self.loss_fn(logits, y)
        return logits, loss
    
    
# SPIKING NEURAL NETWORK
class SNNClassifier(nn.Module):
    def __init__(self, config, num_classes):
        super(SNNClassifier, self).__init__()
        
        self.config = config
        self.time_steps = config.time_steps
        input_size = 28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * input_size * input_size, input_size * input_size)
        self.fc2 = nn.Linear(input_size * input_size, num_classes)
        
    def loss_fn(self, outputs, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y)
        return loss
        
    def forward(self, x, y):
        x = x.unsqueeze(1)
        membrane_potential = torch.zeros(x.size(0), self.fc2.out_features).to(self.config.device)
        spike_trains = []
        
        for t in range(self.time_steps):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            
            membrane_potential = (1 - 0.1) * membrane_potential + torch.relu(self.fc2(x))
            spikes = membrane_potential > 0.5  # Leaky Integrate-and-Fire threshold
            membrane_potential *= (1 - spikes.float())
            spike_trains.append(spikes)
            if t < self.time_steps - 1:
                x = x.view(x.size(0), 1, 28, 28)        # return to original shape
                
        #Convert spike trains to spike rates
        spike_rates = [torch.mean(spikes.float()) for spikes in spike_trains]
        
        #Combine spike rates using a simple operation (e.g., summation)
        combined_spike_rates = torch.stack(spike_rates).sum(dim=0)

        # Use the combined spike rates to modify the loss
        logits = self.fc2(x) + combined_spike_rates #+ spike_trains[-1]
        loss = self.loss_fn(logits, y)
        
        return logits, loss
    
class ANNClassifier(nn.Module):
    def __init__(self, config, num_classes):
        super(ANNClassifier, self).__init__()
        
        self.config = config
        self.time_steps = config.time_steps
        input_size = 28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * input_size * input_size, input_size * input_size)
        self.fc2 = nn.Linear(input_size * input_size, num_classes)
        
    def loss_fn(self, outputs, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y)
        return loss
        
    def forward(self, x, y):
        x = x.unsqueeze(1)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        logits = torch.relu(self.fc2(x))
        loss = self.loss_fn(logits, y)
        
        return logits, loss