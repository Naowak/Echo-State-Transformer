import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

class LSTM(nn.Module):
    def __init__(self, hidden_size=10, num_layers=1, device='cpu'):
        """
        Classe pour un modèle LSTM.

        Paramètres :
        - hidden_size (int) : Dimension de l'espace caché.
        - num_layers (int) : Nombre de couches LSTM.
        - device (str) : Dispositif ('cpu' ou 'cuda').
        """
        super(LSTM, self).__init__()

        # Paramètres du modèle
        self.input_size = None
        self.output_size = None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = None
        self.weight_decay = None
        self.device = device

        # Définir le modèle LSTM
        self.model = None
        self.fc = None

        # Définir la fonction de perte et l'optimiseur
        self.optimizer = None
        self.criterion = None

    def run_training(self, X_train, Y_train, T_train, X_valid=None, Y_valid=None, T_valid=None, epochs=100, batch_size=32, 
                     learning_rate=1e-3, weight_decay=1e-2, patience=5, min_delta=1e-5, classification=False, path=None):
        """
        Entraîne le modèle LSTM.

        Paramètres :
        - X_train : Données d'entrée pour l'entrainement (numpy array ou tenseur). (sample, time, input_dim)
        - Y_train : Données de sortie pour l'entrainement (numpy array ou tenseur). (sample, time, output_dim)
        - T_train : Indices des pas de temps pour les prédictions (numpy array ou tenseur). (sample, time)
        - X_valid : Données d'entrée pour la validation (numpy array ou tenseur). (sample, time, input_dim)
        - Y_valid : Données de sortie pour la validation (numpy array ou tenseur). (sample, time, output_dim)
        - T_valid : Indices des pas de temps pour la validation (numpy array ou tenseur). (sample, time)
        - epochs (int) : Nombre d'époques.
        - batch_size (int) : Taille des mini-lots.
        - learning_rate (float) : Taux d'apprentissage.
        - weight_decay (float) : Pénalité L2.
        - patience (int) : Nombre d'époques sans amélioration avant l'arrêt anticipé.
        - min_delta (float) : Changement minimum pour être considéré comme une amélioration.
        - classification (bool) : Indique si la tâche est une classification.
        """
        # Convertir les données en tenseurs PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32, device=self.device)

        if X_valid is not None and Y_valid is not None and T_valid is not None:
            X_valid = torch.tensor(X_valid, dtype=torch.float32, device=self.device)
            Y_valid = torch.tensor(Y_valid, dtype=torch.float32, device=self.device)
        
        # Define model
        self._define_model(X_train.shape[-1], Y_train.shape[-1], learning_rate, weight_decay, classification)
        
        # Créer un DataLoader
        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loss_history = []
        valid_loss_history = []

        # Early stopping parameters
        patience_counter = 0
        best_valid_loss = float('inf')

        # Entraîner le modèle
        for epoch in range(epochs):
            
            # Init epoch
            self.model.train()
            epoch_loss = []
            
            # Train epoch
            for i, (batch_X, batch_Y) in enumerate(dataloader):
                # Initialiser les états cachés
                h0 = torch.zeros(self.num_layers, batch_X.size(0), self.hidden_size, device=self.device)
                c0 = torch.zeros(self.num_layers, batch_X.size(0), self.hidden_size, device=self.device)

                # Forward pass
                outputs, _ = self.model(batch_X, (h0, c0)) # outputs: (batch_size, seq_length, hidden_size)
                outputs = self.fc(outputs) # outputs: (batch_size, seq_length, output_size) 

                # Compute loss only for the prediction timesteps
                T_batch = T_train[i*batch_size:(i+1)*batch_size]
                loss = self._compute_loss(batch_X, batch_Y, outputs, T_batch, classification)
                epoch_loss.append(loss.item())

                # Backward pass et mise à jour des poids
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Compute validation  
            if X_valid is not None and Y_valid is not None and T_valid is not None:
                # Make predictions
                outputs_valid = torch.tensor(self.run_inference(X_valid, batch_size), device=self.device, dtype=torch.float32)
                loss_valid = self._compute_loss(X_valid, Y_valid, outputs_valid, T_valid, classification)
                valid_loss_history.append(loss_valid.item())

                # if model is improving, save it, else, increase patience
                if loss_valid < best_valid_loss - min_delta:
                    best_valid_loss = loss_valid
                    patience_counter = 0
                    if path is not None:
                        self.save(path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Validation loss stopped decreasing at epoch {epoch+1-patience}. Early stopping.")
                        # print(f"Best validation loss: {best_valid_loss}")
                        # print(f"Current validation loss: {loss_valid.item()}")
                        break

            # Register training loss history
            train_loss_history.append(np.mean(epoch_loss))
        
        return train_loss_history, valid_loss_history

    def run_inference(self, X, batch_size=32):
        """
        Génère des prédictions avec le modèle LSTM.
        
        Paramètres :
        - X : Tenseur d'entrée (batch_size, seq_len, input_dim).
        - batch_size (int) : Taille des mini-lots.
        
        Retourne :
        - Tenseur de sortie (batch_size, seq_len, output_dim).
        """
        # Convertir le tenseur d'entrée en float32
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Prediction par batch
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                
                # Initialiser les états cachés
                h0 = torch.zeros(self.num_layers, X_batch.size(0), self.hidden_size, device=self.device)
                c0 = torch.zeros(self.num_layers, X_batch.size(0), self.hidden_size, device=self.device)

                # Forward pass
                batch_outputs, _ = self.model(X_batch, (h0, c0))
                batch_outputs = self.fc(batch_outputs)
                outputs.append(batch_outputs.cpu().numpy())
            
        return np.concatenate(outputs, axis=0)

    def count_params(self):
        """"
        Compte le nombre de paramètres du modèle.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path):
        """
        Save the model to a file.

        Parameters:
        - path (str): Path to save the model.
        """
        pickle.dump(self, open(path, 'wb'))
        # print(f"Model saved to {path}")

    @staticmethod
    def load(path):
        """
        Load the model from a file.

        Parameters:
        - path (str): Path to load the model from.
        """
        model = pickle.load(open(path, 'rb'))
        print(f"Model loaded from {path}")
        return model
    
    def _define_model(self, input_size, output_size, learning_rate, weight_decay, classification):
        """
        Définit le modèle LSTM.

        Paramètres :
        - input_size (int) : Dimension de l'entrée.
        - output_size (int) : Dimension de la sortie.
        - learning_rate (float) : Taux d'apprentissage.
        - weight_decay (float) : Pénalité L2.
        - classification (bool) : Indique si la tâche est une classification.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, device=self.device)
        self.fc = nn.Linear(self.hidden_size, output_size, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if classification:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()  # Vous pouvez changer cela en fonction de votre tâche
    
    def _compute_loss(self, X, Y, outputs, T, classification):
        """
        Calcule la perte.

        Paramètres :
        - X : Données d'entrée (batch_size, seq_len, input_dim).
        - Y : Données de sortie (batch_size, seq_len, output_dim).
        - outputs : Sorties du modèle (batch_size, seq_len, output_dim).
        - T : Indices des pas de temps pour les prédictions (batch_size, seq_len).
        - classification (bool) : Indique si la tâche est une classification.
        """
        # Select only the prediction timesteps
        preds = []
        truths = []
        for j in range(X.shape[0]):
            preds += [outputs[j, T[j], :]]
            truths += [Y[j, T[j], :]]
        preds = torch.stack(preds)
        truths = torch.stack(truths)

        if classification:
            # Prepare truth tensor for CrossEntropyLoss
            truths = torch.argmax(truths, dim=-1).view(-1) # [B * prediction_timesteps] int: class
            preds = preds.view(-1, Y.shape[-1]) # [B * prediction_timesteps, O] float: logits

        # Compute loss
        loss = self.criterion(preds, truths)
        return loss


