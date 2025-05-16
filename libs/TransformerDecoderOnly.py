import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

class TransformerDecoderOnly(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, device='cpu'):
        """
        Classe pour un modèle Transformer utilisant uniquement le décodeur.

        Paramètres :
        - d_model (int) : Dimension du modèle.
        - nhead (int) : Nombre de têtes dans l'attention multi-têtes.
        - num_layers (int) : Nombre de couches du décodeur.
        - dim_feedforward (int) : Dimension de la couche feedforward.
        - dropout (float) : Taux de dropout.
        - device (str) : Dispositif ('cpu' ou 'cuda').
        """
        super(TransformerDecoderOnly, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = None
        self.weight_decay = None
        self.device = device
        self.input_size = None
        self.output_size = None

        # Modèle Decoder et FC
        self.fc_in = None
        self.transformer = None
        self.fc_out = None

        # Définir la fonction de perte et l'optimiseur
        self.optimizer = None
        self.criterion = None

    def run_training(self, X_train, Y_train, T_train, X_valid=None, Y_valid=None, T_valid=None, epochs=10, batch_size=32, 
                     learning_rate=1e-3, weight_decay=1e-2, patience=5, min_delta=1e-5, classification=False, path=None):
        """
        Entraîne le modèle TransformerDecoderOnly.

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

        # Définir le modèle
        self._define_model(X_train.shape[-1], Y_train.shape[-1], learning_rate, weight_decay, classification)

        # Créer un DataLoader
        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loss_history = []
        valid_loss_history = []

        # Early stopping parameters
        patience_counter = 0
        best_valid_loss = float('inf')

        # Générer le masque de séquence
        mask_seq = self._generate_sequence_mask(Y_train.shape[1])

        # Entraîner le modèle
        for epoch in range(epochs):

            # Init epoch
            self.transformer.train()
            epoch_loss = []

            # Train epoch
            for i, (batch_X, batch_Y) in enumerate(dataloader):
                # Forward pass
                emb_X = self.fc_in(batch_X)
                tr_output = self.transformer(src=emb_X, mask=mask_seq)  # Transformer
                outputs = self.fc_out(tr_output)  # Projection finale
                
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
        Génère des prédictions avec le modèle TransformerDecoderOnly.

        Paramètres :
        - X : Tenseur d'entrée (batch_size, seq_len, input_dim).

        Retourne :
        - torch.Tensor : Séquence prédite (batch_size, seq_len, output_dim).
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        mask_seq = self._generate_sequence_mask(X.shape[1])

        self.transformer.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                emb_X = self.fc_in(X_batch)
                tr_output = self.transformer(src=emb_X, mask=mask_seq)
                output = self.fc_out(tr_output)
                outputs.append(output.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def count_params(self):
        """
        Compte le nombre de paramètres du modèle.

        Retourne :
        - int : Le nombre total de paramètres entraînables.
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

    def _define_model(self, input_size, output_size, learning_rate, weight_decay, classification=False):
        """
        Définir le modèle TransformerDecoderOnly.

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

        # Définir les couches
        self.fc_in = nn.Linear(input_size, self.d_model, device=self.device)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                device=self.device,
                batch_first=True,
            ),
            num_layers=self.num_layers,
        )
        self.fc_out = nn.Linear(self.d_model, output_size, device=self.device)

        # Définir la fonction de perte et l'optimiseur
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if classification:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def _generate_sequence_mask(self, length):
        """
        Génère un masque de séquence pour le décodeur Transformer.

        Paramètres :
        - length (int) : Longueur de la séquence.

        Retourne :
        - torch.Tensor : Masque de séquence (causal).
        """
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)
    
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
