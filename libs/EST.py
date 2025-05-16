import torch
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
# torch.autograd.set_detect_anomaly(True)

class EST(torch.nn.Module):
    """Implementation of the Echo State Transformer model."""

    def __init__(self, num_layers=1, memory_units=4, memory_dim=100, attention_dim=3, dropout=0, memory_connectivity=0.2, device='cpu', dtype=torch.float32):
        """
        Initialize the Echo State Transformer model.

        Parameters:
        - layers (int): Number of layers in the model.
        - memory_units (int): Number of memory units (M).
        - memory_dim (int): Dimension of each memory unit (R).
        - attention_dim (int): Dimension of the attention mechanism (D).
        - dropout (float): Dropout rate.
        - memory_connectivity (float): Connectivity of the memory units.
        - device (str): Device to use ('cpu' or 'cuda').
        - dtype (torch.dtype): Data type of the tensors.
        """
        super(EST, self).__init__()
        # Store model parameters
        self.layers = num_layers
        self.memory_units = memory_units
        self.memory_dim = memory_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.memory_connectivity = memory_connectivity
        self.device = device
        self.dtype = dtype
        
        # Layers 
        self.fc_in_dropout = torch.nn.Dropout(dropout)
        self.fc_in = None # [I, D]
        self.fc_out = None # [D, O]
        self.est_layers = torch.nn.ModuleList([ESTLayer(memory_units, memory_dim, attention_dim, dropout, memory_connectivity, device, dtype) for _ in range(self.layers)])

        # Optimizer & Loss
        self.optimizer = None
        self.criterion = None

    def forward(self, X, states=None):
        """
        Forward pass of the Echo State Transformer model.

        Parameters:
        - X (torch.Tensor): Input tensor [B, I].
        - states (torch.Tensor, optional): Reservoirs states [B, L, M, R]. Defaults to None. (L for layers)

        Returns:
        - Y (torch.Tensor): Output tensor [B, O].
        - states (torch.Tensor): Updated states [B, L, M, R].
        """
        # Move X to the device & batch_size
        X = X.to(self.dtype).to(self.device)
        batch_size = X.shape[0]

        # Init states if not provided
        if states is None:
            states = torch.zeros(batch_size, self.layers, self.memory_units, self.memory_dim, dtype=self.dtype, device=self.device) # [B, L, M, R]
        
        # Input Embedding
        emb = self.fc_in_dropout(X @ self.fc_in) # [B, D]

        # Forward pass through the layers
        new_states = []
        for i, layer in enumerate(self.est_layers):
            emb, ns = layer(emb, states[:, i])
            new_states.append(ns)
        new_states = torch.stack(new_states, dim=1)

        # Output Embeddings
        Y = (emb @ self.fc_out).view(batch_size, self.output_dim) # [B, O]

        return Y, new_states
    
    def run_training(self, X_train, Y_train, T_train, X_valid=None, Y_valid=None, T_valid=None, epochs=100, batch_size=32, 
                     learning_rate=1e-3, weight_decay=1e-2, patience=5, min_delta=1e-5, classification=False, bptt=True, path=None):
        """
        Training function for the Echo State Transformer model.

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
        - bptt (bool) : Indique si le Backpropagation Through Time est utilisé.
        - path (str) : Chemin pour enregistrer le modèle. If None, the model is not saved.
        """
        # Convertir les données en tenseurs PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32, device=self.device)

        if X_valid is not None and Y_valid is not None and T_valid is not None:
            X_valid = torch.tensor(X_valid, dtype=torch.float32, device=self.device)
            Y_valid = torch.tensor(Y_valid, dtype=torch.float32, device=self.device)

        # Define the model
        self._define_model(X_train.shape[-1], Y_train.shape[-1], learning_rate, weight_decay, classification)

        # Créer un DataLoader
        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loss_history = []
        valid_loss_history = []

        # Early stopping parameters
        patience_counter = 0
        best_valid_loss = float('inf')

        # Train the model
        tqdm_bar = tqdm(range(epochs), desc='Training')
        for epoch in tqdm_bar:
            
            # Init epoch
            self.train()
            epoch_loss = []

            # Train epoch
            for i, (batch_X, batch_Y) in enumerate(dataloader):

                # print(f"Batch {i+1}/{len(dataloader)}", end='\r', flush=True)

                # Initialiser les états cachés pour la séquence
                states = None
                outputs = []

                # Forward pass pour chaque temps
                for t in range(batch_X.shape[1]):
                    y_out, states = self.forward(batch_X[:, t], states) # [B, O], [B, M, R]
                    if not bptt:
                        states = states.detach() # Détacher les états pour éviter le calcul du gradient
                    outputs.append(y_out)
                outputs = torch.stack(outputs, dim=1) # [B, T, O]

                # Compute loss only for the prediction timesteps
                T_batch = T_train[i*batch_size:(i+1)*batch_size]
                loss = self._compute_loss(batch_X, batch_Y, outputs, T_batch, classification)
                epoch_loss.append(loss.item())

                # Backward pass et mise à jour des paramètres
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Clamp the spectral radius
                with torch.no_grad():
                    for layer in self.est_layers:
                        layer.memory.clamp_hp()
            
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
            mean_loss = np.mean(epoch_loss)
            train_loss_history.append(mean_loss)
            tqdm_bar.set_description(f"Epoch: {epoch+1}/{epochs} - Loss: {mean_loss:.6f} - Val Loss: {valid_loss_history[-1] if valid_loss_history else 'N/A'}")

        return train_loss_history, valid_loss_history
    
    def run_inference(self, X, batch_size=10, states=None):
        """
        Run the Echo State Transformer model on the input tensor.
        
        Parameters:
            - X (torch.Tensor): Input tensor [sample, sequence, dim]
            - batch_size (int): Number of samples per batch
        
        Returns:
            - Y_hat (torch.Tensor): Predicted tensor [sample, sequence, dim]
        """
        # Convert input tensor to dtype
        self.eval()
        X = torch.tensor(X, dtype=self.dtype, device=self.device)

        # Calcul du nombre de batches
        num_batches = X.shape[0] // batch_size
        Y_hat = []

        # tqdm_bar = tqdm(range(num_batches), desc='Prediction')

        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Extraction d'un batch de données
                X_batch = X[batch_idx * batch_size : (batch_idx + 1) * batch_size]

                # Forward pass pour chaque temps
                outputs = []
                for t in range(X_batch.shape[1]):
                    y_out, states = self.forward(X_batch[:, t], states) # [B, O], [B, M, R]
                    outputs.append(y_out)
                
                # Convertir la séquence de sorties en tenseur [batch_size, sequence_length, output_dim]
                Y_hat_batch = torch.stack(outputs, dim=1)
                Y_hat.append(Y_hat_batch)
            
            # Concaténer les batches de prédictions
            Y_hat = torch.cat(Y_hat, dim=0)

        return Y_hat.cpu().numpy()

    def count_params(self):
        """
        Count the number of parameters in the network.

        Returns:
        - num_params (int): Number of parameters in the network
        """
        return sum(p.numel() for p in self.parameters())
    
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


    def _define_model(self, input_dim, output_dim, learning_rate, weight_decay, classification=False):
        """
        Define the model architecture.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - learning_rate (float): Learning rate for the optimizer.
        - weight_decay (float): Weight decay for the optimizer.
        - classification (bool): Indicates if the task is a classification task.
        """
        # Define the input and output dims
        self.input_dim = input_dim # I
        self.output_dim = output_dim # O
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.classification = classification

        # FC layers (embeddings)
        self.fc_in = torch.nn.Parameter(torch.empty(self.input_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [I, D]
        self.fc_out = torch.nn.Parameter(torch.empty(self.attention_dim, self.output_dim, dtype=self.dtype, device=self.device)) # [D, O]
        torch.nn.init.normal_(self.fc_in)
        torch.nn.init.normal_(self.fc_out)

        # EST layers
        for layer in self.est_layers:
            layer._define_model()

        # Separate hyperparameters from the rest of the parameters
        hp_names = ['memory.sr', 'memory.temperature']
        hyperparameters = []
        parameters = []
        for n, p in self.named_parameters():
            is_hp = False
            for hp in hp_names:
                if hp in n:
                    hyperparameters.append(p)
                    is_hp = True
                    break
            if not is_hp:
                parameters.append(p)

        # Define the optimizer with different learning rates depending on the parameters/hyperparameters
        config = [
            {'params': parameters, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': hyperparameters, 'lr': learning_rate, 'weight_decay': 0}
        ]
        self.optimizer = torch.optim.Adam(config)
        
        # Select the loss function 
        if classification:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        print(f"Model defined with {self.count_params()} parameters")

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


class ESTLayer(torch.nn.Module):
    """Implementation of the Echo State Transformer layer."""

    def __init__(self, memory_units=4, memory_dim=100, attention_dim=3, dropout=0, memory_connectivity=0.2, device='cpu', dtype=torch.float32):
        """
        Initialize the Echo State Transformer layer.

        Parameters:
        - memory_units (int): Number of memory units (M).
        - memory_dim (int): Dimension of each memory unit (R).
        - attention_dim (int): Dimension of the attention mechanism (D).
        - dropout (float): Dropout rate.
        - memory_connectivity (float): Connectivity of the memory units.
        - device (str): Device to use ('cpu' or 'cuda').
        - dtype (torch.dtype): Data type of the tensors.
        """
        super(ESTLayer, self).__init__()
        # Store model parameters
        self.memory_units = memory_units # M
        self.memory_dim = memory_dim # R
        self.attention_dim = attention_dim # D
        self.dropout = dropout
        self.memory_connectivity = memory_connectivity
        self.device = device
        self.dtype = dtype

        # Norm, activation, dropout
        self.norm1 = torch.nn.RMSNorm(normalized_shape=(attention_dim), eps=1e-8, device=device, dtype=self.dtype)
        self.norm2 = torch.nn.RMSNorm(normalized_shape=(attention_dim), eps=1e-8, device=device, dtype=self.dtype)
        self.norm3 = torch.nn.RMSNorm(normalized_shape=(attention_dim), eps=1e-8, device=device, dtype=self.dtype)
        self.gelu = torch.nn.GELU()
        self.attn_dropout = torch.nn.Dropout(self.dropout)
        self.self_attn_dropout = torch.nn.Dropout(self.dropout)
        self.ff_out_dropout = torch.nn.Dropout(dropout)

        # Initialize memory with fixed weights
        self.memory = None
        
        # Query, Key, Value weights 
        self.Wq = None # [M, D, D]
        self.Wk = None # [M, D, D]
        self.Wv = None # [M, D, D]
        self.SWq = None # [D, D]
        self.SWk = None # [D, D]
        self.SWv = None # [D, D]

        # Other Layers
        self.Wreduce = None # [M*D, D]
        self.ff_in = None # [D, 4*D]
        self.ff_out = None # [4*D, D]

    def forward(self, emb, Si):
        """
        Forward pass of an Echo State Transformer layer.

        Parameters:
        - emb (torch.Tensor): Input tensor [B, I]
        - Si (torch.Tensor, optional): Reservoirs states [B, M, R].

        Returns:
        - Y (torch.Tensor): Output tensor [B, O]
        """
        # Init batch_size & previous state
        batch_size = emb.shape[0] # B
        Si_ = Si # [B, M, R]
        Sout_ = (Si_.unsqueeze(2) @ self.memory.Wout).squeeze(2) # [B, M, D] # /!\ Can be computed once

        # Attention on previous states
        Q = emb.view(batch_size, 1, 1, self.attention_dim) @ self.Wq # [B, M, 1, D]
        K = Sout_.unsqueeze(1) @ self.Wk.unsqueeze(0) # [B, M, M, D]
        V = Sout_.unsqueeze(1) @ self.Wv.unsqueeze(0) # [B, M, M, D]
        attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout if self.training else 0) # [B, M, 1, D]
        update = self.norm1(self.attn_dropout(attn.squeeze(2) + emb.unsqueeze(1))) # [B, M, D]

        # Memory update
        Si, Sout = self.memory(update, Si_) # [B, M, R], [B, M, D]

        # Attention on current state
        SQ = Sout @ self.SWq # [B, M, D]
        SK = Sout @ self.SWk # [B, M, D]
        SV = Sout @ self.SWv # [B, M, D]
        self_attn = torch.nn.functional.scaled_dot_product_attention(SQ, SK, SV, dropout_p=self.dropout if self.training else 0) # [B, M, D]
        self_update = self.norm2(self.self_attn_dropout(self_attn + Sout)) # [B, M, D]

        # Knowledge Enhancement
        SUreduce = self.Wreduce(self_update.view(batch_size, self.memory_units * self.attention_dim)) # [B, D]
        Z = self.gelu(self.ff_in(SUreduce))
        OUT = self.norm3(self.ff_out_dropout(self.ff_out(Z) + SUreduce))

        return OUT, Si # [B, O], [B, M, R]

    def _define_model(self):
        """
        Define the model architecture.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - learning_rate (float): Learning rate for the optimizer.
        - weight_decay (float): Weight decay for the optimizer.
        - classification (bool): Indicates if the task is a classification task.
        """
        # Initialize memory with fixed weights
        self.memory = Memory(units=self.memory_units, neurons=self.memory_dim, input_dim=self.attention_dim, output_dim=self.attention_dim, 
                            res_connectivity=self.memory_connectivity, input_connectivity=self.memory_connectivity,
                             device=self.device, dtype=self.dtype)
        
        # Query, Key, Value weights
        self.Wq = torch.nn.Parameter(torch.empty(self.memory_units, self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [M, D, D]
        self.Wk = torch.nn.Parameter(torch.empty(self.memory_units, self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [M, D, D]
        self.Wv = torch.nn.Parameter(torch.empty(self.memory_units, self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [M, D, D]
        self.SWq = torch.nn.Parameter(torch.empty(self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [D, D]
        self.SWk = torch.nn.Parameter(torch.empty(self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [D, D]
        self.SWv = torch.nn.Parameter(torch.empty(self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [D, D]
        torch.nn.init.kaiming_uniform_(self.Wq, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.Wk, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.Wv, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.SWq, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.SWk, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.SWv, a=5**0.5)

        # Linear layers (kaiming uniform init)
        self.Wreduce = torch.nn.Linear(self.memory_units * self.attention_dim, self.attention_dim, bias=True, dtype=self.dtype, device=self.device) # [M*D, D]
        self.ff_in = torch.nn.Linear(self.attention_dim, 4 * self.attention_dim, bias=True, dtype=self.dtype, device=self.device) # [D, 4*D]
        self.ff_out = torch.nn.Linear(4 * self.attention_dim, self.attention_dim, bias=True, dtype=self.dtype, device=self.device) # [4*D, D]


class Memory(torch.nn.Module):
    """Implements a reservoir network."""

    def __init__(self, units=None, neurons=None, input_dim=None, output_dim=None, input_scaling=1.0, res_connectivity=0.2, 
                 input_connectivity=0.2, bias_prob=0.5, device='cpu', dtype=torch.float32):
        """
        Create a reservoir with the given parameters.

        Parameters:
        - units (int): Number of reservoirs.
        - neurons (int): Number of neurons in each reservoir.
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - input_scaling (float): Input scaling.
        - res_connectivity (float): Connectivity of the recurrent weight matrix.
        - input_connectivity (float): Connectivity of the input weight matrix.
        - bias_prob (float): Probability of bias.
        - device (str): Device to use ('cpu' or 'cuda').
        - dtype (torch.dtype): Data type of the tensors.
        """
        super(Memory, self).__init__()
        # Check the parameters
        if units is None or neurons is None or input_dim is None or output_dim is None:
            raise ValueError("You must provide the number of units, neurons and input/output dimension")
        
        # Store the parameters
        self.units = units # M
        self.neurons = neurons # R
        self.input_dim = input_dim # D
        self.output_dim = output_dim # D
        self.input_scaling = input_scaling
        self.res_connectivity = res_connectivity
        self.input_connectivity = input_connectivity
        self.bias_prob = bias_prob
        self.device = device
        self.dtype = dtype

        # Create matrices
        W = _initialize_matrix((units, neurons, neurons), res_connectivity, distribution='normal', dtype=dtype, device=device)
        Win = _initialize_matrix((units, input_dim, neurons), input_connectivity, distribution='fixed_bernoulli', dtype=dtype, device=device)
        bias = _initialize_matrix((units, 1, neurons), bias_prob, distribution='bernoulli', dtype=dtype, device=device)
        Wout = _initialize_matrix((units, neurons, output_dim), 1.0, distribution='normal', dtype=dtype, device=device)
        adaptive_lr = torch.nn.init.uniform_(torch.empty((units, input_dim, 1), device=device, dtype=dtype))
        initial_sr = _get_spectral_radius(W).view(units, 1, 1)
        sr = torch.rand(units, 1, 1, dtype=dtype, device=device)
        temperature = torch.ones(1, dtype=dtype, device=device)

        # Register W, Win & bias as buffer
        self.W = torch.nn.Buffer(W / initial_sr) # [M, R, R] Set SR to 1
        self.Win = torch.nn.Buffer(Win) # [M, D, R]
        self.bias = torch.nn.Buffer(bias) # [M, 1, R]

        # Register parameters 
        self.Wout = torch.nn.Parameter(Wout) # [M, R, D]
        self.sr = torch.nn.Parameter(sr) # [M, 1, 1]
        self.adaptive_lr = torch.nn.Parameter(adaptive_lr) # [M, D, 1] 
        self.temperature = torch.nn.Parameter(temperature) # [1]

        # Register the non-zero positions of W and Win, and the corresponding positions in x
        self.w_pos = W.transpose(-2, -1).nonzero(as_tuple=True) 
        self.win_pos = Win.transpose(-2, -1).nonzero(as_tuple=True)
        self.xw_pos = (self.w_pos[0], torch.zeros(self.w_pos[1].shape, dtype=int, device=device), self.w_pos[2])
        self.xwin_pos = (self.win_pos[0], torch.zeros(self.win_pos[1].shape, dtype=int, device=device), self.win_pos[2])
        

    def forward(self, X, state):
        """
        Forward pass of the reservoir network.
        
        Parameters:
        - X (torch.Tensor): Input tensor [batch, units, input_dim].
        - state (torch.Tensor, optional): Initial states [batch, units, neurons].

        Returns:
        - new_state (torch.Tensor): Updated state [batch, units, neurons].
        """  
        # Reshape X & state
        batch_size = X.shape[0] # B
        X = X.view(batch_size, self.units, 1, self.input_dim) # [B, M, 1, D]
        state = state.view(batch_size, self.units, 1, self.neurons) # [B, M, 1, R]

        # Adaptive Leak Rate
        lr = torch.softmax((X @ self.adaptive_lr) / self.temperature, dim=1) # [batch, units, 1, 1]

        # Feed
        feed = _sparse_mm_subhead(X, self.Win,  self.xwin_pos, self.win_pos, None) # [B, subM=k, 1, R]
        # feed = _sparse_mm_subhead(X, self.Win,  self.xwin_pos, self.win_pos, w_heads) # [B, subM=k, 1, R]

        # Adjust the spectral radius
        W = self.W * self.sr # [M, R, R]

        # Echo
        echo = _sparse_mm_subhead(state, W, self.xw_pos, self.w_pos, None) + self.bias # [B, subM=k, 1, R]
        # echo = _sparse_mm_subhead(state, W, self.xw_pos, self.w_pos, w_heads) + self.bias[w_heads] # [B, subM=k, 1, R]

        # Update the selected heads
        new_state = ((1 - lr) * state) + lr * torch.tanh(feed + echo) # [B, subM=k, 1, R]
        # heads_updated = ((1 - lr[batchs, w_heads]) * state[batchs, w_heads]) + lr[batchs, w_heads] * torch.tanh(feed + echo) # [B, subM=k, 1, R]
        # new_state = state.clone() # [B, M, 1, R]
        # new_state[batchs, w_heads] = heads_updated # [B, M, 1, R]

        output = new_state @ self.Wout # [B, M, 1, D]

        return new_state.squeeze(2), output.squeeze(2) # [B, M, R], [B, M, D] 
    
    def clamp_hp(self):
        """
        Clamp the hyperparameters of the reservoir.
        """
        # self.lr.data.clamp_(1e-5, 1)
        self.sr.data.clamp_(1e-5, 10)
    


def _initialize_matrix(shape, connectivity, distribution='normal', dtype=torch.float32, device='cpu'):
    """
    Initialize a matrix with a given shape and connectivity.

    Parameters:
    - shape (tuple): Shape of the matrix.
    - connectivity (float): Connectivity of the matrix.
    - distribution (str): Distribution of the matrix values ('normal' or 'bernoulli').
    - kwargs: Additional arguments for the distribution.

    Returns:
    - torch.Tensor: Initialized matrix.
    """
    if distribution == 'normal':
        matrix = torch.tensor(np.random.normal(size=shape, loc=0, scale=1), device=device, dtype=dtype)
        mask = _fixed_bernoulli(shape, connectivity, device=device)
        return matrix * mask
    
    elif distribution == 'bernoulli':
        return torch.bernoulli(torch.full(shape, connectivity, device=device, dtype=dtype))
    
    elif distribution == 'fixed_bernoulli':
        return _fixed_bernoulli(shape, connectivity, device=device)
    
    else:
        raise ValueError("Unsupported distribution type")

def _get_spectral_radius(matrix):
    """
    Get the spectral radius of a matrix.

    Parameters:
    - matrix (torch.Tensor): The matrix to analyze.

    Returns:
    - float: The spectral radius of the matrix.
    """
    # Convert the matrix to float32
    matrix = matrix.to(torch.float32) # eigenvalues does not support dtype

    # Compute the eigenvalues
    device = str(matrix.device)
    if 'mps' in device:
        # MPS does not support eigvals
        eigenvalues = torch.linalg.eigvals(matrix.to('cpu')).to(matrix.device) # So we temporarily move the matrix to the CPU
    else:
        eigenvalues = torch.linalg.eigvals(matrix) 
    
    # Compute the maximum eigenvalue
    abs_eigenvalue = torch.sqrt(eigenvalues.real**2 + eigenvalues.imag**2) # Cuda does not support torch.abs on Complex Number
    spectral_radius = torch.max(abs_eigenvalue, dim=-1).values

    return spectral_radius

def _fixed_bernoulli(shape, connectivity, device='cpu', dtype=torch.float32):
    """
    Generate a connectivity matrix with a given shape and connectivity.

    Parameters:
    - shape (tuple): Shape of the matrix (head, line, column) or (line, column).
    - connectivity (float): Connectivity of the matrix.

    Every column has the same number of ones. (This constraint allows sparse matrix multiplication)

    Returns:
    - torch.Tensor: Connectivity matrix (head, line, column).
    """

    # Check the connectivity
    if not 0 < connectivity <= 1:
        raise ValueError("Connectivity must be > 0 et <= 1")
    
    # If len(shape) == 2, add a dimension
    if len(shape) == 2:
        shape = (1, shape[0], shape[1])
    
    # Init matrix & nb connections
    nb_connections = max(1, int(connectivity * shape[-2])) # At least one connection
    matrix = torch.zeros(shape, device=device, dtype=dtype)
    
    # For each column, set the connections
    for head in range(shape[-3]):
        for col in range(shape[-1]):
            indices = torch.randperm(shape[-2])[:nb_connections]
            matrix[head, indices, col] = 1
    
    return matrix

def _sparse_mm_subhead(x, W, x_pos, w_pos, w_heads):
    """
    Perform a sparse matrix multiplication between x and W on a subset of W heads for each batch of x.

    Parameters:
    - x (torch.Tensor): Input tensor [B, H, 1, D].
    - W (torch.Tensor): Weight tensor [B, H, D, O].
    - x_pos (torch.Tensor): Indices of the corresponding values in x.
    - w_pos (torch.Tensor): Indices of the non-zero values in W.
    - w_heads (int): Heads to use per batch. [B, subH]

    With B as the batch size, H as the number of heads, O as the number of output dim, and D as the dim of the input sequence.

    Returns:
    - torch.Tensor: Result tensor [B, subH, 1, O].
    """
    # Extract subW (only non-zero values) & select heads
    subW = W.transpose(-2, -1)[w_pos].reshape(W.shape[0], W.shape[-1], -1).transpose(-2, -1) # [H, connectivity, O]
    # subW = subW[w_heads] # [B, subH, connectivity, O]

    # Extract subX (corresponding values of x) & select heads per batch
    subX = x[:, *x_pos].reshape(x.shape[0], x.shape[1], W.shape[-1], -1).transpose(-2, -1) # [B, H, connectivity, O]
    # batchs = torch.arange(x.shape[0]).view(-1, 1).expand_as(w_heads) # [B, subH]
    # subX = subX[batchs, w_heads] # [B, subH, connectivity, O]

    # Compute the result 
    result = (subW * subX).sum(-2).view(subX.shape[0], subX.shape[1], 1, subW.shape[-1]) # [B, subH, 1, O]

    return result











