import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MIRT_2PL(nn.Module):
    """
    2PL Model: 2-Parameter Logistic Model
    p_ij = c_j + (1 - c_j) * sigmoid(a_j * (theta_i - b_j))
    
    Args:
        n_students (int): Number of students
        n_items (int): Number of items
        n_dims (int): Number of dimensions (latent traits)
    
    Attributes:
        theta (nn.Parameter): Latent trait parameter for students, shape (n_students, n_dims)
        a (nn.Parameter): Discrimination parameter for items, shape (n_items, n_dims)
        b (nn.Parameter): Difficulty parameter for items, shape (n_items,)
    """
    def __init__(self, n_students, n_items, n_dims):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_students, n_dims))
        self.a = nn.Parameter(torch.ones(n_items, n_dims))
        self.b = nn.Parameter(torch.zeros(n_items))
    
    def forward(self):
        logits = torch.matmul(self.theta, self.a.t()) - self.b.unsqueeze(0)
        prob = torch.sigmoid(logits)
        return prob

class MIRT_GPCM(nn.Module):
    """
    Generalized Partial Credit Model (GPCM)
    s_{ij,c} = c * (theta_i - a_j) - b_j
    p_{ij,c} = exp(s_{ij,c}) / sum_{c'} exp(s_{ij,c'})
    
    Args:
        n_students (int): Number of students
        n_items (int): Number of items
        n_dims (int): Number of dimensions (latent traits)
        n_categories (int): Number of categories
    
    Attributes:
        theta (nn.Parameter): Latent trait parameter for students, shape (n_students, n_dims)
        a (nn.Parameter): Discrimination parameter for items, shape (n_items, n_dims)
        b (nn.Parameter): Difficulty parameter for items, shape (n_items, n_categories - 1)
    """
    
    def __init__(self, n_students, n_items, n_dims, n_categories=5):
        super().__init__()
        self.n_students = n_students
        self.n_items = n_items
        self.n_dims = n_dims
        self.n_categories = n_categories
        
        self.theta = nn.Parameter(torch.zeros(n_students, n_dims))
        self.a = nn.Parameter(torch.ones(n_items, n_dims))
        self.b = nn.Parameter(torch.zeros(n_items, n_categories - 1))

    def forward(self):

        score = torch.matmul(self.theta, self.a.t())  # (n_students, n_dims) x (n_dims, n_items) -> (n_students, n_items)

        
        # with torch.no_grad():
        #     # (n_items, n_categories)
        #     step_full = torch.zeros(self.n_items, self.n_categories, dtype=self.b.dtype, device=self.b.device)
        #     step_full[:, 1:] = torch.cumsum(self.b, dim=1)
        
        zero_column = torch.zeros(self.n_items, 1, dtype=self.b.dtype, device=self.b.device)
        step_full = torch.cat([zero_column, torch.cumsum(self.b, dim=1)], dim=1)
        

        c_tensor = torch.arange(self.n_categories, device=score.device, dtype=score.dtype)  # (n_categories,)
        

        score_expanded = score.unsqueeze(-1).expand(-1, -1, self.n_categories) # (n_students, n_items, n_categories)
        c_expanded = c_tensor.view(1, 1, -1).expand(self.n_students, self.n_items, self.n_categories)
        
        
        step_full_expanded = step_full.unsqueeze(0).expand(self.n_students, -1, -1) # (1, n_items, n_categories)
        
        # s_{ij,c} = c * score_ij - step_full[j,c]
        s = c_expanded * score_expanded - step_full_expanded

        prob = torch.softmax(s, dim=2)  # (n_students, n_items, n_categories)
        return prob

class MIRT:
    """
    Simple MIRT class for fitting IRT models of arbitrary dimensions.
    
    Args:
        n_students (int): Number of students
        n_items (int): Number of items
        n_dims (int): Number of dimensions (latent traits)
        n_categories (int): Number of categories (polytomous only)
    """

    def __init__(self, n_students, n_items, n_dims=2, n_categories=5):
        self.n_students = n_students
        self.n_items = n_items
        self.n_dims = n_dims
        self.n_categories = n_categories
        
        self.model = None
        self.response_tensor = None
        self.history = None

    def specify_model(self, model_type="2PL"):
        if model_type == "2PL":
            self.model = MIRT_2PL(self.n_students, self.n_items, self.n_dims)
        elif model_type == "GPCM":
            self.model = MIRT_GPCM(self.n_students, self.n_items, self.n_dims, n_categories=self.n_categories)
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not implemented.")

    def fit(self, response_matrix, n_epochs=1000, lr=0.01):
        """
        Fit the IRT model with matrix response.
        
        Parameters:
            response_matrix (np.ndarray): Response matrix of shape (n_students, n_items)
            n_epochs (int): Number of epochs to train
            lr (float): Learning rate for optimizer
        
        Returns:
            nn.Module: Trained IRT model of type MIRT_2PL or MIRT_GPCM
        """
        
        if self.model is None:
            raise ValueError("No model specified. Call specify_model() first.")
        
        self.response_tensor = torch.tensor(response_matrix, dtype=torch.long)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        loss_history = []
        for _ in tqdm(range(n_epochs), desc="Training MIRT Model", leave=False):
            optimizer.zero_grad()
            prob = self.model()

            if prob.dim() == 2:
                # 2PL: -sum_{i,j}[y_ij * log(prob_ij) + (1-y_ij)*log(1-prob_ij)]
                prob_clamped = torch.clamp(prob, 1e-6, 1 - 1e-6)
                y = self.response_tensor.float()
                log_likelihood = y * torch.log(prob_clamped) + (1 - y) * torch.log(1 - prob_clamped)
                loss = -log_likelihood.sum()
            else:
                # GPCM: prob[i,j,c] -> X_{ij}=c.
                prob_clamped = torch.clamp(prob, 1e-6, 1.0)
                log_prob = torch.log(prob_clamped) # (n_students, n_items, n_categories)
                

                # Gather by response_tensor:
                gathered_log_prob = log_prob.gather(
                    dim=2,
                    index=self.response_tensor.unsqueeze(-1)
                ).squeeze(-1)  # (n_students, n_items)
                
                loss = -gathered_log_prob.sum()
            
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        
        self.history = loss_history
        return self.model