import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class OnePLModel(nn.Module):
    """
    2PL Model: 2-Parameter Logistic Model
    p_ij =  sigmoid(theta_i - b_j)
    """
    def __init__(self, n_items, n_dims):
        super().__init__()
        self.n_items = n_items
        self.n_dims = n_dims
        # self.a = nn.Parameter(torch.ones(n_items, n_dims))
        self.b = nn.Parameter(torch.zeros(n_items))
    
    def forward(self, theta):
        logits = theta - self.b.unsqueeze(0)
        return torch.sigmoid(logits)
    
    def compute_loss(self, theta, responses):
        prob = self(theta)
        prob_clamped = torch.clamp(prob, 1e-6, 1-1e-6)
        y = responses.float()
        log_likelihood = y * torch.log(prob_clamped) + (1 - y) * torch.log(1 - prob_clamped)
        return -log_likelihood.sum()

class TwoPLModel(nn.Module):
    """
    2PL Model: 2-Parameter Logistic Model
    p_ij =  sigmoid(a_j * (theta_i - b_j))
    """
    def __init__(self, n_items, n_dims):
        super().__init__()
        self.n_items = n_items
        self.n_dims = n_dims
        self.a = nn.Parameter(torch.ones(n_items, n_dims))
        self.b = nn.Parameter(torch.zeros(n_items))
    
    def forward(self, theta):
        logits = torch.matmul(theta, self.a.t()) - self.b.unsqueeze(0)
        return torch.sigmoid(logits)
    
    def compute_loss(self, theta, responses):
        prob = self(theta)
        prob_clamped = torch.clamp(prob, 1e-6, 1-1e-6)
        y = responses.float()
        log_likelihood = y * torch.log(prob_clamped) + (1 - y) * torch.log(1 - prob_clamped)
        return -log_likelihood.sum()
    
    
class GPCMModel(nn.Module):
    """
    Generalized Partial Credit Model (GPCM)
    s_{ij,c} = c * (theta_i - a_j) - b_j
    p_{ij,c} = exp(s_{ij,c}) / sum_{c'} exp(s_{ij,c'})
    """
    def __init__(self, n_items, n_dims, n_categories):
        super().__init__()
        self.n_items = n_items
        self.n_dims = n_dims
        self.n_categories = n_categories
        self.a = nn.Parameter(torch.ones(n_items, n_dims))
        self.b = nn.Parameter(torch.zeros(n_items, n_categories - 1))
    
    def forward(self, theta):

        score = torch.matmul(theta, self.a.t())
        # Thresholds start from 0 and accumulate.
        zero_column = torch.zeros(self.n_items, 1, dtype=self.b.dtype, device=self.b.device)
        threshold = torch.cat([zero_column, torch.cumsum(self.b, dim=1)], dim=1)
        n_categories = self.n_categories
        c_tensor = torch.arange(n_categories, device=score.device, dtype=score.dtype)
        score_expanded = score.unsqueeze(-1).expand(-1, -1, n_categories)
        c_expanded = c_tensor.view(1, 1, -1).expand(score.shape[0], score.shape[1], n_categories)
        threshold_expanded = threshold.unsqueeze(0).expand(score.shape[0], -1, -1)
        
        # Compute category scores: s = c * score - threshold.
        s = c_expanded * score_expanded - threshold_expanded
        return torch.softmax(s, dim=2)
    
    def compute_loss(self, theta, responses):
        prob = self(theta)
        prob_clamped = torch.clamp(prob, 1e-6, 1.0)
        log_prob = torch.log(prob_clamped)
        gathered_log_prob = log_prob.gather(dim=2, index=responses.unsqueeze(-1)).squeeze(-1)
        return -gathered_log_prob.sum()


class MIRT(nn.Module):
    def __init__(self, n_students, n_items, n_dims, n_categories):
        """
        Base MIRT class for single response model.
        
        Args:
            n_students (int): Number of students.
            n_items (int): Number of items.
            n_dims (int): Latent trait dimensions.
            n_categories (int): Number of categories (for polytomous items).
        """
        super().__init__()
        self.n_students = n_students
        self.n_items = n_items
        self.n_dims = n_dims
        self.n_categories = n_categories
        self.theta = nn.Parameter(torch.randn(n_students, n_dims))
        self.item_model = None
        self.history = None
    
    def specify_model(self, model_type="2PL"):
        
        if model_type == "2PL":
            self.item_model = TwoPLModel(self.n_items, self.n_dims)
        elif model_type == "GPCM":
            self.item_model = GPCMModel(self.n_items, self.n_dims, self.n_categories)
        elif model_type == "1PL":
            self.item_model = OnePLModel(self.n_items, self.n_dims)
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not implemented.")
    
    def forward(self):
        return self.item_model(self.theta)
    
    def compute_loss(self, responses):
        return self.item_model.compute_loss(self.theta, responses)
        
    def fit(self, response, n_epochs=1000, lr=0.01):
        """
        Fit the single-model MIRT.
        
        Args:
            response (np.ndarray): Response matrix of shape (n_students, n_items).
            n_epochs (int): Training epochs.
            lr (float): Learning rate.
        
        Returns:
            self: The trained model.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        response_tensor = torch.tensor(response, dtype=torch.long)
        loss_history = []
        for epoch in tqdm(range(n_epochs), desc="Training MIRT Model", leave=False):
            optimizer.zero_grad()
            loss = self.compute_loss(response_tensor)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        self.history = loss_history
        return self

class JointMIRT(MIRT):
    def __init__(self, n_students, n_dims):
        """
        Joint MIRT class for multiple response models.
        
        Args:
            n_students (int): Number of students.
            n_dims (int): Latent trait dimensions.
        """
        super().__init__(n_students, n_items=0, n_dims=n_dims, n_categories=0)
        self.item_models = nn.ModuleList()
    
    def add_model(self, model_type, n_items, n_categories=None):
        """
        Add an item model to the joint estimation.
        
        Args:
            model_type (str): "2PL" for binary items or "GPCM" for polytomous items.
            n_items (int): Number of items for this model.
            n_categories (int, optional): Number of categories (required for GPCM).
        """
        if model_type == "2PL":
            model = TwoPLModel(n_items, self.n_dims)
        elif model_type == "GPCM":
            if n_categories is None:
                raise ValueError("n_categories must be provided for GPCM")
            model = GPCMModel(n_items, self.n_dims, n_categories)
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not implemented.")
        self.item_models.append(model)
    
    def forward(self):
        """
        Returns a list of probability tensors—one for each item model.
        """
        return [model(self.theta) for model in self.item_models]
    
    def fit(self, responses, n_epochs=1000, lr=0.01):
        """
        Jointly fit all item models using a shared theta.
        
        Args:
            responses (list): A list of response matrices (np.ndarray).
            n_epochs (int): Training epochs.
            lr (float): Learning rate.
        
        Returns:
            self: The trained joint model.
        """
        if not isinstance(responses, list):
            raise ValueError("For JointMIRT, responses must be provided as a list of response matrices.")
        if len(responses) != len(self.item_models):
            raise ValueError("The number of response matrices must equal the number of item models added.")
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        responses_torch = [torch.tensor(resp, dtype=torch.long) for resp in responses]
        loss_history = []
        for epoch in tqdm(range(n_epochs), desc="Training Joint MIRT", leave=False):
            optimizer.zero_grad()
            total_loss = 0
            # Compute loss for each item model using its own compute_loss method.
            for model, resp_tensor in zip(self.item_models, responses_torch):
                loss = model.compute_loss(self.theta, resp_tensor)
                total_loss += loss
            total_loss.backward()
            optimizer.step()
            loss_history.append(total_loss.item())
        self.history = loss_history
        return self
