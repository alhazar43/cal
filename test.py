import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt


class AdaptiveTesting:
    """
    Adaptive testing module for Item Response Theory (IRT).
    
    Selects items based on information gain with randomness and updates
    student's latent trait (theta) estimate after each response.
    
    Supports:
    1. Likert scale 1-5 (RIASEC items)
    2. Likert scale 1-7 (TIPI items)
    3. Binary 0/1 (VCL items)
    """
    
    def __init__(self, item_params_file, n_dims=2, init_theta=None, randomness=0.2):
        """
        Initialize the adaptive testing module.
        
        Args:
            item_params_file: Path to JSON file with item parameters
            n_dims: Number of dimensions for the latent trait
            init_theta: Initial theta estimate (defaults to zeros)
            randomness: Probability of selecting a random item instead of most informative
        """
        self.n_dims = n_dims
        self.randomness = randomness
        self.item_pool = self._load_item_params(item_params_file)
        self.available_items = list(range(len(self.item_pool)))
        self.administered_items = []
        self.responses = []
        self.info_gains = []
        self.theta_history = []
        
        # Initialize theta
        if init_theta is None:
            self.theta = np.zeros(n_dims)
        else:
            self.theta = np.array(init_theta)
            
        # Track initial theta
        self.theta_history.append(self.theta.copy())
            
        # Convert theta to PyTorch tensor
        self.theta_tensor = torch.tensor(self.theta, dtype=torch.float32).view(1, n_dims)
    
    def _load_item_params(self, item_params_file):
        """Load item parameters from JSON file."""
        with open(item_params_file, 'r') as f:
            item_pool = json.load(f)
        
        # Convert parameters to PyTorch tensors
        for item in item_pool:
            item['a'] = torch.tensor(item['a'], dtype=torch.float32)
            item['b'] = torch.tensor(item['b'], dtype=torch.float32)
        
        return item_pool
    
    def _get_item_type_and_categories(self, item):
        """Get item type and number of response categories."""
        item_type = item['item_type']
        
        if item_type == 'binary':
            return 'binary', 2
        elif "TIPI" in item['item_name']:
            return 'likert', 7  # 1-7 scale for TIPI
        else:
            return 'likert', 5  # 1-5 scale for RIASEC
    
    def _compute_response_prob(self, item, theta):
        """Compute response probabilities based on item type."""
        item_type, n_categories = self._get_item_type_and_categories(item)
        a = item['a']
        b = item['b']
        
        if item_type == 'binary':
            # 2PL model
            logit = torch.matmul(theta, a) - b[0]
            return torch.sigmoid(logit)
        else:
            # GPCM model
            score = torch.matmul(theta, a)
            
            # Calculate scores for each category
            scores = torch.zeros(n_categories, dtype=torch.float32)
            
            # First category (c=0) has score 0 (reference category)
            scores[0] = 0.0
            
            # For categories c > 0, calculate cumulative score
            for c in range(1, n_categories):
                if c-1 < len(b):
                    scores[c] = c * score - torch.sum(b[:c])
            
            # Apply softmax to get probabilities
            max_score = torch.max(scores)
            exp_scores = torch.exp(scores - max_score)  # Numerical stability
            probs = exp_scores / torch.sum(exp_scores)
            
            return probs.squeeze()
    
    def _compute_item_information(self, item, theta):
        """Compute information gain for an item at current theta estimate."""
        item_type, n_categories = self._get_item_type_and_categories(item)
        a = item['a']
        
        if item_type == 'binary':
            # 2PL model information function
            p = self._compute_response_prob(item, theta)
            info = torch.sum((a**2) * p * (1 - p))
        
        elif item_type == 'likert':
            # Calculate probabilities for each category
            probs = self._compute_response_prob(item, theta)
            
            # Calculate information using variance of expected score
            categories = torch.arange(n_categories, dtype=torch.float32)
            expected_c = torch.sum(categories * probs)
            var_c = torch.sum(((categories - expected_c)**2) * probs)
            
            # Information is weighted by squared discrimination
            info = var_c * torch.sum(a**2)
        
        return info.item()
    
    def select_next_item(self):
        """Select next item based on information gain with randomness."""
        if not self.available_items:
            return None, None
        
        # With probability 'randomness', select a random item
        if np.random.random() < self.randomness:
            next_item_idx = np.random.choice(self.available_items)
            info_gain = self._compute_item_information(self.item_pool[next_item_idx], self.theta_tensor)
            return next_item_idx, info_gain
        
        # Calculate information for all available items
        infos = []
        for idx in self.available_items:
            item = self.item_pool[idx]
            info = self._compute_item_information(item, self.theta_tensor)
            infos.append(info)
        
        # Select the item with the highest information
        max_info_idx = np.argmax(infos)
        next_item_idx = self.available_items[max_info_idx]
        info_gain = infos[max_info_idx]
        
        return next_item_idx, info_gain
    
    def _compute_negative_log_likelihood(self, item, response, theta):
        """Compute negative log-likelihood for a response to an item."""
        item_type, _ = self._get_item_type_and_categories(item)
        
        if item_type == 'binary':
            # 2PL model log-likelihood
            p = self._compute_response_prob(item, theta)
            p = torch.clamp(p, 1e-6, 1-1e-6)  # Numerical stability
            
            if response == 1:
                neg_log_likelihood = -torch.log(p)
            else:
                neg_log_likelihood = -torch.log(1 - p)
        
        elif item_type == 'likert':
            # GPCM model log-likelihood
            probs = self._compute_response_prob(item, theta)
            probs = torch.clamp(probs, 1e-6, 1.0)  # Numerical stability
            
            # Convert response to 0-based index for indexing into probs
            response_idx = response - 1 if item_type == 'likert' else response
            
            # Return negative log-likelihood for the selected category
            neg_log_likelihood = -torch.log(probs[response_idx])
        
        return neg_log_likelihood
    
    def _estimate_theta(self, n_iterations=50, lr=0.05):

        # Initialize theta as parameter to optimize
        theta_param = nn.Parameter(self.theta_tensor.clone())
        optimizer = optim.Adam([theta_param], lr=lr)
        
        # Optimize for n iterations
        for _ in range(n_iterations):
            optimizer.zero_grad()
            
            # Compute negative log-likelihood
            neg_log_likelihood = 0
            for item_idx, response in zip(self.administered_items, self.responses):
                item = self.item_pool[item_idx]
                neg_log_likelihood += self._compute_negative_log_likelihood(item, response, theta_param)
            
            neg_log_likelihood.backward()
            optimizer.step()
        
        # Update theta
        self.theta_tensor = theta_param.detach().clone()
        self.theta = self.theta_tensor.squeeze().numpy()
    
    def update_theta(self, item_idx, response):
        # Validate response
        item = self.item_pool[item_idx]
        item_type, n_categories = self._get_item_type_and_categories(item)
        
        if item_type == 'binary':
            if response not in [0, 1]:
                raise ValueError(f"Binary item response must be 0 or 1, got {response}")
        else:  # likert
            if response < 1 or response >= n_categories:
                valid_range = f"1-{n_categories-1}"
                raise ValueError(f"Likert response must be in range {valid_range}, got {response}")
        
        # Calculate information gain
        info_gain = self._compute_item_information(item, self.theta_tensor)
        
        # Add the administered item and response to the history
        self.administered_items.append(item_idx)
        self.responses.append(response)
        self.info_gains.append(info_gain)
        
        # Remove the item from available items
        if item_idx in self.available_items:
            self.available_items.remove(item_idx)
        
        # Update theta using maximum likelihood estimation
        self._estimate_theta()
        
        # Store updated theta in history
        self.theta_history.append(self.theta.copy())
        
        return self.theta.copy(), info_gain
    
    def get_results_dataframe(self):
        rows = []
        
        for i, (item_idx, response, info_gain) in enumerate(zip(
                self.administered_items, self.responses, self.info_gains)):
            
            # Get item details
            item = self.item_pool[item_idx]
            item_type, _ = self._get_item_type_and_categories(item)
            

            administered_item = {
                'item_id': item_idx,
                'type': item_type,
                'a': item['a'].tolist(),
                'b': item['b'].tolist(),
                'item_name': item['item_name']
            }
            
            # Get theta values after this response
            theta = self.theta_history[i+1]
            
            # Build the row
            row = {
                'est_theta_1': theta[0],
                'est_theta_2': theta[1] if len(theta) > 1 else None,
                'item_type': item_type,
                'response': response,
                'info_gain': info_gain,
                'administered_item': administered_item
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_theta_history(self, save_path=None):
        """Plot the trajectory of theta values over time."""
        theta_history = np.array(self.theta_history)
        n_dims = theta_history.shape[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each dimension
        for dim in range(n_dims):
            ax.plot(range(len(theta_history)), theta_history[:, dim], 
                    marker='o', label=f'Dimension {dim+1}')
        
        # Add annotations for administered items
        for i, item_idx in enumerate(self.administered_items):
            item_name = self.item_pool[item_idx]['item_name']
            # Annotate at the i+1 position (since theta_history starts with initial theta)
            ax.annotate(item_name, (i+1, theta_history[i+1, 0]), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        ax.set_xlabel('Item Number')
        ax.set_ylabel('Theta Value')
        ax.set_title('Theta Estimates After Each Item Response')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def run_test(self, n_items=None, responses=None, verbose=False):
        """Run a complete adaptive test and return results dataframe."""
        if n_items is None:
            n_items = len(self.item_pool)
        
        n_items = min(n_items, len(self.item_pool))
        
        for i in range(n_items):
            # Select next item
            item_idx, info_gain = self.select_next_item()
            if item_idx is None:
                if verbose:
                    print("No more items available.")
                break
            
            # Get item details
            item = self.item_pool[item_idx]
            item_type, n_categories = self._get_item_type_and_categories(item)
            
            # Get or generate response
            if responses is not None and i < len(responses):
                response = responses[i]
            else:
                # Simulate response
                if item_type == 'binary':
                    response = np.random.choice([0, 1])
                else:  # likert
                    response = np.random.choice(range(1, n_categories))
            
            # Update theta
            self.update_theta(item_idx, response)
        
        # Return results
        return self.get_results_dataframe()


def run_adaptive_test_simulation(params_file, n_items=10, n_dims=2, init_theta=None, 
                               randomness=0.2, responses=None, verbose=False, plot=True,
                               plot_save_path=None):

    adaptive_test = AdaptiveTesting(
        params_file, 
        n_dims=n_dims, 
        init_theta=init_theta, 
        randomness=randomness
    )
    

    results_df = adaptive_test.run_test(n_items=n_items, responses=responses, verbose=verbose)

    fig = None
    if plot:
        fig = adaptive_test.plot_theta_history(save_path=plot_save_path)
    
    return results_df, fig, adaptive_test


if __name__ == "__main__":

    results_df, fig, test = run_adaptive_test_simulation(
        params_file='combined_params.json',
        n_items=20,
        n_dims=2,
        init_theta=np.random.normal(0, 1, 2),  
        randomness=0.1,          
        verbose=False,
        plot=True,
        plot_save_path='theta_plot.png'
    )
    
    # Display the results table
    results_df.to_csv('adaptive_test_results.csv', index=False)