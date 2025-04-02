import json
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
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
    
    def __init__(self, item_params_file, n_dims=2, init_theta=None, randomness=0.2, prior_weight=0.5):
        """
        Initialize the adaptive testing module.
        
        Args:
            item_params_file: Path to JSON file with item parameters
            n_dims: Number of dimensions for the latent trait
            init_theta: Initial theta estimate (defaults to zeros)
            randomness: Probability of selecting a random item instead of most informative
            prior_weight: Weight of the standard normal prior in theta estimation (higher = stronger regularization)
        """
        self.n_dims = n_dims
        self.randomness = randomness
        self.prior_weight = prior_weight
        self.item_pool = self._load_item_params(item_params_file)
        self.available_items = list(range(len(self.item_pool)))
        self.administered_items = []
        self.responses = []
        self.info_gains = []
        self.theta_history = []
        
        # Initialize theta as numpy array
        if init_theta is None:
            self.theta = np.zeros(n_dims)
        else:
            # Ensure initial theta is reasonable
            init_theta = np.array(init_theta)
            self.theta = np.clip(init_theta, -3.0, 3.0)
            
        # Track initial theta
        self.theta_history.append(self.theta.copy())
    
    def _load_item_params(self, item_params_file):
        """Load item parameters from JSON file."""
        with open(item_params_file, 'r') as f:
            item_pool = json.load(f)
        
        # Convert parameters to numpy arrays (we'll convert to tensors as needed)
        for item in item_pool:
            item['a'] = np.array(item['a'], dtype=np.float32)
            item['b'] = np.array(item['b'], dtype=np.float32)
        
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
    
    def _compute_2pl_prob(self, a, b, theta):
        """
        Compute binary item probability using 2PL model.
        
        Args:
            a: Discrimination parameter (numpy array)
            b: Difficulty parameter (numpy array)
            theta: Ability parameter (numpy array)
            
        Returns:
            Probability of correct response (float)
        """
        # Convert to numpy for consistent calculations
        logit = np.dot(theta, a) - b[0]
        return 1.0 / (1.0 + np.exp(-logit))
    
    def _compute_gpcm_probs(self, a, b, theta, n_categories):
        """
        Compute category probabilities using GPCM model.
        
        Args:
            a: Discrimination parameter (numpy array)
            b: Step difficulty parameters (numpy array)
            theta: Ability parameter (numpy array)
            n_categories: Number of response categories
            
        Returns:
            Probabilities for each category (numpy array)
        """
        # Calculate score
        score = np.dot(theta, a)
        
        # Calculate scores for each category
        scores = np.zeros(n_categories)
        
        # First category (c=0) has score 0 (reference category)
        scores[0] = 0.0
        
        # For categories c > 0, calculate cumulative score
        for c in range(1, n_categories):
            if c-1 < len(b):
                scores[c] = c * score - np.sum(b[:c])
        
        # Apply softmax to get probabilities
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)  # Numerical stability
        probs = exp_scores / np.sum(exp_scores)
        
        return probs
    
    def _compute_item_information(self, item, theta):
        """Compute information gain for an item at current theta estimate."""
        item_type, n_categories = self._get_item_type_and_categories(item)
        a = item['a']
        
        if item_type == 'binary':
            # 2PL model information function
            p = self._compute_2pl_prob(a, item['b'], theta)
            info = np.sum((a**2) * p * (1 - p))
        
        elif item_type == 'likert':
            # Calculate probabilities for each category
            probs = self._compute_gpcm_probs(a, item['b'], theta, n_categories)
            
            # Calculate information using variance of expected score
            categories = np.arange(n_categories)
            expected_c = np.sum(categories * probs)
            var_c = np.sum(((categories - expected_c)**2) * probs)
            
            # Information is weighted by squared discrimination
            info = var_c * np.sum(a**2)
        
        return float(info)
    
    def select_next_item(self):
        """Select next item based on information gain with randomness."""
        if not self.available_items:
            return None, None
        
        # With probability 'randomness', select a random item
        if np.random.random() < self.randomness:
            next_item_idx = np.random.choice(self.available_items)
            info_gain = self._compute_item_information(self.item_pool[next_item_idx], self.theta)
            return next_item_idx, info_gain
        
        # Calculate information for all available items
        infos = []
        for idx in self.available_items:
            item = self.item_pool[idx]
            info = self._compute_item_information(item, self.theta)
            infos.append(info)
        
        # Select the item with the highest information
        max_info_idx = np.argmax(infos)
        next_item_idx = self.available_items[max_info_idx]
        info_gain = infos[max_info_idx]
        
        return next_item_idx, info_gain
    
    def _compute_neg_log_likelihood(self, theta_param):
        """
        Compute negative log-likelihood with prior for all administered items given theta.
        
        Args:
            theta_param: Theta parameter as a numpy array
            
        Returns:
            Negative log-likelihood value with prior
        """
        neg_log_likelihood = 0.0
        
        for item_idx, response in zip(self.administered_items, self.responses):
            item = self.item_pool[item_idx]
            item_type, n_categories = self._get_item_type_and_categories(item)
            
            if item_type == 'binary':
                # 2PL model log-likelihood
                p = self._compute_2pl_prob(item['a'], item['b'], theta_param)
                p = np.clip(p, 1e-6, 1-1e-6)  # Numerical stability
                
                if response == 1:
                    neg_log_likelihood -= np.log(p)
                else:
                    neg_log_likelihood -= np.log(1 - p)
            
            elif item_type == 'likert':
                # GPCM model log-likelihood
                probs = self._compute_gpcm_probs(item['a'], item['b'], theta_param, n_categories)
                probs = np.clip(probs, 1e-6, 1.0)  # Numerical stability
                
                # Convert response to 0-based index for indexing into probs
                response_idx = response - 1 if item_type == 'likert' else response
                
                # Add negative log-likelihood for the selected category
                neg_log_likelihood -= np.log(probs[response_idx])
        
        # Add standard normal prior (regularization term)
        # This encourages theta to stay close to standard normal distribution
        prior_penalty = self.prior_weight * np.sum(theta_param**2)
        
        return neg_log_likelihood + prior_penalty
    
    def _estimate_theta(self, n_iterations=100):
        """
        Estimate theta using maximum a posteriori (MAP) estimation with constraints.
        """
        # If no items have been administered yet, return initial theta
        if not self.administered_items:
            return
        
        # Use scipy's minimize function for optimization
        from scipy.optimize import minimize
        
        # Define objective function for minimization
        def objective(theta_flat):
            theta_reshaped = theta_flat.reshape(self.n_dims)
            return self._compute_neg_log_likelihood(theta_reshaped)
        
        # Set bounds to keep theta within reasonable range (-4 to 4)
        # This is approximately 99.99% of the standard normal distribution
        bounds = [(-4.0, 4.0) for _ in range(self.n_dims)]
        
        # Initialize with current theta and optimize
        initial_theta_flat = self.theta.flatten()
        
        # Ensure initial values are within bounds
        initial_theta_flat = np.clip(initial_theta_flat, -4.0, 4.0)
        
        # Use L-BFGS-B method which supports bounds
        result = minimize(
            objective, 
            initial_theta_flat, 
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Update theta with the result
        self.theta = result.x.reshape(self.n_dims)
    
    def update_theta(self, item_idx, response):
        """Update theta estimate based on the response to an item."""
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
        info_gain = self._compute_item_information(item, self.theta)
        
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
        """Get results as a pandas DataFrame in the specified format."""
        rows = []
        
        for i, (item_idx, response, info_gain) in enumerate(zip(
                self.administered_items, self.responses, self.info_gains)):
            
            # Get item details
            item = self.item_pool[item_idx]
            item_type, _ = self._get_item_type_and_categories(item)
            
            # Create administered_item dict with relevant info
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
            
            if verbose:
                print(f"Item {i+1}: {item['item_name']}, Response: {response}, "
                      f"Theta: {self.theta}")
        
        # Return results
        return self.get_results_dataframe()


def run_adaptive_test_simulation(params_file, n_items=10, n_dims=2, init_theta=None, 
                               randomness=0.2, prior_weight=0.5, responses=None, verbose=False, 
                               plot=True, plot_save_path=None):
    """
    Run an adaptive test simulation with specified parameters.
    
    Args:
        params_file: Path to item parameters file
        n_items: Maximum number of items to administer
        n_dims: Number of dimensions for latent trait
        init_theta: Initial theta estimate (defaults to zeros)
        randomness: Probability of selecting a random item
        responses: Pre-determined responses for simulation
        verbose: Whether to print progress information
        plot: Whether to plot theta trajectory
        plot_save_path: Path to save the plot
        
    Returns:
        results_df: DataFrame with test results
        fig: Matplotlib figure of theta trajectory (if plot=True)
        adaptive_test: Testing object for further analysis
    """
    # Initialize adaptive test
    adaptive_test = AdaptiveTesting(
        params_file, 
        n_dims=n_dims, 
        init_theta=init_theta, 
        randomness=randomness,
        prior_weight=prior_weight
    )
    
    # Run the test
    results_df = adaptive_test.run_test(n_items=n_items, responses=responses, verbose=verbose)
    
    # Plot theta trajectory
    fig = None
    if plot:
        fig = adaptive_test.plot_theta_history(save_path=plot_save_path)
    
    return results_df, fig, adaptive_test


if __name__ == "__main__":
    # Example: Run an adaptive test simulation with specific parameters
    fig_dir = 'figures'
    res_dir = 'results'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    for i in tqdm(range(20)):
        results_df, fig, test = run_adaptive_test_simulation(
            params_file='combined_params.json',
            n_items=20,
            n_dims=2,
            init_theta=np.random.randn(2),  # Starting with a non-zero initial theta
            randomness=0.1,          # 30% chance of selecting a random item
            prior_weight=0.2,        # Stronger regularization to ensure standard normal distribution
            verbose=False,
            plot=True
        )
    
        
        results_df.to_csv(os.path.join(res_dir, f'atr_{i}.csv'), index=False)
        fig.savefig(os.path.join(fig_dir, f'theta_trace_{i}.png'))