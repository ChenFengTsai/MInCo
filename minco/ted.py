"""
TED (Temporal Disentanglement) module for MInCo
Based on "Temporal Disentanglement of Representations for Improved Generalisation in Reinforcement Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


class TEDClassifier(nn.Module):
    """
    TED classifier for temporal disentanglement
    Distinguishes between temporal and non-temporal observation pairs
    """
    
    def __init__(self, embedding_size, hidden_dim=None, dropout=0.1):
        super(TEDClassifier, self).__init__()
        self.embedding_size = embedding_size
        
        if hidden_dim is None:
            hidden_dim = min(embedding_size, 256)
        
        # Simple neural network classifier
        self.network = nn.Sequential(
            nn.Linear(2 * embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, samples):
        """
        Forward pass for TED classifier
        
        Args:
            samples: [batch_size, 2, embedding_size] - pairs of embeddings
        
        Returns:
            logits: [batch_size, 1] - classification logits
        """
        batch_size = samples.shape[0]
        x1 = samples[:, 0]  # [batch_size, embedding_size]
        x2 = samples[:, 1]  # [batch_size, embedding_size]
        
        # Concatenate and classify
        concat_features = torch.cat([x1, x2], dim=1)
        return self.network(concat_features)


class TEDClassifierOriginal(nn.Module):
    """
    Original TED classifier implementation following the paper exactly
    Uses the mathematical formulation from Equation 1 in TED paper
    """
    
    def __init__(self, embedding_size):
        super(TEDClassifierOriginal, self).__init__()
        self.embedding_size = embedding_size
        
        # Parameters for TED classifier (from paper Equation 1)
        self.k1 = nn.Parameter(torch.empty(embedding_size))
        self.k2 = nn.Parameter(torch.empty(embedding_size))
        self.b = nn.Parameter(torch.empty(embedding_size))
        self.k_bar = nn.Parameter(torch.empty(embedding_size))
        self.b_bar = nn.Parameter(torch.empty(embedding_size))
        self.c = nn.Parameter(torch.empty(1))
        
        # Initialize parameters
        nn.init.normal_(self.k1, std=0.02)
        nn.init.normal_(self.k2, std=0.02)
        nn.init.normal_(self.b, std=0.02)
        nn.init.normal_(self.k_bar, std=0.02)
        nn.init.normal_(self.b_bar, std=0.02)
        nn.init.normal_(self.c, std=0.02)
    
    def forward(self, samples):
        """
        Apply TED classification based on Equation 1 in the TED paper
        y(x_t) = Σ|k1_i * z_i(x1) + k2_i * z_i(x2) + b_i| - (k̄_i * z_i(x1) + b̄_i)² + c
        """
        batch_size = samples.shape[0]
        x1 = samples[:, 0]  # [batch_size, embedding_size]
        x2 = samples[:, 1]  # [batch_size, embedding_size]
        
        # Equation 1 from TED paper
        linear_term = torch.abs(self.k1 * x1 + self.k2 * x2 + self.b)
        marginal_term = torch.square(self.k_bar * x1 + self.b_bar)
        
        # Sum across feature dimension
        linear_sum = torch.sum(linear_term, dim=1)
        marginal_sum = torch.sum(marginal_term, dim=1)
        
        # Final output
        output = linear_sum - marginal_sum + self.c
        return output.view(batch_size, 1)


class TEDTargetEncoder(nn.Module):
    """
    Target encoder wrapper for TED with exponential moving average updates
    """
    
    def __init__(self, main_encoder, tau=0.01):
        super(TEDTargetEncoder, self).__init__()
        self.tau = tau
        
        # Create target encoder as copy of main encoder
        self.target_encoder = type(main_encoder)(
            main_encoder.state_based,
            main_encoder.obs_size,
            main_encoder.embedding_size,
            main_encoder.activation_function
        )
        
        # Initialize target with same weights as main
        self.update_target_encoder(main_encoder, tau=1.0)
    
    def update_target_encoder(self, main_encoder, tau=None):
        """Update target encoder with EMA of main encoder params"""
        if tau is None:
            tau = self.tau
        
        with torch.no_grad():
            for target_param, main_param in zip(
                self.target_encoder.parameters(), 
                main_encoder.parameters()
            ):
                target_param.data.copy_(
                    (1 - tau) * target_param.data + tau * main_param.data
                )
    
    def __call__(self, *args, **kwargs):
        """Forward pass through target encoder (no gradients)"""
        with torch.no_grad():
            return self.target_encoder(*args, **kwargs)


class TEDLoss(nn.Module):
    """
    TED loss computation module
    Handles the creation of temporal/non-temporal pairs and loss calculation
    """
    
    def __init__(self, config, classifier_type='simple'):
        super(TEDLoss, self).__init__()
        self.config = config
        
        # TED parameters
        self.ted_coefficient_start = getattr(config, 'ted_coefficient_start', 0.0)
        self.ted_coefficient_end = getattr(config, 'ted_coefficient_end', 0.1)
        self.ted_warmup_ratio = getattr(config, 'ted_warmup_ratio', 0.2)
        self.use_target_encoder = getattr(config, 'use_target_encoder', False)
        
        # Calculate warmup steps
        total_steps = getattr(config, 'steps', 500000)
        self.ted_warmup_steps = int(total_steps * self.ted_warmup_ratio)
        
        # Environment-specific adjustments
        env_name = getattr(config, 'env_name', '').lower()
        # if 'cheetah' in env_name:
        #     self.ted_coefficient_end *= 0.3
        #     print(f"Cheetah environment detected: TED coefficient reduced to {self.ted_coefficient_end}")
        
        # Initialize classifier
        if classifier_type == 'simple':
            self.classifier = TEDClassifier(config.embedding_size)
        elif classifier_type == 'original':
            self.classifier = TEDClassifierOriginal(config.embedding_size)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Target encoder (if requested)
        self.target_encoder = None
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def initialize_target_encoder(self, main_encoder):
        """Initialize target encoder if needed"""
        if self.use_target_encoder:
            self.target_encoder = TEDTargetEncoder(
                main_encoder, 
                tau=getattr(self.config, 'target_tau', 0.01)
            )
    
    def get_ted_coefficient(self, step):
        """Get current TED coefficient with warmup schedule"""
        if step < self.ted_warmup_steps:
            progress = step / self.ted_warmup_steps
            return self.ted_coefficient_start + progress * (
                self.ted_coefficient_end - self.ted_coefficient_start)
        return self.ted_coefficient_end
    
    def update_target_encoder(self, main_encoder):
        """Update target encoder if it exists"""
        if self.target_encoder is not None:
            self.target_encoder.update_target_encoder(main_encoder)
    
    def forward(self, embeds, nonterms, step, main_encoder=None):
        """
        Compute TED loss
        
        Args:
            embeds: [batch_size, seq_len, embedding_size] - encoded observations
            nonterms: [batch_size, seq_len, 1] - non-terminal mask
            step: current training step
            main_encoder: main encoder (needed if using target encoder)
        
        Returns:
            ted_loss: scalar tensor
            ted_coefficient: current coefficient value
        """
        ted_coefficient = self.get_ted_coefficient(step)
        
        if ted_coefficient == 0:
            return torch.tensor(0.0, device=embeds.device), ted_coefficient
        
        # Choose loss computation method
        if self.use_target_encoder and self.target_encoder is not None:
            ted_loss = self._compute_ted_loss_with_target(embeds, nonterms, main_encoder)
        else:
            ted_loss = self._compute_ted_loss_simple(embeds, nonterms)
        
        return ted_loss, ted_coefficient
    
    def _compute_ted_loss_simple(self, embeds, nonterms):
        """
        TED loss using single encoder (recommended approach)
        """
        batch_size, seq_len, embed_dim = embeds.shape
        
        if seq_len < 4:  # Need sufficient sequence length
            return torch.tensor(0.0, device=embeds.device)
        
        all_samples = []
        all_labels = []
        valid_mask = nonterms.squeeze(-1)  # [batch_size, seq_len]
        
        # 1. TEMPORAL PAIRS (consecutive timesteps) - label = 1
        num_temporal_samples = min(3, seq_len - 1)
        for i in range(num_temporal_samples):
            t = i
            if t + 1 >= seq_len:
                continue
                
            valid_transitions = valid_mask[:, t] * valid_mask[:, t + 1]
            if valid_transitions.sum() == 0:
                continue
            
            obs_t = embeds[:, t]      # [batch_size, embed_dim]
            obs_t1 = embeds[:, t + 1] # [batch_size, embed_dim]
            
            temporal_samples = torch.stack([obs_t, obs_t1], dim=1)
            temporal_labels = torch.ones((batch_size, 1), device=embeds.device)
            
            # Filter valid transitions
            temporal_samples = temporal_samples[valid_transitions.bool()]
            temporal_labels = temporal_labels[valid_transitions.bool()]
            
            if len(temporal_samples) > 0:
                all_samples.append(temporal_samples)
                all_labels.append(temporal_labels)
        
        # 2. LARGE GAP PAIRS (same episode, non-consecutive) - label = 0
        min_gap = max(seq_len // 3, 4)  # At least 1/3 of sequence or 4 timesteps
        if seq_len > min_gap:
            for t in range(seq_len - min_gap):
                t_future = t + min_gap
                if t_future >= seq_len:
                    break
                
                valid_transitions = valid_mask[:, t] * valid_mask[:, t_future]
                if valid_transitions.sum() == 0:
                    continue
                
                obs_t = embeds[:, t]
                obs_t_future = embeds[:, t_future]
                
                large_gap_samples = torch.stack([obs_t, obs_t_future], dim=1)
                large_gap_labels = torch.zeros((batch_size, 1), device=embeds.device)
                
                large_gap_samples = large_gap_samples[valid_transitions.bool()]
                large_gap_labels = large_gap_labels[valid_transitions.bool()]
                
                if len(large_gap_samples) > 0:
                    all_samples.append(large_gap_samples)
                    all_labels.append(large_gap_labels)
        
        # 3. CROSS-EPISODE PAIRS (different episodes) - label = 0
        rnd_idx = torch.randperm(batch_size, device=embeds.device)
        t1 = torch.randint(0, seq_len, (1,)).item()
        t2 = torch.randint(0, seq_len, (1,)).item()
        
        obs_ep1 = embeds[:, t1]
        obs_ep2 = embeds[rnd_idx, t2]  # Different episodes
        
        cross_episode_samples = torch.stack([obs_ep1, obs_ep2], dim=1)
        cross_episode_labels = torch.zeros((batch_size, 1), device=embeds.device)
        
        all_samples.append(cross_episode_samples)
        all_labels.append(cross_episode_labels)
        
        if not all_samples:
            return torch.tensor(0.0, device=embeds.device)
        
        # Combine all samples and compute loss
        samples = torch.cat(all_samples)
        labels = torch.cat(all_labels)
        
        predictions = self.classifier(samples)
        return self.loss_fn(predictions, labels)
    
    def _compute_ted_loss_with_target(self, embeds, nonterms, main_encoder):
        """
        TED loss using target encoder approach
        """
        if main_encoder is None:
            raise ValueError("main_encoder required when using target encoder")
        
        batch_size, seq_len, embed_dim = embeds.shape
        
        if seq_len < 2:
            return torch.tensor(0.0, device=embeds.device)
        
        # Get target embeddings (this would need the original observations)
        # For now, implement similar to simple version but with target encoder concept
        # This is a simplified version - full implementation would need access to original obs
        
        all_samples = []
        all_labels = []
        valid_mask = nonterms.squeeze(-1)
        
        # Temporal pairs using current and "target" (next timestep) embeddings
        for t in range(seq_len - 1):
            valid_transitions = valid_mask[:, t] * valid_mask[:, t + 1]
            if valid_transitions.sum() == 0:
                continue
            
            obs_t = embeds[:, t]      # Main encoder at time t
            obs_t1 = embeds[:, t + 1] # "Target" encoder at time t+1
            
            temporal_samples = torch.stack([obs_t, obs_t1], dim=1)
            temporal_labels = torch.ones((batch_size, 1), device=embeds.device)
            
            temporal_samples = temporal_samples[valid_transitions.bool()]
            temporal_labels = temporal_labels[valid_transitions.bool()]
            
            if len(temporal_samples) > 0:
                all_samples.append(temporal_samples)
                all_labels.append(temporal_labels)
        
        # Non-temporal pairs (similar to simple version)
        rnd_idx = torch.randperm(batch_size, device=embeds.device)
        t1 = torch.randint(0, seq_len, (1,)).item()
        t2 = torch.randint(0, seq_len, (1,)).item()
        
        obs_ep1 = embeds[:, t1]
        obs_ep2 = embeds[rnd_idx, t2]
        
        non_temporal_samples = torch.stack([obs_ep1, obs_ep2], dim=1)
        non_temporal_labels = torch.zeros((batch_size, 1), device=embeds.device)
        
        all_samples.append(non_temporal_samples)
        all_labels.append(non_temporal_labels)
        
        if not all_samples:
            return torch.tensor(0.0, device=embeds.device)
        
        samples = torch.cat(all_samples)
        labels = torch.cat(all_labels)
        
        predictions = self.classifier(samples)
        return self.loss_fn(predictions, labels)


class TEDModule(nn.Module):
    """
    Complete TED module for integration with MInCo
    """
    
    def __init__(self, config, classifier_type='simple'):
        super(TEDModule, self).__init__()
        self.config = config
        self.ted_loss = TEDLoss(config, classifier_type)
        
        # Store whether TED is enabled
        self.enabled = getattr(config, 'use_ted', False)
    
    def initialize_target_encoder(self, main_encoder):
        """Initialize target encoder if needed"""
        if self.enabled:
            self.ted_loss.initialize_target_encoder(main_encoder)
    
    def parameters(self):
        """Return TED parameters for optimizer"""
        if self.enabled:
            return self.ted_loss.classifier.parameters()
        return []
    
    def compute_loss(self, embeds, nonterms, step, main_encoder=None):
        """
        Compute TED loss
        
        Returns:
            ted_loss: scalar tensor (0 if TED disabled)
            ted_coefficient: current coefficient value
            metrics: dict of logging metrics
        """
        if not self.enabled:
            return torch.tensor(0.0, device=embeds.device), 0.0, {}
        
        ted_loss, ted_coefficient = self.ted_loss(embeds, nonterms, step, main_encoder)
        
        metrics = {
            'ted_loss': ted_loss.item(),
            'ted_coefficient': ted_coefficient
        }
        
        return ted_loss, ted_coefficient, metrics
    
    def update_target_encoder(self, main_encoder):
        """Update target encoder if it exists"""
        if self.enabled:
            self.ted_loss.update_target_encoder(main_encoder)