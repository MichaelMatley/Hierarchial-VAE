"""
Learning rate and β-annealing schedules for VAE training.

Provides various scheduling strategies for controlling training dynamics.
"""

import numpy as np
import math


class BetaScheduler:
    """
    β-annealing scheduler for VAE training.
    
    Controls the strength of the KL divergence term over training.
    Starting with β=0 (pure autoencoder) helps prevent posterior collapse.
    
    Supported modes:
        - 'constant': Fixed β throughout training
        - 'linear': Linear increase from 0 to max_beta
        - 'cosine': Cosine annealing schedule
        - 'cyclical': Periodic annealing cycles
        - 'sigmoid': Sigmoid warmup curve
        
    Args:
        mode (str): Scheduling mode
        max_beta (float): Maximum β value
        warmup_epochs (int): Epochs for warmup (for non-constant modes)
        cycle_length (int): Cycle length for cyclical mode
        
    Example:
        >>> scheduler = BetaScheduler(mode='linear', max_beta=1.0, warmup_epochs=20)
        >>> for epoch in range(100):
        ...     beta = scheduler.get_beta(epoch)
        ...     # Use beta in loss calculation
    """
    
    def __init__(self, mode='linear', max_beta=1.0, warmup_epochs=20, cycle_length=20):
        self.mode = mode
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        self.cycle_length = cycle_length
    
    def get_beta(self, epoch):
        """
        Get β value for current epoch.
        
        Args:
            epoch (int): Current training epoch (0-indexed)
            
        Returns:
            float: β value for this epoch
        """
        if self.mode == 'constant':
            return self.max_beta
        
        elif self.mode == 'linear':
            if epoch < self.warmup_epochs:
                return (epoch / self.warmup_epochs) * self.max_beta
            return self.max_beta
        
        elif self.mode == 'cosine':
            if epoch < self.warmup_epochs:
                # Cosine warmup: smooth S-curve from 0 to max_beta
                progress = epoch / self.warmup_epochs
                return self.max_beta * (1 - math.cos(progress * math.pi)) / 2
            return self.max_beta
        
        elif self.mode == 'cyclical':
            # Periodic cycles: useful for encouraging exploration
            cycle_progress = (epoch % self.cycle_length) / self.cycle_length
            return cycle_progress * self.max_beta
        
        elif self.mode == 'sigmoid':
            # Sigmoid warmup: steep middle transition
            if epoch < self.warmup_epochs:
                x = (epoch / self.warmup_epochs) * 12 - 6  # Map to [-6, 6]
                sigmoid = 1 / (1 + math.exp(-x))
                return self.max_beta * sigmoid
            return self.max_beta
        
        else:
            raise ValueError(f"Unknown scheduling mode: {self.mode}")
    
    def get_schedule(self, num_epochs):
        """
        Generate full schedule for visualization.
        
        Args:
            num_epochs (int): Total training epochs
            
        Returns:
            list: β values for each epoch
        """
        return [self.get_beta(epoch) for epoch in range(num_epochs)]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                     num_cycles=0.5, min_lr=0):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps (int): Number of warmup steps
        num_training_steps (int): Total training steps
        num_cycles (float): Number of cosine cycles (0.5 = single decay to min)
        min_lr (float): Minimum learning rate
        
    Returns:
        LambdaLR: Learning rate scheduler
        
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer, 
        ...     num_warmup_steps=1000,
        ...     num_training_steps=10000
        ... )
        >>> for epoch in range(epochs):
        ...     train_one_epoch()
        ...     scheduler.step()
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    
    return LambdaLR(optimizer, lr_lambda)


class CyclicalAnnealingScheduler:
    """
    Cyclical annealing for both β and learning rate.
    
    Implements the schedule from "Cyclical Annealing Schedule: A Simple Approach 
    to Mitigating KL Vanishing" (Fu et al., 2019).
    
    Useful for preventing the model from getting stuck in local minima and
    encouraging more thorough exploration of the latent space.
    
    Args:
        num_cycles (int): Number of annealing cycles
        ratio_increase (float): Proportion of cycle spent increasing β
        ratio_zero (float): Proportion of cycle at β=0
        max_beta (float): Maximum β value
        
    Example:
        >>> scheduler = CyclicalAnnealingScheduler(
        ...     num_cycles=4,
        ...     ratio_increase=0.5,
        ...     ratio_zero=0.25
        ... )
        >>> for epoch in range(100):
        ...     beta = scheduler.get_beta(epoch, total_epochs=100)
    """
    
    def __init__(self, num_cycles=4, ratio_increase=0.5, ratio_zero=0.25, max_beta=1.0):
        self.num_cycles = num_cycles
        self.ratio_increase = ratio_increase
        self.ratio_zero = ratio_zero
        self.max_beta = max_beta
    
    def get_beta(self, epoch, total_epochs):
        """
        Get β value for current epoch within cyclical schedule.
        
        Args:
            epoch (int): Current epoch
            total_epochs (int): Total training epochs
            
        Returns:
            float: β value for this epoch
        """
        # Determine cycle parameters
        epochs_per_cycle = total_epochs / self.num_cycles
        
        # Position within current cycle
        cycle_position = (epoch % epochs_per_cycle) / epochs_per_cycle
        
        # Zero phase
        if cycle_position < self.ratio_zero:
            return 0.0
        
        # Increase phase
        elif cycle_position < (self.ratio_zero + self.ratio_increase):
            increase_progress = (cycle_position - self.ratio_zero) / self.ratio_increase
            return self.max_beta * increase_progress
        
        # Constant phase
        else:
            return self.max_beta
    
    def get_schedule(self, total_epochs):
        """Generate full schedule."""
        return [self.get_beta(epoch, total_epochs) for epoch in range(total_epochs)]


def plot_schedules(num_epochs=100, save_path='schedules.png'):
    """
    Visualize different β-annealing schedules.
    
    Args:
        num_epochs (int): Number of epochs to plot
        save_path (str): Path to save figure
    """
    import matplotlib.pyplot as plt
    
    schedulers = {
        'Linear': BetaScheduler('linear', warmup_epochs=20),
        'Cosine': BetaScheduler('cosine', warmup_epochs=20),
        'Cyclical': BetaScheduler('cyclical', cycle_length=20),
        'Sigmoid': BetaScheduler('sigmoid', warmup_epochs=20),
        'Constant': BetaScheduler('constant')
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, scheduler in schedulers.items():
        schedule = scheduler.get_schedule(num_epochs)
        ax.plot(schedule, label=name, linewidth=2.5, alpha=0.8)
    
    # Add cyclical annealing schedule
    cyclic_scheduler = CyclicalAnnealingScheduler(num_cycles=4)
    cyclic_schedule = cyclic_scheduler.get_schedule(num_epochs)
    ax.plot(cyclic_schedule, label='Cyclical (Fu et al.)', 
            linewidth=2.5, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('β Value', fontsize=12)
    ax.set_title('β-Annealing Schedules Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Schedule visualization saved to {save_path}")


if __name__ == '__main__':
    # Test schedulers
    print("Testing β schedulers...")
    
    linear = BetaScheduler('linear', warmup_epochs=20)
    print(f"Linear schedule (epoch 10): {linear.get_beta(10):.3f}")
    print(f"Linear schedule (epoch 30): {linear.get_beta(30):.3f}")
    
    cyclical = BetaScheduler('cyclical', cycle_length=20)
    print(f"Cyclical schedule (epoch 10): {cyclical.get_beta(10):.3f}")
    print(f"Cyclical schedule (epoch 25): {cyclical.get_beta(25):.3f}")
    
    # Generate visualization
    plot_schedules()