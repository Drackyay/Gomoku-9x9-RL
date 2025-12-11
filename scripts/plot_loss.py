"""
Plot initial loss from loss_history.csv (first 100 rows)
"""

import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Get project root
project_root = Path(__file__).parent
csv_path = project_root / "metrics" / "loss_history.csv"

# Read data
initial_losses = []

print(f"Reading {csv_path}...")

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    for i, row in enumerate(reader):
        if i >= 100:  # Only first 100 rows
            break
        
        try:
            initial_loss = float(row['initial_loss'])
            initial_losses.append(initial_loss)
        except (ValueError, KeyError) as e:
            print(f"Warning: Skipping row {i+2} due to error: {e}")
            continue

# Use row order (1 to 100) as iteration numbers
iterations = list(range(1, len(initial_losses) + 1))

print(f"Loaded {len(initial_losses)} data points")

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(iterations, initial_losses, 'b-', linewidth=1.5, marker='o', markersize=3, alpha=0.7)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Over 100 Iterations', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add statistics
mean_loss = np.mean(initial_losses)
min_loss = np.min(initial_losses)
max_loss = np.max(initial_losses)
last_loss = initial_losses[-1] if initial_losses else 0

# Add text box with statistics
stats_text = f'Mean: {mean_loss:.4f}\nMin: {min_loss:.4f}\nMax: {max_loss:.4f}\nLast: {last_loss:.4f}'
plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save plot
output_path = project_root / "metrics" / "initial_loss_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
print(f"Statistics:")
print(f"  Mean loss: {mean_loss:.4f}")
print(f"  Min loss: {min_loss:.4f}")
print(f"  Max loss: {max_loss:.4f}")
print(f"  Last loss: {last_loss:.4f}")

