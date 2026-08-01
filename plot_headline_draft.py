import matplotlib.pyplot as plt

# 1. Define the four difficulty splits for the X-axis
levels = ['Warm', 'Cold-Drug', 'Cold-Target', 'Cold-Pair']

# 2. Fake placeholder data (Y-axis)
fake_fidelity = [0.62, 0.51, 0.47, 0.35]      # Precision@k scores
fake_accuracy = [0.89, 0.81, 0.78, 0.69]      # AUROC/AUPRC scores

# 3. Set up the plot
plt.figure(figsize=(8, 5))

# Plot Explanation fidelity
plt.plot(levels, fake_fidelity, marker='o', linestyle='-', color='blue', 
         label='Explanation fidelity (precision@k)', linewidth=2)

# Plot Standard prediction accuracy
plt.plot(levels, fake_accuracy, marker='s', linestyle='--', color='red', 
         label='Prediction accuracy (AUROC)', linewidth=2)

# 4. Add labels, title, and formatting
plt.ylabel('Score')
plt.title('DRAFT MOCK-UP: Model Accuracy vs. Explanation Fidelity\n(*Replace with real results later*)', 
          fontweight='bold', color='gray')
plt.ylim(0, 1) # Scores are bounded between 0 and 1
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

# 5. Save the figure as requested by the guide
output_filename = 'headline_figure_draft.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Draft figure successfully generated and saved as '{output_filename}'!")
plt.show()
