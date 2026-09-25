import os
import json
import matplotlib.pyplot as plt

def plot_curves():
    hist_path = os.path.join(os.path.dirname(__file__), 'results', 'dan_dg_history.json')
    if not os.path.exists(hist_path):
        print(f"Could not find {hist_path}. You must run training for DAN-DG first!")
        return

    with open(hist_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Classification Loss', color=color, fontsize=12)
    ax1.plot(epochs, history['cls_loss'], color=color, linewidth=2, label='Cls Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('MMD Penalty', color=color, fontsize=12)
    ax2.plot(epochs, history['mmd_loss'], color=color, linestyle='--', linewidth=2, label='MMD Penalty')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Task 3: DAN-DG Training Curves (Cls Loss vs MMD)')
    fig.tight_layout()
    
    plots_dir = os.path.join(os.path.dirname(__file__), 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'dan_dg_training_curves.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved DAN-DG training curves to {out_path}")

if __name__ == '__main__':
    plot_curves()
