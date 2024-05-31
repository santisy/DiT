import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')  # Use 'valid' to avoid boundary effects
    return np.concatenate([y[:box_pts//2], y_smooth, y[-box_pts//2+1:]])

def parse_log_file(log_file):
    losses = []
    with open(log_file, 'r') as file:
        for line in file:
            if 'Train Loss:' in line:
                parts = line.split()
                loss_index = parts.index('Loss:') + 1
                loss = float(parts[loss_index].replace(',', ''))
                if loss < 500:
                    losses.append(loss)
    return losses

def main(log_files, start_ratio, smooth_factor=10):
    plt.figure(figsize=(10, 6))
    colors = ['navy', 'darkgreen', 'darkred', 'black']  # Formal color choices
    ax = plt.gca()
    for i, log_file in enumerate(log_files):
        losses = parse_log_file(log_file)
        min_length = min(map(len, [parse_log_file(f) for f in log_files]))
        losses = losses[:min_length]
        start_at = int(len(losses) * start_ratio)
        losses = losses[start_at:]
        if smooth_factor > 1:
            losses_smoothed = smooth(losses, smooth_factor)
        else:
            losses_smoothed = losses
        plt.plot(np.arange(start_at, min_length) * 100, losses_smoothed, label=f'{log_file.split("/")[-1].split(".")[0]}', color=colors[i % len(colors)], linewidth=3)
    
    #plt.title('Training Loss Over Time', fontsize=18)
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Training Loss', fontsize=20)
    plt.legend(fontsize=20)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)  # Make left spine thicker
    ax.spines['bottom'].set_linewidth(2)  # Make bottom spine thicke
    plt.savefig("VAE_train_compare.pdf", format='pdf', dpi=300, bbox_inches='tight')  # Save as PDF with 300 dpi
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training loss from log files.')
    parser.add_argument('log_files', type=str, nargs='+', help='Path to the log files to parse and plot')
    parser.add_argument("-s", "--start_ratio", type=float, default=1.0)
    parser.add_argument("-w", "--smooth_window", type=int, default=10)
    args = parser.parse_args()
    main(args.log_files, args.start_ratio, args.smooth_window)

