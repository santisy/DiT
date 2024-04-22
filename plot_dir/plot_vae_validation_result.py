import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_validation_log_file(log_file):
    steps = []
    losses = []
    with open(log_file, 'r') as file:
        for line in file:
            if 'Step' in line:
                parts = line.split()
                step = int(parts[1].strip(':'))
                loss = float(parts[2])
                if step % 10000 == 0:
                    steps.append(step)
                    losses.append(loss)
    return steps, losses

def main(log_files):
    plt.figure(figsize=(10, 6))
    colors = ['navy', 'darkgreen', 'darkred', 'black']  # Formal color choices
    markers = ['o', 's', '^', 'D']  # Different markers for each line
    ax = plt.gca()
    
    for i, log_file in enumerate(log_files):
        steps, losses = parse_validation_log_file(log_file)
        plt.plot(steps, losses, label=f'{log_file.split("/")[-1].split(".")[0]}',
                 color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle='-', linewidth=3)

    plt.xlabel('Validation at Iters', fontsize=20)
    plt.ylabel('Validation Result', fontsize=20)
    plt.legend(fontsize=20)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)  # Make left spine thicker
    ax.spines['bottom'].set_linewidth(2)  # Make bottom spine thicke
    plt.savefig("VAE_val_compare.pdf", format='pdf', dpi=300, bbox_inches='tight')  # Save as PDF with 300 dpi
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot validation loss from log files.')
    parser.add_argument('log_files', type=str, nargs='+', help='Path to the log files to parse and plot')
    args = parser.parse_args()
    main(args.log_files)

