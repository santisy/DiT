import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", type=str, required=True)
parser.add_argument("-n", "--name", type=str, required=True)
args = parser.parse_args()

# Read the log file
file_path = args.input_file
name = args.name

with open(file_path, 'r') as file:
    log_data = file.readlines()

# Parse the log data
extraction_times = []
refinement_times = []

for line in log_data:
    if 'Time taken Extraction' in line:
        time_str = line.split('Time taken Extraction: ')[1].split(' seconds')[0].strip()
        extraction_times.append(float(time_str))
    elif 'Time taken by Refinement' in line:
        time_str = line.split('Time taken by Refinement: ')[1].split(' seconds')[0].strip()
        refinement_times.append(float(time_str))

# Ensure both lists are of the same length
if len(extraction_times) > len(refinement_times):
    extraction_times = extraction_times[:len(refinement_times)]
else:
    refinement_times = refinement_times[:len(extraction_times)]

print(f"Extraction Time: {np.mean(extraction_times)}+-{2 * np.std(extraction_times)}")
print(f"Refinement Time: {np.mean(refinement_times)}+-{2 * np.std(refinement_times)}")

# Convert to DataFrame for easier plotting
data = pd.DataFrame({
    'Extraction Times': extraction_times,
    'Refinement Times': refinement_times
})

# Plot the histograms with dual x-axes
fig, ax1 = plt.subplots(figsize=(8, 6))

color = 'tab:blue'
ax1.set_xlabel(f'{name} Extraction Time (seconds)', fontsize=18, color=color)
ax1.set_ylabel('Counts', fontsize=18)
ax1.hist(data['Extraction Times'], bins=30, alpha=0.7, label='Extraction Times', color=color, edgecolor='black')
ax1.tick_params(axis='x', labelsize=18, labelcolor=color)
ax1.tick_params(axis='y', labelsize=18)

ax2 = ax1.twiny()  # instantiate a second axes that shares the same y-axis
color = 'tab:red'
ax2.set_xlabel(f'{name} Refinement Time (seconds)', fontsize=18, color=color)
ax2.hist(data['Refinement Times'], bins=30, alpha=0.7, label='Refinement Times', color=color, edgecolor='black')
ax2.tick_params(axis='x', labelsize=18, labelcolor=color)

# Title and legend
fig.legend(loc='center right', fontsize=18)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

# Show the plot
plt.show()

import ace_tools as tools; tools.display_dataframe_to_user(name="Extraction and Refinement Times", dataframe=data)

