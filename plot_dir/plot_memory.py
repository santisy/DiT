import matplotlib.pyplot as plt

# Function to read memory records from a file and plot them
def plot_memory_usage(file_path):
    # Read memory records from the file
    with open(file_path, 'r') as file:
        memory_records = [float(line.strip()) for line in file]
    
    # Plot the memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(memory_records, label='Total Memory Usage (MB)')
    plt.xlabel('Time (record index)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Total Memory Usage Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# Assuming the memory records are stored in 'memory_records.txt'
plot_memory_usage('total_mem.txt')

