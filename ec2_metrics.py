import numpy as np
import pandas as pd

def generate_synthetic_metrics(n_samples=1000):
    cpu_mean, cpu_std = 40, 10
    disk_read_ops_mean, disk_read_ops_std = 50, 15
    disk_write_ops_mean, disk_write_ops_std = 50, 15
    disk_read_bytes_mean, disk_read_bytes_std = 1e6, 2e5
    disk_write_bytes_mean, disk_write_bytes_std = 1e6, 2e5
    network_in_mean, network_in_std = 5e5, 1e5
    network_out_mean, network_out_std = 5e5, 1e5
    network_packets_in_mean, network_packets_in_std = 300, 50
    network_packets_out_mean, network_packets_out_std = 300, 50

    data = {
        'CPUUtilization': np.random.normal(cpu_mean, cpu_std, n_samples),
        'DiskReadOps': np.random.normal(disk_read_ops_mean, disk_read_ops_std, n_samples),
        'DiskWriteOps': np.random.normal(disk_write_ops_mean, disk_write_ops_std, n_samples),
        'DiskReadBytes': np.random.normal(disk_read_bytes_mean, disk_read_bytes_std, n_samples),
        'DiskWriteBytes': np.random.normal(disk_write_bytes_mean, disk_write_bytes_std, n_samples),
        'NetworkIn': np.random.normal(network_in_mean, network_in_std, n_samples),
        'NetworkOut': np.random.normal(network_out_mean, network_out_std, n_samples),
        'NetworkPacketsIn': np.random.normal(network_packets_in_mean, network_packets_in_std, n_samples),
        'NetworkPacketsOut': np.random.normal(network_packets_out_mean, network_packets_out_std, n_samples),
    }
    
    df = pd.DataFrame(data)
    df['CPUUtilization'] = df['CPUUtilization'].clip(lower=0, upper=100)
    
    anomaly_condition = (
        (df['CPUUtilization'] > 80) |
        (df['DiskReadOps'] > (disk_read_ops_mean + 2 * disk_read_ops_std)) |
        (df['DiskWriteOps'] > (disk_write_ops_mean + 2 * disk_write_ops_std)) |
        (df['DiskReadBytes'] > (disk_read_bytes_mean + 2 * disk_read_bytes_std)) |
        (df['DiskWriteBytes'] > (disk_write_bytes_mean + 2 * disk_write_bytes_std)) |
        (df['NetworkIn'] > (network_in_mean + 2 * network_in_std)) |
        (df['NetworkOut'] > (network_out_mean + 2 * network_out_std)) |
        (df['NetworkPacketsIn'] > (network_packets_in_mean + 2 * network_packets_in_std)) |
        (df['NetworkPacketsOut'] > (network_packets_out_mean + 2 * network_packets_out_std))
    )
    
    df['anomaly'] = anomaly_condition.astype(int)
    return df

if __name__ == "__main__":
    df = generate_synthetic_metrics(n_samples=10000)
    df.to_csv("synthetic_ec2_metrics.csv", index=False)
    print("Synthetic EC2 metrics with anomaly labels generated and written to 'synthetic_ec2_metrics.csv'.")
