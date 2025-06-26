import psutil
import GPUtil
from tabulate import tabulate

def monitor_resources():
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count(logical=True)

    # RAM usage
    virtual_memory = psutil.virtual_memory()
    ram_total = virtual_memory.total / (1024 ** 2)  # Convert to MB
    ram_used = virtual_memory.used / (1024 ** 2)  # Convert to MB
    ram_percent = virtual_memory.percent

    # Disk usage
    disk_usage = psutil.disk_usage('/')
    disk_total = disk_usage.total / (1024 ** 3)  # Convert to GB
    disk_used = disk_usage.used / (1024 ** 3)  # Convert to GB
    disk_percent = disk_usage.percent

    # GPU usage (if available)
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "Name": gpu.name,
            "Load (%)": gpu.load * 100,
            "Memory Used (MB)": gpu.memoryUsed,
            "Memory Total (MB)": gpu.memoryTotal,
            "Memory Utilization (%)": gpu.memoryUtil * 100
        })

    # Display results
    print("=== System Resource Usage ===")
    print(f"CPU Usage: {cpu_percent}% ({cpu_count} cores)")
    print(f"RAM Usage: {ram_used:.2f} MB / {ram_total:.2f} MB ({ram_percent}%)")
    print(f"Disk Usage: {disk_used:.2f} GB / {disk_total:.2f} GB ({disk_percent}%)")

    if gpu_info:
        print("\n=== GPU Usage ===")
        print(tabulate(gpu_info, headers="keys", tablefmt="pretty"))
    else:
        print("\nNo GPU detected.")

if __name__ == "__main__":
    monitor_resources()