import subprocess

def ping():
    ip_addresses = [
        "8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", 
        "208.67.222.222", "208.67.220.220", "9.9.9.9", 
        "149.112.112.112", "4.2.2.2", "54.85.114.82"
    ]

    for ip in ip_addresses:
        try:
            response = subprocess.run(["ping", "-c", "1", ip], capture_output=True, text=True)
            print(response.stdout)
        except Exception as e:
            print(f"Failed to ping {ip}: {e}")

ping()