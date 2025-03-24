import hashlib
import time
import os
import platform
import shutil

# Define the hosts file path based on OS
HOSTS_FILE = r"C:\Windows\System32\Drivers\etc\hosts" if platform.system() == "Windows" else "/etc/hosts"
BACKUP_FILE = "ProjectFiles/CYBR-424/hosts_backup.txt"

# List of known malicious domains (for demonstration)
MALICIOUS_DOMAINS = ["malicious.com", "badwebsite.com", "fakebank.com"]

def get_file_hash(file_path):
    """Calculate the hash of the file content."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def backup_hosts_file():
    """Create a backup of the original hosts file."""
    if not os.path.exists(BACKUP_FILE):
        shutil.copy(HOSTS_FILE, BACKUP_FILE)
        print("[INFO] Hosts file backup created.")

def detect_changes():
    """Monitor the hosts file for changes."""
    original_hash = get_file_hash(BACKUP_FILE)
    while True:
        time.sleep(5)  # Check every 5 seconds
        current_hash = get_file_hash(HOSTS_FILE)
        if current_hash != original_hash:
            print("[ALERT] Hosts file has been modified!")
            check_for_malicious_changes()
            original_hash = get_file_hash(HOSTS_FILE)  # Update reference after handling

def check_for_malicious_changes():
    """Detect and fix malicious modifications in the hosts file."""
    with open(HOSTS_FILE, "r") as f:
        lines = f.readlines()
    
    malicious_detected = False
    clean_lines = []

    for line in lines:
        if any(domain in line for domain in MALICIOUS_DOMAINS):
            print(f"[WARNING] Malicious entry detected: {line.strip()}")
            malicious_detected = True
        else:
            clean_lines.append(line)

    if malicious_detected:
        print("[INFO] Restoring clean hosts file...")
        with open(HOSTS_FILE, "w") as f:
            f.writelines(clean_lines)
        print("[SUCCESS] Malicious entries removed and hosts file restored.")

if __name__ == "__main__":
    print("[INFO] Starting hosts file monitor...")
    backup_hosts_file()
    detect_changes()