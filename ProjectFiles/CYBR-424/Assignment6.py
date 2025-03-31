import os
import time
import subprocess
import re
from datetime import datetime, timedelta

# Configuration
ICMP_THRESHOLD = 50  # Max allowed pings per minute
SSH_THRESHOLD = 5  # Max failed SSH attempts per hour
MOTD_FILE = "/etc/motd"
SSH_LOG_FILE = "/var/log/auth.log"
ICMP_LOG_FILE = "/var/log/icmp_alert.log"
SSH_ALERT_LOG_FILE = "/var/log/ssh_alert.log"

def detect_icmp_flood():
    """Monitors ICMP traffic using tcpdump and detects excessive pings."""
    try:
        print("[INFO] Monitoring ICMP traffic...")
        result = subprocess.run(
            ["timeout", "60", "tcpdump", "-i", "eth0", "icmp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        icmp_count = len(result.stdout.splitlines())

        if icmp_count > ICMP_THRESHOLD:
            alert_message = f"[ALERT] ICMP Ping Flood detected! ({icmp_count} pings in last minute)\n"
            print(alert_message)
            log_alert(ICMP_LOG_FILE, alert_message)
            update_motd(alert_message)
    except Exception as e:
        print(f"[ERROR] ICMP detection failed: {e}")

def detect_ssh_bruteforce():
    """Checks SSH log file for excessive failed login attempts."""
    try:
        print("[INFO] Monitoring SSH login attempts...")
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        with open(SSH_LOG_FILE, "r") as log_file:
            logs = log_file.readlines()
        
        failed_attempts = sum(1 for line in logs if "Failed password" in line and 
                              parse_log_time(line) > one_hour_ago)

        if failed_attempts > SSH_THRESHOLD:
            alert_message = f"[ALERT] Excessive SSH login attempts detected! ({failed_attempts} failed attempts in last hour)\n"
            print(alert_message)
            log_alert(SSH_ALERT_LOG_FILE, alert_message)
            update_motd(alert_message)
    except Exception as e:
        print(f"[ERROR] SSH detection failed: {e}")

def parse_log_time(log_line):
    """Extracts the timestamp from a log line and returns a datetime object."""
    try:
        log_time_str = " ".join(log_line.split()[:3])  # Example: "Oct 31 14:15:22"
        log_time = datetime.strptime(log_time_str, "%b %d %H:%M:%S")
        log_time = log_time.replace(year=datetime.now().year)
        return log_time
    except ValueError:
        return datetime.min

def update_motd(alert_message):
    """Updates the /etc/motd file with the latest alert."""
    try:
        with open(MOTD_FILE, "a") as motd:
            motd.write(alert_message)
    except Exception as e:
        print(f"[ERROR] Failed to update MOTD: {e}")

def log_alert(log_file, message):
    """Logs alerts to a specified file."""
    with open(log_file, "a") as log:
        log.write(f"{datetime.now()} - {message}")

def main():
    """Runs the intrusion detection system continuously."""
    while True:
        detect_icmp_flood()
        detect_ssh_bruteforce()
        time.sleep(60)

if __name__ == "__main__":
    main()