import subprocess

def run_snort_interface():
    subprocess.run(["snort", "-A", "console", "-i", "eth0", "-c", "/etc/snort/snort.conf"])

run_snort_interface()