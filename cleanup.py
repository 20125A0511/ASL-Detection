import os
import signal
import subprocess
import sys

def kill_process_on_port(port):
    try:
        # Find process using the port
        cmd = f"lsof -i :{port} -t"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if stdout:
            pid = stdout.decode().strip()
            print(f"Found process {pid} using port {port}")
            os.kill(int(pid), signal.SIGTERM)
            print(f"Killed process {pid}")
        else:
            print(f"No process found using port {port}")
    except Exception as e:
        print(f"Error killing process on port {port}: {str(e)}")

def main():
    # Kill processes on common ports
    ports = [5000, 5001, 5002, 5003, 5004]
    for port in ports:
        kill_process_on_port(port)
    
    # Kill any Python processes that might be running the app
    try:
        subprocess.run(["pkill", "-f", "python.*app.py"], check=False)
        print("Killed any existing Python app processes")
    except Exception as e:
        print(f"Error killing Python processes: {str(e)}")

if __name__ == "__main__":
    main() 