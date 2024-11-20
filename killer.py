import os
import subprocess
import platform
import psutil
import time

def kill_gpu_processes():
    """
    Aggressively kill all GPU-related processes using system commands.
    WARNING: This will forcefully terminate all GPU processes.
    """
    try:
        system = platform.system().lower()
        
        # List of common GPU-using process names
        gpu_process_names = [
            'python', 'python3',
            'jupyter', 'jupyter-notebook', 'jupyter-lab',
            'tensorboard',
            'nvidia-smi',
            'torch', 'pytorch',
            'tensorflow',
            'cuda'
        ]
        
        print("Attempting to kill GPU processes...")
        
        if system == 'windows':
            # Windows commands
            for proc_name in gpu_process_names:
                try:
                    subprocess.run(f'taskkill /F /IM {proc_name}.exe', shell=True)
                except:
                    pass
                    
            # Force kill Python processes
            os.system('wmic process where "commandline like \'%python%\'" delete')
            
        else:
            # Linux/Unix commands
            # First try using nvidia-smi to get GPU process IDs
            try:
                nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode()
                for pid in nvidia_smi_output.strip().split('\n'):
                    if pid:
                        os.system(f'kill -9 {pid}')
            except:
                print("Could not get GPU processes from nvidia-smi")
            
            # Then try killing by process names
            for proc_name in gpu_process_names:
                os.system(f'pkill -9 -f {proc_name}')
            
            # Additional Linux commands to force cleanup
            os.system('killall -9 python python3 2>/dev/null')
            os.system('pkill -9 -f "jupyter|tensor|torch" 2>/dev/null')
        
        # Use psutil as a backup to catch remaining processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                pinfo = proc.info
                cmdline = ' '.join(pinfo['cmdline']).lower() if pinfo['cmdline'] else ''
                if any(gpu_term in cmdline for gpu_term in ['gpu', 'cuda', 'nvidia', 'torch', 'tensorflow']):
                    proc.kill()
            except:
                continue
        
        print("Waiting for processes to terminate...")
        time.sleep(2)  # Give some time for processes to be killed
        
        # Verify if cleanup worked
        if system != 'windows':
            try:
                remaining = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).decode()
                if remaining.strip():
                    print(f"Remaining GPU processes: {remaining}")
                else:
                    print("All GPU processes terminated")
            except:
                print("Could not verify GPU processes")
                
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    print("WARNING: This will forcefully terminate ALL GPU-related processes!")
    print("Make sure to save your work before continuing!")
    response = input("Do you want to continue? (y/n): ")
    
    if response.lower() == 'y':
        kill_gpu_processes()
        print("\nGPU cleanup completed")
    else:
        print("Cleanup cancelled")