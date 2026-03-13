import psutil
import time
import os

TIMEOUT = 600

def killer():
    instances = []
    now = time.time()
    for p in psutil.process_iter(attrs=['pid', 'name', 'cmdline', 'create_time']):
        if p.info['cmdline'] and ('--result_file' in p.info['cmdline'] or '--results_file' in p.info['cmdline']) and p.info['name'] == 'python3':
            instances.append((p.info['pid'], now - p.info['create_time']))

    print(f'[!] {TIMEOUT=} {len(instances)=} running {instances=}')
    for (pid, runtime) in instances:
        if runtime > TIMEOUT:
            print('Kill instance:', pid)
            os.system(f'kill -9 {pid}')

if __name__ == "__main__":
    while True:
        try:
            killer()
        except Exception as e:
            print(f'[!] {e=}')
        time.sleep(10)