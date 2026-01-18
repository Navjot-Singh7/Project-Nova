import torch

def check_cuda():
    print("python executable:", sys.executable)
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cudnn.version():", torch.backends.cudnn.version())
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    
    if torch.cuda.is_available():
        try:
            print("CUDA device name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("Failed to get device name:", e)
        
        # quick tensor test
        try:
            x = torch.rand(3, 3).to('cuda')
            print("Tensor on CUDA:", x)
        except Exception as e:
            print("Tensor CUDA test failed:", e)
    else:
        print("CUDA is not available according to PyTorch.")

    # Try system checks
    print('\n-- System checks --')
    print('nvidia-smi:')
    print(run_cmd(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_capability', '--format=csv,noheader']))
    print('\nnvcc --version:')
    print(run_cmd(['nvcc', '--version']))

if __name__ == "__main__":
    import sys
    import subprocess
    import torch


    def run_cmd(cmd):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=False, text=True)
            return out.strip()
        except Exception as e:
            return f"(failed) {e}"


    def check_cuda():
        print("python executable:", sys.executable)
        print("torch.__version__:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("torch.backends.cudnn.version():", torch.backends.cudnn.version())
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("torch.cuda.device_count():", torch.cuda.device_count())

        if torch.cuda.is_available():
            try:
                print("CUDA device name:", torch.cuda.get_device_name(0))
            except Exception as e:
                print("Failed to get device name:", e)

            # quick tensor test
            try:
                x = torch.rand(3, 3).to('cuda')
                print("Tensor on CUDA:", x)
            except Exception as e:
                print("Tensor CUDA test failed:", e)
        else:
            print("CUDA is not available according to PyTorch.")

        # Try system checks
        print('\n-- System checks --')
        print('nvidia-smi:')
        print(run_cmd(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_capability', '--format=csv,noheader']))
        print('\nnvcc --version:')
        print(run_cmd(['nvcc', '--version']))


    if __name__ == '__main__':
        check_cuda()
