import torch
print(torch.cuda.is_available())  # Returns True if CUDA is available, False otherwise
print(torch.cuda.device_count())  # Returns the number of GPUs available
print(torch.cuda.get_device_name(0))  # Returns the name of the first GPU

'''import sys
print(sys.executable)
print(sys.path)'''