import torch
import src.config as config
from src.model import HomeAudioCNN

try:
    from torchinfo import summary
except ImportError:
    print("Error: 'torchinfo' not found.")
    exit()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating summary on device: {device}")

    model = HomeAudioCNN(num_classes=config.NUM_CLASSES).to(device)
    
    input_shape = (config.BATCH_SIZE, 3, config.N_MFCCS, 216)
    
    print("-" * 80)
    print(f"Model: HomeAudioCNN")
    print(f"Input Shape: {input_shape}")
    print("-" * 80)

    summary(
        model, 
        input_size=input_shape,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=20,
        verbose=1,
        device=device
    )

if __name__ == "__main__":
    main()