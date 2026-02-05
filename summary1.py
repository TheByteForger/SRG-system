# visualize_architecture.py
import torch
from torchviz import make_dot
import src.config as config
from src.model import HomeAudioCNN

def generate_large_diagram():

    device = torch.device('cpu')
    model = HomeAudioCNN(num_classes=config.NUM_CLASSES).to(device)
    x = torch.randn(1, 3, 40, 216).to(device)


    y = model(x)

    dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

    dot.attr(dpi="300")
    dot.attr(rankdir="TB")
    dot.attr(fontsize="20")
    dot.attr(concentrate="true")
    
    filename = "home_audio_cnn_large"
    dot.format = 'png'
    dot.render(filename)
    
    print(f"High-Res Diagram saved to: {filename}.png")
    print("(If the PNG is still too small, check the generated PDF file if available, as it has infinite resolution.)")

if __name__ == "__main__":
    generate_large_diagram()