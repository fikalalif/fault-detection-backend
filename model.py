import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
from pathlib import Path

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = smp.Unet(
    encoder_name="resnet34",
    in_channels=3,
    classes=1,
    encoder_weights=None
)
model.load_state_dict(torch.load("models/unet_resnet34.pth", map_location=device))
model.to(device)
model.eval()

# preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict_mask(image_path: str, save_path: str = None):
    """
    Run inference on one image
    """
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)
        pred_mask = (pred > 0.5).float().cpu().squeeze(0).squeeze(0)  # shape HxW

    # convert to PIL
    mask_img = transforms.ToPILImage()(pred_mask)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(save_path)

    return mask_img
