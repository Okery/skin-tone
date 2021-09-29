import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

from utils import SkinDetector, show


def main(args):
    mean = torch.tensor((0.485, 0.456, 0.406)).reshape(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).reshape(3, 1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model = SkinDetector(args.score_threshold)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    del ckpt

    image = Image.open(args.image_path).convert("RGB")
    image = transforms.ToTensor()(image)
    img = (image - mean) / std

    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        mask = model(img).squeeze(0)

    prefix, ext = os.path.splitext(args.image_path)

    for i, t in enumerate(args.tones):
        show(image, mask, t, "{}_{}{}".format(prefix, i, ext))

    print("Done!")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument('--tones', nargs="+", type=float, default=[-0.4, -0.2, 0.0, 0.2, 0.4])
    args = parser.parse_args()
    
    main(args)
    
    