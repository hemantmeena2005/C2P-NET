import argparse
import os
import torch
import torchvision.transforms as tfs
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
from models.C2PNet import C2PNet


def dehaze_image(net, image_path, device):
    img = Image.open(image_path).convert('RGB')
    img_t = tfs.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = net(img_t)
    pred_clamped = pred.clamp(0, 1).squeeze(0).cpu()
    return pred_clamped


def main():
    parser = argparse.ArgumentParser(description="Run C2PNet Dehazing on single images or a directory.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to hazy image or directory of images')
    parser.add_argument('-m', '--model', type=str, default='trained_models/ITS.pkl', help='Path to model checkpoint (.pkl)')
    parser.add_argument('-o', '--output', type=str, default='results', help='Path to output file or directory')
    parser.add_argument('--blocks', type=int, default=19, help='Number of residual blocks')
    parser.add_argument('--gps', type=int, default=3, help='Number of groups')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Checkpoint not found at '{args.model}'. Please download weights to 'trained_models/' first.")

    net = C2PNet(gps=args.gps, blocks=args.blocks).to(device)
    ckp = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = ckp['model'] if 'model' in ckp else ckp
    net.load_state_dict(state_dict)
    net.eval()

    if os.path.isfile(args.input):
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        out_file = args.output if (args.output.endswith('.png') or args.output.endswith('.jpg')) else os.path.join(args.output, os.path.basename(args.input))
        res = dehaze_image(net, args.input, device)
        vutils.save_image(res, out_file)
        print(f"Saved dehazed image to: {out_file}")
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        images = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for img_name in tqdm(images, desc="Processing images"):
            in_path = os.path.join(args.input, img_name)
            out_path = os.path.join(args.output, img_name)
            res = dehaze_image(net, in_path, device)
            vutils.save_image(res, out_path)
        print(f"Saved all results to: {args.output}")
    else:
        raise ValueError(f"Input path '{args.input}' does not exist.")


if __name__ == '__main__':
    main()
