import os
import time
import torch
import torchvision.transforms as tfs
from PIL import Image
import gradio as gr
from models.C2PNet import C2PNet

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Starting C2P-Net Server on device: {device}")

MODELS = {
    'Indoor Scene (ITS - 42.56 dB)': 'trained_models/ITS.pkl',
    'Outdoor Scene (OTS - 36.68 dB)': 'trained_models/OTS.pkl'
}

loaded_nets = {}

def get_net(model_key):
    if model_key in loaded_nets:
        return loaded_nets[model_key]
    
    path = MODELS[model_key]
    print(f"[*] Loading checkpoint from {path} onto {device}...")
    net = C2PNet(gps=3, blocks=19).to(device)
    ckp = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckp['model'] if 'model' in ckp else ckp
    net.load_state_dict(state_dict)
    net.eval()
    loaded_nets[model_key] = net
    return net

# Preload default indoor model
try:
    if os.path.exists(MODELS['Indoor Scene (ITS - 42.56 dB)']):
        get_net('Indoor Scene (ITS - 42.56 dB)')
except Exception as e:
    print(f"Preload warning: {e}")

def dehaze_inference(input_img, model_choice, quality_preset):
    if input_img is None:
        return None, "No image provided"
    
    t0 = time.time()
    img = input_img.convert('RGB')
    orig_w, orig_h = img.size
    
    # Cloud CPU optimization: intelligent scaling based on preset
    dim_map = {
        "⚡ Fast (~360p)": 360,
        "⚖️ Balanced (~540p)": 540,
        "💎 High Detail (~720p)": 720
    }
    target_max = dim_map.get(quality_preset, 540)
    
    if max(orig_w, orig_h) > target_max:
        ratio = target_max / max(orig_w, orig_h)
        proc_w, proc_h = int(orig_w * ratio), int(orig_h * ratio)
        proc_img = img.resize((proc_w, proc_h), Image.Resampling.BILINEAR)
    else:
        proc_img = img
        proc_w, proc_h = orig_w, orig_h

    net = get_net(model_choice)
    img_t = tfs.ToTensor()(proc_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = net(img_t)
    
    pred = pred.clamp(0, 1).squeeze(0).cpu()
    out_pil = tfs.ToPILImage()(pred)
    
    # High-quality Lanczos upscale back to original user resolution
    if (proc_w, proc_h) != (orig_w, orig_h):
        out_pil = out_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    
    dt = round(time.time() - t0, 2)
    stats = f"⚡ Dehazed in {dt}s | Quality Profile: {quality_preset} | Resolution: {orig_w}x{orig_h}"
    return out_pil, stats

# Demo Samples
examples_list = []
sample_candidates = [
    ["demo_samples/indoor_1.png", "Indoor Scene (ITS - 42.56 dB)", "⚡ Fast (~360p)"],
    ["demo_samples/indoor_2.png", "Indoor Scene (ITS - 42.56 dB)", "⚖️ Balanced (~540p)"],
    ["demo_samples/outdoor_1.jpg", "Outdoor Scene (OTS - 36.68 dB)", "⚡ Fast (~360p)"],
    ["demo_samples/outdoor_2.jpg", "Outdoor Scene (OTS - 36.68 dB)", "⚖️ Balanced (~540p)"]
]

for img_path, m_choice, q_choice in sample_candidates:
    if os.path.exists(img_path):
        examples_list.append([img_path, m_choice, q_choice])

# Gradio Web Interface
with gr.Blocks(title="C2P-Net Cloud Studio", theme=gr.themes.Soft(primary_hue="cyan")) as demo:
    gr.Markdown(
        """
        # 🌫️ C2P-Net: Physics-Aware Single Image Dehazing
        ### *CVPR 2023 - Curricular Contrastive Regularization for Physics-aware Single Image Dehazing*
        Restore hazy and foggy photos online 24/7 with zero local compute.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Image(type="pil", label="1. Input Hazy Image")
            
            with gr.Row():
                model_select = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value='Indoor Scene (ITS - 42.56 dB)',
                    label="2. Scene Model"
                )
                quality_select = gr.Dropdown(
                    choices=["⚡ Fast (~360p)", "⚖️ Balanced (~540p)", "💎 High Detail (~720p)"],
                    value="⚖️ Balanced (~540p)",
                    label="3. Speed / Quality Profile"
                )
            
            run_btn = gr.Button("✨ Enhance & Dehaze Image", variant="primary", size="lg")
            
        with gr.Column():
            output_box = gr.Image(type="pil", label="4. Restored Clear Image")
            status_text = gr.Markdown("Ready to dehaze.")
            
    run_btn.click(
        fn=dehaze_inference,
        inputs=[input_box, model_select, quality_select],
        outputs=[output_box, status_text]
    )

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[input_box, model_select, quality_select],
            outputs=[output_box, status_text],
            fn=dehaze_inference,
            cache_examples=False,
            label="🖼️ Quick Demo Test Images (Click any image to test)"
        )

if __name__ == "__main__":
    # Render provides PORT environment variable
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
