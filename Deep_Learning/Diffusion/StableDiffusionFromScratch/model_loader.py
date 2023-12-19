from Deep_Learning.Diffusion.StableDiffusionFromScratch.nn.clip import CLIP
from Deep_Learning.Diffusion.StableDiffusionFromScratch.nn.vae_encoder import VAE_Encoder
from Deep_Learning.Diffusion.StableDiffusionFromScratch.nn.vae_decoder import VAE_Decoder
from Deep_Learning.Diffusion.StableDiffusionFromScratch.nn.diffusion import Diffusion
from Deep_Learning.Diffusion.StableDiffusionFromScratch.model_converter import load_from_standard_weights


def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
