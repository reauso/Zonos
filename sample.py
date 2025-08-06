import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

print('Load Model')
# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

print('Load Sample Audio file and make embedding')
# wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
wav, sampling_rate = torchaudio.load("assets/Rene_120s_DE.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

print('Prepare')
# cond_dict = make_cond_dict(text="Hi my name is Rene and my favourite color is blue.", speaker=speaker, language="de-de")
cond_dict = make_cond_dict(
    text="Hallo, mein Name ist Rene und meine Lieblingsfarbe ist Blau.",
    speaker=speaker,
    language="de",
)
conditioning = model.prepare_conditioning(cond_dict)

print('Generate Code')
codes = model.generate(conditioning)

print('Create wav')
wavs = model.autoencoder.decode(codes).cpu()

print('Export Wav')
torchaudio.save("sample_de.wav", wavs[0], model.autoencoder.sampling_rate)
