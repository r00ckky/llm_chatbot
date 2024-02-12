from TTS.tts.configs.tortoise_config import TortoiseConfig
from TTS.tts.models.tortoise import Tortoise

config = TortoiseConfig()
model = Tortoise.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)

# with random speaker
output_dict = model.synthesize(text, config, speaker_id="random", extra_voice_dirs=None, **kwargs)
