import subprocess
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

import torch

# GPU 메모리 캐시 청소
torch.cuda.empty_cache()

# 이후에 모델 로드 및 연산 수행

def convert_m4a_to_wav(input_file, output_file, sample_rate=8000):
    # ffmpeg를 사용하여 m4a 파일을 wav 파일로 변환하고 샘플링 속도를 조정
    command = ['ffmpeg', '-i', input_file, '-ar', str(sample_rate), '-ac', '1', output_file]
    subprocess.run(command)

# 8000Hz로 샘플링 속도 조정
convert_m4a_to_wav('input.m4a', 'output.wav', 8000)

# SpeechBrain 모델 로드
run_opts={"device":"cuda"}

# SpeechBrain 모델 로드 시 run_opts 사용
model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir='pretrained_models/sepformer-wham', run_opts=run_opts)

# 변환된 wav 파일을 모델에 입력하여 분리
est_sources = model.separate_file(path='output.wav')

# 분리된 소스를 wav 파일로 저장
torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)
