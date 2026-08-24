import argparse
import gin
import torch

#utils.py
#
#1. eval_mode
#   → 잠깐 RQ-VAE를 eval 상태로 실행
#
#2. parse_config
#   → .gin 설정 파일 읽기

def eval_mode(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to gin config file.")
    args = parser.parse_args()
    gin.parse_config_file(args.config_path)

