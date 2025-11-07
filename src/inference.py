import torch
import pickle
import config
import models
import torchvision
import argparse
import numpy as np
from utils import imshow, preprocess_labels, preprocess_old, generate_noise
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inp", help="Inference input")
args = parser.parse_args()

num_gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if num_gpus >= 1 else "cpu")


def init_inference():
    """Returns initialized models"""
    lm = models.AlphabetLSTM(
        config.NUM_TOKENS,
        config.EMBEDDING_SIZE,
        config.Z_LEN // 2,
        config.NUM_LAYERS,
        config.PADDING_IDX,
    ).to(device)
    gen = models.GeneratorNetwork(
        config.Z_LEN,
        config.CHUNKS,
        config.EMBEDDING_SIZE,
        config.CBN_MLP_DIM,
        config.BATCH_SIZE,
    ).to(device)
    return lm, gen


def init_old():
    """Returns initialized models"""
    lm = models.AlphabetLSTM(
        27,
        config.EMBEDDING_SIZE,
        config.Z_LEN // 2,
        config.NUM_LAYERS,
        config.PADDING_IDX,
    ).to(device)
    gen = models.GeneratorNetwork(
        config.Z_LEN,
        config.CHUNKS,
        config.EMBEDDING_SIZE,
        config.CBN_MLP_DIM,
        config.BATCH_SIZE,
    ).to(device)
    return lm, gen


def inference_tb(inp, writer):
    lm, gen = init_inference()
    checkpoint = torch.load(f"{config.OUT_DIR}/checkpoint.pt", map_location=device, weights_only=False)

    ctoi_file = open(f"{config.BASE_DIR}/src/ctoi.txt", "rb")
    encoding_dict = pickle.load(ctoi_file)
    ctoi_file.close()

    # print(
    #     f'Checkpoint Details:\n Trained for: {checkpoint["epoch"]} epochs, Final Generator loss: {checkpoint["gen_loss"]}, Log File: {checkpoint["log_file"]}'
    # )
    lm.load_state_dict(checkpoint["lm"])
    gen.load_state_dict(checkpoint["gen"])

    test = preprocess_labels([inp] * config.BATCH_SIZE, encoding_dict)
    with torch.no_grad():
        # lm.eval()
        # gen.eval()
        zin = generate_noise(config.Z_LEN, config.BATCH_SIZE, device)
        gin = lm(test.to(device))
        gout = gen(zin, gin)
        tgrid = torchvision.utils.make_grid(gout.detach().cpu(), nrow=4)
        writer.add_image(str(checkpoint["epoch"]), tgrid)

    # print(f'Inference Finished. Check "out" directory for {args.inp}.png')


def inference(inp, filename, current_log):

    os.makedirs(f"{config.OUT_DIR}/{current_log}/inference", exist_ok=True)
    lm, gen = init_inference()
    checkpoint = torch.load(f"{config.OUT_DIR}/{current_log}/checkpoint.pt",  map_location=device, weights_only=False)

    ctoi_file = open(f"{config.BASE_DIR}/src/ctoi.txt", "rb")
    encoding_dict = pickle.load(ctoi_file)
    ctoi_file.close()
    # print(
    #     f'Checkpoint Details:\n Trained for: {checkpoint["epoch"]} epochs, Final Generator loss: {checkpoint["gen_loss"]}, Log File: {checkpoint["log_file"]}'
    # )
    lm.load_state_dict(checkpoint["lm"])
    gen.load_state_dict(checkpoint["gen"])

    test = preprocess_labels([inp] * config.BATCH_SIZE, encoding_dict)
    with torch.no_grad():
        zin = generate_noise(config.Z_LEN, config.BATCH_SIZE, device)
        gin = lm(test.to(device))
        gout = gen(zin, gin)
        tgrid = torchvision.utils.make_grid(gout.detach().cpu(), nrow=4)
        imshow(tgrid, f"{config.OUT_DIR}/{current_log}/inference/{filename}.png")

    print(f'Inference Finished. Check "out" directory for {filename}.png')


if __name__ == "__main__":
    inference(args.inp, args.inp)
