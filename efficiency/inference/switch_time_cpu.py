from utils import cpu_time, stable_mean
import torch
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=2048)
    parser.add_argument("--s2_dim", type=int, default=64)
    parser.add_argument("--lora_dim", type=int, default=32)
    args = parser.parse_args()

    input_dim = args.input_dim
    s2_dim = args.s2_dim
    lora_dim = args.lora_dim

    weight = torch.rand(input_dim, input_dim, dtype=torch.bfloat16)
    A = torch.rand(input_dim, lora_dim, dtype=torch.bfloat16)
    B = torch.rand(lora_dim, input_dim, dtype=torch.bfloat16)
    C = torch.rand(s2_dim, input_dim, dtype=torch.bfloat16)
    indices = random.sample(range(input_dim), s2_dim)
    times = []
    for round in range(50):
        start_time = cpu_time()
        weight += torch.matmul(A, B)
        end_time = cpu_time()
        if round >= 10:
            times.append((end_time - start_time))
        if times:
            print(f"LoRA Time (CPU) in Round {round}: {stable_mean(times) * 1000}")
    times = []
    for round in range(50):
        start_time = cpu_time()
        weight[indices] += C
        end_time = cpu_time()
        if round >= 10:
            times.append((end_time - start_time))
        if times:
           print(f"S2FT Time in Round {round}: {stable_mean(times) * 1000}")

if __name__ == "__main__":
    main()
    



