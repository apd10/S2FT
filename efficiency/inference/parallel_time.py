from utils import cuda_time, stable_mean
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=2048)
    parser.add_argument("--s2_dim", type=int, default=64)
    parser.add_argument("--lora_dim", type=int, default=32)
    parser.add_argument("--num_adapters", type=int, default=10)
    args = parser.parse_args()

    input_dim = args.input_dim
    n = args.num_adapters
    s2_dim = args.s2_dim
    lora_dim = args.lora_dim
    x = torch.rand(n, input_dim, dtype=torch.bfloat16).cuda()
    weight = torch.rand(input_dim, input_dim, dtype=torch.bfloat16).cuda()
    A = torch.rand(n, input_dim, lora_dim, dtype=torch.bfloat16).cuda()
    B = torch.rand(n, lora_dim, input_dim, dtype=torch.bfloat16).cuda()
    C = torch.rand(n, s2_dim, input_dim, dtype=torch.bfloat16).cuda()
    D = torch.rand(n, input_dim, s2_dim, dtype=torch.bfloat16).cuda()
    indices = torch.stack([torch.randperm(input_dim)[:s2_dim] for _ in range(n)]).cuda()
    times = []
    for round in range(50):
        start_time = cuda_time()
        x1 = torch.matmul(x, weight)
        x1 += torch.einsum('nb,nbd->nd', torch.einsum('nb,nbd->nd', x, A), B)
        end_time = cuda_time()
        if round >= 10:
            times.append((end_time - start_time))
        if times:
            print(f"LoRA Time in Round {round}: {stable_mean(times) * 1000}")
    times = []
    for round in range(50):
        start_time = cuda_time()
        x1 = torch.matmul(x, weight)
        x1.scatter_add_(1, indices, torch.einsum('nb,nbd->nd', x, D))
        end_time = cuda_time()
        if round >= 10:
            times.append((end_time - start_time))
        if times:
            print(f"S2FT Time 1 in Round {round}: {stable_mean(times) * 1000}")
    times = []
    for round in range(50):
        start_time = cuda_time()
        x1 = torch.matmul(x, weight)
        x1 += torch.einsum('nb,nbd->nd', torch.gather(x, 1, indices), C)
        end_time = cuda_time()
        if round >= 10:
            times.append((end_time - start_time))
        if times:
            print(f"S2FT Time 2 in Round {round}: {stable_mean(times) * 1000}")

if __name__ == "__main__":
    main()
    



