import time
import torch
from torch.utils.data import DataLoader, TensorDataset



def benchmark_num_workers(dataset, batch_size, max_workers=8):
    results = {}
    for num_workers in range(max_workers):  # 0 .. max_workers
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers+1,
            pin_memory=False,
        )

        # Warmup (skip first epoch to avoid startup overhead)
        for _ in loader:
            break

        start = time.time()
        for _ in loader:
            pass
        end = time.time()

        duration = end - start
        results[num_workers] = duration
        print(f"num_workers={num_workers+1}: {duration:.2f} seconds")

    return results

if __name__ == '__main__':
    N, D = 10000, 100
    dataset = TensorDataset(torch.randn(N, D), torch.randint(0, 2, (N,)))
    batch_size = 64
    results = benchmark_num_workers(dataset, batch_size, max_workers=8)
    print(results)
