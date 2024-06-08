# minimal-ddpmp

## Installation and running

```bash
# optinal virtual env to keep it clean
python3 -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
# to replicate just run 
python train.py
# or also you can set this args
python train.py --image_size 128 --train_batch_size 64 --num_epochs 50 --learning_rate 1e-4
```

## Results after 50 epoch

![image](ddpm-butterflies-128/samples/0049.png)

## HFU Calculation UNet

I couldn't find an MFU calculation for UNet, so I estimated the HFU with the help of [FlopCounter](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py).

### Approach

Trying different batch sizes and adjusting model dimensions was my first step. This approach aims to reduce the number of memory accesses for the same training, thereby increasing the HFU.

### Batch Size Experimentation

- Increasing the batch size to **72** yielded total FLOPs per step of **26.8T** compared to **23.8T** per step for a batch size of 64.

### Performance Metrics

- With the lowest step time of 0.3 seconds and the default **23.8T** FLOPs per step, I was able to achieve:
  - Operations per second = \(23.8 \times 10^{12}\) / 0.3 = \(79.33 \times 10^{12}\) operations/second
  - Given the promised **312T** operations per second (A100), this results in an HFU of approximately 25.43%.

### Calculating HFU for Batch Size 72

- Total FLOPs per step: **26.8T**
- Step time: 0.3 seconds
  - Operations per second = \(26.8 \times 10^{12}\) / 0.3 = \(89.33 \times 10^{12}\) operations/second
  - HFU = (\(89.33 \times 10^{12}\)) / (\(312 \times 10^{12}\)) = 28.63%

### Observations

- So far, `torch.compile` increases the speed of intermediate steps but not the average step time, indicating that UNet might not be ideal for `torch.compile`.
- With different shapes, I might be able to get the Triton compiler to access memory more efficiently, but I have not tried that yet.
- Changing to [Denoising Diffusion Implicit Models (DDIM)](https://huggingface.co/papers/2010.02502) did not change total FLOPs per step and moreover degraded convergence for now.
