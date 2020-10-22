# LambdaNetworks: Modeling long-range Interactions without Attention

## Experimnets (CIFAR10)

| Model | k | h | u | m | Params (M) | Acc (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ResNet18 baseline ([ref](https://github.com/kuangliu/pytorch-cifar)) ||||| 14 | 93.02
| LambdaResNet18 | 16 | 4 | 4 | 7 | 8.6 | 93.20 (65 Epochs) |
| LambdaResNet18 | 16 | 4 | 4 | 5 | 8.6 | 91.58 (70 Epochs) |
| LambdaResNet18 | 16 | 4 | 1 | 23 | 8 | wip |
| ResNet50 baseline ([ref](https://github.com/kuangliu/pytorch-cifar)) ||||| 23.5 | 93.62 |
| LambdaResNet50 | 16 | 4 | 4 | 7 | 13 | wip |

## Usage
```python
import torch

from model import LambdaConv, LambdaResNet50, LambdaResNet152

x = torch.randn([2, 3, 32, 32])
conv = LambdaConv(3, 128)
print(conv(x).size()) # [2, 128, 32, 32]

# reference
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

model = LambdaResNet50()
print(get_n_params(model)) # 14.9M (Ours) / 15M(Paper)

model = LambdaResNet152()
print(get_n_params(model)) # 32.8M (Ours) / 35M (Paper)
```

## train

```bash
$cd LambdaNetworks
$python main.py --model_name LambdaResNet50
```

## few train logs
```bash
LambdaResNet50 Model Parameters: 15.00 M Params
load checkpoint if needed
start iteration
[Epoch:  1 100/391]  Loss:30.9855, Acc: 10.2336  |  Correct  1323/12928   | 0.13sec/iter  12.72sec/100iter 
[Epoch:  1 200/391]  Loss:20.9384, Acc: 10.0319  |  Correct  2581/25728   | 0.13sec/iter  13.12sec/100iter 
[Epoch:  1 300/391]  Loss:15.9862, Acc: 10.0135  |  Correct  3858/38528   | 0.13sec/iter  12.77sec/100iter  52.15sec/epoch | eval | Acc: 10.120  |  1012 / 10000   | 3.83sec/eval | Best Model Saving... | Learning Rate:[0.1] -->[0.09755307053217621]
[Epoch:  2 100/391]  Loss: 2.9403, Acc: 10.1330  |  Correct  1310/12928   | 0.13sec/iter  12.78sec/100iter 
[Epoch:  2 200/391]  Loss: 2.8210, Acc:  9.9580  |  Correct  2562/25728   | 0.13sec/iter  12.77sec/100iter 
[Epoch:  2 300/391]  Loss: 2.7090, Acc:  9.9252  |  Correct  3824/38528   | 0.13sec/iter  12.85sec/100iter  52.47sec/epoch | eval | Acc: 10.080  |  1008 / 10000   | 3.87sec/eval | Learning Rate:[0.09755307053217621] -->[0.0904518046337755]
[Epoch:  3 100/391]  Loss: 2.3494, Acc:  9.9551  |  Correct  1287/12928   | 0.13sec/iter  13.45sec/100iter 
[Epoch:  3 200/391]  Loss: 2.3581, Acc:  9.7909  |  Correct  2519/25728   | 0.13sec/iter  12.70sec/100iter 
[Epoch:  3 300/391]  Loss: 2.3697, Acc:  9.7851  |  Correct  3770/38528   | 0.13sec/iter  12.77sec/100iter  52.51sec/epoch | eval | Acc: 10.050  |  1005 / 10000   | 3.85sec/eval | Learning Rate:[0.0904518046337755] -->[0.0793913236883622]
[Epoch:  4 100/391]  Loss: 2.4642, Acc: 10.0866  |  Correct  1304/12928   | 0.13sec/iter  12.78sec/100iter 
[Epoch:  4 200/391]  Loss: 2.4064, Acc: 10.0941  |  Correct  2597/25728   | 0.13sec/iter  13.00sec/100iter 
[Epoch:  4 300/391]  Loss: 2.3866, Acc: 10.0758  |  Correct  3882/38528   | 0.13sec/iter  12.87sec/100iter  52.59sec/epoch | eval | Acc: 10.110  |  1011 / 10000   | 3.90sec/eval | Learning Rate:[0.0793913236883622] -->[0.0654543046337755]
[Epoch:  5 100/391]  Loss: 2.3285, Acc: 10.0557  |  Correct  1300/12928   | 0.13sec/iter  13.12sec/100iter 
[Epoch:  5 200/391]  Loss: 2.3313, Acc: 10.1096  |  Correct  2601/25728   | 0.13sec/iter  13.06sec/100iter 
[Epoch:  5 300/391]  Loss: 2.3316, Acc: 10.1770  |  Correct  3921/38528   | 0.13sec/iter  12.77sec/100iter  52.54sec/epoch | eval | Acc: 10.170  |  1017 / 10000   | 3.85sec/eval | Best Model Saving... | Learning Rate:[0.0654543046337755] -->[0.05000500000000001]
```



## Parameters
| Model | k | h | u | m | Params (M), Paper | Params (M), Ours |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|LambdaResNet50| 16 | 4 | 1 | 23 | 15.0 | 14.9 |
|LambdaResNet50| 16 | 4 | 4 | 7 | 16.0 | 16.0 |
|LambdaResNet152| 16 | 4 | 1 | 23 | 35 | 32.8 |
|LambdaResNet200| 16 | 4 | 1 | 23 | 42 | 35.29 |

## Ablation Parameters
| k | h | u | Params (M), Paper | Params (M), Ours |
|:-:|:-:|:-:|:-:|:-:|
| ResNet baseline ||| 25.6 | 25.5
| 8 | 2 | 1 | 14.8 | 15.0 |
| 8 | 16 | 1 | 15.6 | 14.9 |
| 2 | 4 | 1 | 14.7 | 14.6 |
| 4 | 4 | 1 | 14.7 | 14.66 |
| 8 | 4 | 1 | 14.8 | 14.66 |
| 16 | 4 | 1 | 15.0 | 14.99 |
| 32 | 4 | 1 | 15.4 | 15.4 |
| 2 | 8 | 1 | 14.7 | 14.5 |
| 4 | 8 | 1 | 14.7 | 14.57 |
| 8 | 8 | 1 | 14.7 | 14.74 |
| 16 | 8 | 1 | 15.1 | 14.1 |
| 32 | 8 | 1 | 15.7 | 15.76 |
| 8 | 8 | 4 | 15.3 | 15.26 |
| 8 | 8 | 8 | 16.0 | 16.0 |
| 16 | 4 | 4 | 16.0 | 16.0 |
