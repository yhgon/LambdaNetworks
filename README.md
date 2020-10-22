# LambdaNetworks: Modeling long-range Interactions without Attention

Community Pytorch implementation for [LambdaNetworks: Modeling long-range Interactions without Attention ](https://openreview.net/pdf?id=xTJEN-ggl1b ). the code is based on [leaderj1001](https://github.com/leaderj1001/LambdaNetworks)'s work and maintained by Hyungon Ryu (NVIDIA AI Tech. Center).  

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

## few train logs for LambdaResNet50 with CIFAR10
```
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
[Epoch:  6 100/391]  Loss: 2.3258, Acc: 10.1949  |  Correct  1318/12928   | 0.13sec/iter  12.84sec/100iter 
[Epoch:  6 200/391]  Loss: 2.3225, Acc: 10.4128  |  Correct  2679/25728   | 0.13sec/iter  12.83sec/100iter 
[Epoch:  6 300/391]  Loss: 2.3216, Acc: 10.2393  |  Correct  3945/38528   | 0.13sec/iter  12.89sec/100iter  52.60sec/epoch | eval | Acc: 10.010  |  1001 / 10000   | 3.87sec/eval | Learning Rate:[0.05000500000000001] -->[0.03455569536622451]
[Epoch:  7 100/391]  Loss: 2.3331, Acc:  9.9087  |  Correct  1281/12928   | 0.13sec/iter  12.86sec/100iter 
[Epoch:  7 200/391]  Loss: 2.3250, Acc:  9.7637  |  Correct  2512/25728   | 0.13sec/iter  12.87sec/100iter 
[Epoch:  7 300/391]  Loss: 2.3213, Acc:  9.7877  |  Correct  3771/38528   | 0.13sec/iter  12.86sec/100iter  52.53sec/epoch | eval | Acc: 10.170  |  1017 / 10000   | 3.87sec/eval | Learning Rate:[0.03455569536622451] -->[0.020618676311637812]
[Epoch:  8 100/391]  Loss: 2.3109, Acc:  9.6071  |  Correct  1242/12928   | 0.13sec/iter  13.09sec/100iter 
[Epoch:  8 200/391]  Loss: 2.3124, Acc: 10.0513  |  Correct  2586/25728   | 0.13sec/iter  12.83sec/100iter 
[Epoch:  8 300/391]  Loss: 2.3113, Acc: 10.0654  |  Correct  3878/38528   | 0.13sec/iter  12.74sec/100iter  52.60sec/epoch | eval | Acc: 10.130  |  1013 / 10000   | 3.89sec/eval | Learning Rate:[0.020618676311637812] -->[0.009558195366224508]
[Epoch:  9 100/391]  Loss: 2.3083, Acc: 10.0015  |  Correct  1293/12928   | 0.13sec/iter  12.77sec/100iter 
[Epoch:  9 200/391]  Loss: 2.3075, Acc: 10.2612  |  Correct  2640/25728   | 0.13sec/iter  12.81sec/100iter 
[Epoch:  9 300/391]  Loss: 2.3073, Acc: 10.3379  |  Correct  3983/38528   | 0.13sec/iter  12.98sec/100iter  52.48sec/epoch | eval | Acc: 10.140  |  1014 / 10000   | 3.87sec/eval | Learning Rate:[0.009558195366224508] -->[0.0024569294678237993]
[Epoch: 10 100/391]  Loss: 2.3077, Acc: 10.4579  |  Correct  1352/12928   | 0.13sec/iter  12.92sec/100iter 
[Epoch: 10 200/391]  Loss: 2.3064, Acc: 10.4322  |  Correct  2684/25728   | 0.13sec/iter  12.84sec/100iter 
[Epoch: 10 300/391]  Loss: 2.3057, Acc: 10.3042  |  Correct  3970/38528   | 0.13sec/iter  13.11sec/100iter  52.44sec/epoch | eval | Acc: 10.020  |  1002 / 10000   | 3.87sec/eval | Learning Rate:[0.0024569294678237993] -->[0.1]
[Epoch: 11 100/391]  Loss: 2.3245, Acc: 10.0170  |  Correct  1295/12928   | 0.13sec/iter  12.90sec/100iter 
[Epoch: 11 200/391]  Loss: 2.3299, Acc: 10.1718  |  Correct  2617/25728   | 0.13sec/iter  12.80sec/100iter 
[Epoch: 11 300/391]  Loss: 2.3383, Acc: 10.0914  |  Correct  3888/38528   | 0.13sec/iter  12.80sec/100iter  52.42sec/epoch | eval | Acc: 10.130  |  1013 / 10000   | 3.85sec/eval | Learning Rate:[0.1] -->[0.09938447858805392]
[Epoch: 12 100/391]  Loss: 2.3278, Acc:  9.9551  |  Correct  1287/12928   | 0.13sec/iter  12.91sec/100iter 
[Epoch: 12 200/391]  Loss: 2.3273, Acc: 10.1407  |  Correct  2609/25728   | 0.13sec/iter  12.91sec/100iter 
[Epoch: 12 300/391]  Loss: 2.3268, Acc: 10.2523  |  Correct  3950/38528   | 0.13sec/iter  12.81sec/100iter  52.55sec/epoch | eval | Acc: 10.120  |  1012 / 10000   | 3.87sec/eval | Learning Rate:[0.09938447858805392] -->[0.09755307053217621]
[Epoch: 13 100/391]  Loss: 2.3684, Acc:  9.5606  |  Correct  1236/12928   | 0.13sec/iter  12.89sec/100iter 
[Epoch: 13 200/391]  Loss: 2.3870, Acc:  9.5771  |  Correct  2464/25728   | 0.13sec/iter  12.90sec/100iter 
[Epoch: 13 300/391]  Loss: 2.3653, Acc:  9.7306  |  Correct  3749/38528   | 0.13sec/iter  13.10sec/100iter  52.43sec/epoch | eval | Acc: 10.070  |  1007 / 10000   | 3.86sec/eval | Learning Rate:[0.09755307053217621] -->[0.09455087117679745]
[Epoch: 14 100/391]  Loss: 2.3200, Acc: 10.3032  |  Correct  1332/12928   | 0.13sec/iter  12.98sec/100iter 
[Epoch: 14 200/391]  Loss: 2.3195, Acc: 10.3467  |  Correct  2662/25728   | 0.13sec/iter  12.86sec/100iter 
[Epoch: 14 300/391]  Loss: 2.3203, Acc: 10.1900  |  Correct  3926/38528   | 0.13sec/iter  12.78sec/100iter  52.55sec/epoch | eval | Acc: 10.140  |  1014 / 10000   | 3.85sec/eval | Learning Rate:[0.09455087117679745] -->[0.0904518046337755]

...

[Epoch:306 100/391]  Loss: 0.0170, Acc: 99.4585  |  Correct 12858/12928   | 0.13sec/iter  12.63sec/100iter 
[Epoch:306 200/391]  Loss: 0.0169, Acc: 99.4520  |  Correct 25587/25728   | 0.13sec/iter  12.67sec/100iter 
[Epoch:306 300/391]  Loss: 0.0171, Acc: 99.4342  |  Correct 38310/38528   | 0.13sec/iter  12.64sec/100iter  51.31sec/epoch | eval | Acc: 71.810  |  7181 / 10000   | 3.84sec/eval | Learning Rate:[0.00025073959002352015] -->[0.00016411790001226746]
[Epoch:307 100/391]  Loss: 0.0180, Acc: 99.4740  |  Correct 12860/12928   | 0.13sec/iter  12.60sec/100iter 
[Epoch:307 200/391]  Loss: 0.0184, Acc: 99.4364  |  Correct 25583/25728   | 0.13sec/iter  12.66sec/100iter 
[Epoch:307 300/391]  Loss: 0.0175, Acc: 99.4757  |  Correct 38326/38528   | 0.13sec/iter  12.60sec/100iter  51.38sec/epoch | eval | Acc: 71.830  |  7183 / 10000   | 3.81sec/eval | Learning Rate:[0.00016411790001226746] -->[9.67108188151324e-05]
[Epoch:308 100/391]  Loss: 0.0166, Acc: 99.4972  |  Correct 12863/12928   | 0.13sec/iter  12.55sec/100iter 
[Epoch:308 200/391]  Loss: 0.0159, Acc: 99.5336  |  Correct 25608/25728   | 0.13sec/iter  12.61sec/100iter 
[Epoch:308 300/391]  Loss: 0.0160, Acc: 99.5354  |  Correct 38349/38528   | 0.13sec/iter  12.70sec/100iter  51.41sec/epoch | eval | Acc: 71.810  |  7181 / 10000   | 3.83sec/eval | Learning Rate:[9.67108188151324e-05] -->[4.854433314505857e-05]
[Epoch:309 100/391]  Loss: 0.0144, Acc: 99.5746  |  Correct 12873/12928   | 0.13sec/iter  12.61sec/100iter 
[Epoch:309 200/391]  Loss: 0.0146, Acc: 99.5686  |  Correct 25617/25728   | 0.13sec/iter  12.56sec/100iter 
[Epoch:309 300/391]  Loss: 0.0156, Acc: 99.5120  |  Correct 38340/38528   | 0.13sec/iter  12.57sec/100iter  51.90sec/epoch | eval | Acc: 71.910  |  7191 / 10000   | 3.82sec/eval | Learning Rate:[4.854433314505857e-05] -->[1.9637012099169403e-05]

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
