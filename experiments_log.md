# CNN Experiments

> Notes: `acc` = accuracy. `GAP = eval - train` (same metric you used throughout).

### Baseline experiment (Experiment 0)

- 50 epochs
- Adam, lr = 0.01, batch-size = 128

- features: 2 conv (3x3) + 2 batchnorm + 2 ReLU + 2 maxpool (2x2) (3→16→32)
- 2 hidden layers in the classifier (flat_size = 2048, last_neurons = 512)
- no dropout in the hidden layers

From loss ≈ 1.1 onwards, consistent overfitting; eval does not improve.
From loss ≈ 0.8 (epoch ~20), improvement becomes very small until the end (only ~0.13 less in 30 epochs).

**Loss goes from 2 → 0.67 and acc stays around 0.6–0.8 on both train and eval (2 + 0.3 sec/epoch).**

---

### Experiment 1 (dropout 0.1 in hidden layers)

- dropout 0.1 in each hidden layer

Overfitting is reduced (both loss and acc).
After epoch 20, loss improves less than in the baseline (~0.1 improvement in 30 epochs).
Also, after epoch 20 the **overfitting reduction** becomes more irregular.

**Loss goes from 2 → 0.95 and acc stays around 0.6–0.7 (2 + 0.3 sec/epoch).**

---

### Experiment 2 (dropout 0.1 + 3 hidden layers)

- dropout 0.1 in each hidden layer
- 3 hidden layers in the classifier (flat_size = 2048, last_neurons = 256)

Training gets worse: after epoch 10, loss drops from ~1 to ~0.87 and overfitting increases.

**Loss goes from 2 → 0.87 and acc stays around 0.6–0.7 (2 + 0.3 sec/epoch).**

---

### Experiment 3 (no dropout + 3 hidden layers)

- no dropout in hidden layers

Very similar to the previous one, no big changes in overfitting. I think dropout should be very small.

**Loss goes from 2 → 0.8 and acc stays around 0.6–0.7 (2 + 0.3 sec/epoch).**

---

### Experiment 4 (no dropout + 4 hidden layers)

- 4 hidden layers in the classifier (flat_size = 2048, last_neurons = 128)

Goal: check if the model can memorize more in 50 epochs. If yes, then fix overfitting; if not, change the feature extractor. Also consider weight decay.

Overfitting increases a lot compared to the previous experiment. After epoch 10, train keeps improving while eval starts degrading. After epoch 20, eval loss increases a lot instead of stabilizing.
Accuracy: train goes above ~0.8 but eval drops below ~0.7 as training continues (strong overfitting).

Between epochs 20 and 50, train loss drops ~0.14.

**Loss goes from 1.8 → 0.49 and acc reaches 0.85 on train but 0.68 on eval (2 + 0.3 sec/epoch).**

---

### Experiment 5 (no dropout + 5 hidden layers)

- 5 hidden layers in the classifier (flat_size = 2048, last_neurons = 64)

Probably unnecessary. It does not improve vs Experiment 4 and is slightly worse.

---

### Experiment 6 (lr = 0.001 + 4 hidden layers)

- 4 hidden layers in the classifier (flat_size = 2048, last_neurons = 128)
lr = 0.001

Huge difference: it overfits / “memorizes” in ~10 epochs. From here on, the focus is: improve learning and reduce overfitting.

Best eval loss: 0.8517 | Best train loss: 0.6096 | GAP: 0.2422 | Epoch: 4
Best eval acc : 0.7392 | Best train acc : 0.9060 | GAP: 0.1668 | Epoch: 8
Average epoch time: 2.6889

---

### Experiment 7 (dynamic dropout)

- dropout = 0.1*(nlayers-i) in each hidden layer

No significant improvement in eval acc and only ~0.1 improvement in eval loss.

Best eval loss: 0.7455 | Best train loss: 0.5348 | GAP: 0.2107 | Epoch: 11
Best eval acc : 0.7557 | Best train acc : 0.8571 | GAP: 0.1014 | Epoch: 15
Average epoch time: 2.5636

---

### Experiment 8 (more dynamic dropout)

- dropout = 0.15*(nlayers-i) in each hidden layer

Much worse than before. I will keep dynamic dropout = 0.1.

Best eval loss: 0.7492 | Best train loss: 0.6217 | GAP: 0.1275 | Epoch: 17
Best eval acc : 0.7529 | Best train acc : 0.8502 | GAP: 0.0973 | Epoch: 31
Average epoch time: 2.5137

---

### Experiment 9 (dynamic dropout + 2 transforms)

- dropout = 0.1*(nlayers-i) in each hidden layer

- transforms.RandomCrop(32, padding=4)
- transforms.RandomHorizontalFlip() (train dataset)

Eval improves a lot, and it stays better than train most of the time.
After epoch 20, improvement is small but still consistent.

Next: add 1–2 more transformations. If there is no clear gain, then apply weight decay or check other factors.

Best eval loss: 0.6461 | Best train loss: 0.7739 | GAP: -0.1278 | Epoch: 40
Best eval acc : 0.7842 | Best train acc : 0.7392 | GAP: -0.0450 | Epoch: 45
Average epoch time: 2.6433

---

### Experiment 10 (4 transforms)

- transforms.ColorJitter(brightness=0.3, hue=0.1, contrast=0.3, saturation=0.3)
- transforms.RandomRotation(degrees=(0,5))

No improvement after 50 epochs, only a big increase in execution time and CPU usage. Discard for now and proceed with weight decay.

Best eval loss: 0.6700 | Best train loss: 0.8709 | GAP: -0.2009 | Epoch: 45
Best eval acc : 0.7739 | Best train acc : 0.7025 | GAP: -0.0714 | Epoch: 45
Average epoch time: 4.2015 (I switched from 6→10 num_workers compared to the previous experiment; otherwise it would be even slower)

---

### Experiment 11 (WD = 1e-4 + AdamW)

- Opt: AdamW
- WD = 1e-4

Minimal improvement vs Experiment 9. Next: try higher WD.

Best eval loss: 0.6355 | Best train loss: 0.7496 | GAP: -0.1140 | Epoch: 47
Best eval acc : 0.7854 | Best train acc : 0.7438 | GAP: -0.0416 | Epoch: 47
Average epoch time: 2.7123

---

### Experiment 12 (WD = 1e-3)

- WD = 1e-3

No clear improvement…

Best eval loss: 0.6345 | Best train loss: 0.7529 | GAP: -0.1184 | Epoch: 47
Best eval acc : 0.7895 | Best train acc : 0.7429 | GAP: -0.0466 | Epoch: 47
Average epoch time: 2.6438

---

### Experiment 13 (WD = 3e-3)

- WD = 3e-3

Still no improvement…

Next: move to feature extractor changes, and keep WD = 1e-3.

Best eval loss: 0.6353 | Best train loss: 0.7594 | GAP: -0.1241 | Epoch: 47
Best eval acc : 0.7862 | Best train acc : 0.7422 | GAP: -0.0440 | Epoch: 48
Average epoch time: 2.7519

---

### Experiment 14 (WD = 1e-3 + new conv layer in features)

- features = (1 Conv2D + 1 Pooling + 1 Activation) × 3
- convs ⇒ (3→16→32→64)
- pooling ⇒ (16×16→8×8→4×4)

Clear improvement (~2% acc). Maybe 3 pooling layers is too aggressive; idea: use 2 convs before the last pooling.

Best eval loss: 0.5556 | Best train loss: 0.6141 | GAP: -0.0585 | Epoch: 47
Best eval acc : 0.8091 | Best train acc : 0.7941 | GAP: -0.0150 | Epoch: 47
Average epoch time: 2.5095

---

### Experiment 15 (remove 1 pooling)

- features = (1 Conv2D + 1 Pooling + 2 Conv2D + 1 Pooling + Activations after each conv)
- convs ⇒ (3→16→32→64)
- pooling ⇒ (16×16→8×8)

Clear improvement (~3% acc).

Best eval loss: 0.5098 | Best train loss: 0.5423 | GAP: -0.0325 | Epoch: 48
Best eval acc : 0.8306 | Best train acc : 0.8178 | GAP: -0.0128 | Epoch: 48
Average epoch time: 5.4977

---

### Experiment 16 (first scheduler: MultiStepLR)

- MultiStepLR(opt, milestones=[20, 35], gamma=0.1)

I set milestone 20 because improvement slows down a lot around that point, and 35 to keep fine-tuning.

Performance dropped, so I retried with a smaller gamma (smoother drop) and moved milestones later (possible early scheduling).

Best eval loss: 0.5242 | Best train loss: 0.5658 | GAP: -0.0416 | Epoch: 41
Best eval acc : 0.8221 | Best train acc : 0.8048 | GAP: -0.0173 | Epoch: 37
Average epoch time: 5.4845

---

### Experiment 17 (change MultiStepLR)

- MultiStepLR(opt, milestones=[30, 40], gamma=0.3)

Clear improvement in eval acc (~+1%). Best eval acc happens very late (epoch ~48/49 out of 50), so I increased training by +20 epochs to check if it keeps improving.

Best eval loss: 0.4748 | Best train loss: 0.4856 | GAP: -0.0108 | Epoch: 49
Best eval acc : 0.8403 | Best train acc : 0.8340 | GAP: -0.0063 | Epoch: 48
Average epoch time: 5.4731

---

### Experiment 18 (increase epochs +20)

- Epochs = 70

Minimal improvement (~0.5%). Not worth +20 epochs. Next: two final experiments, increasing feature **capacity** vs **depth**, to see what scales better toward >90% eval acc.

Best eval loss: 0.4661 | Best train loss: 0.4628 | GAP: 0.0033 | Epoch: 67
Best eval acc : 0.8450 | Best train acc : 0.8411 | GAP: -0.0039 | Epoch: 64
Average epoch time: 5.4871

---

### Experiment 19 (increase feature capacity)

- Last layer before the final pooling: 32→64 changed to 32→96.

Goal: check whether improving indefinitely would require more capacity or more depth. This experiment focuses on capacity.

No improvement; slightly worse (~0.2%), and almost 2× slower.

Best eval loss: 0.4668 | Best train loss: 0.4557 | GAP: 0.0110 | Epoch: 63
Best eval acc : 0.8434 | Best train acc : 0.8442 | GAP: 0.0008 | Epoch: 63
Average epoch time: 9.9719

---

### Experiment 20 (increase feature depth)

- Revert Experiment 19 (back to 32→64 before pooling).
- Add one extra layer after the last pooling: 64→92.

Same philosophy as before: test whether depth gives improvement.

Clear improvement (~+2%), although I made a mistake here: I forgot to add the final activation and BatchNorm.

Best eval loss: 0.4214 | Best train loss: 0.3272 | GAP: 0.0942 | Epoch: 63
Best eval acc : 0.8654 | Best train acc : 0.8875 | GAP: 0.0221 | Epoch: 68
Average epoch time: 9.6658

---

### Experiment 21 (BN + Activation on the last layer)

- Layer after the last pooling: 64→92 + BN + ReLU

Better eval loss, and a tiny drop in eval acc (0.04%).
It reaches best eval acc 9 epochs earlier than Experiment 20, although it reaches best eval loss 4 epochs later.

Best eval loss: 0.4111 | Best train loss: 0.3121 | GAP: 0.0989 | Epoch: 67
Best eval acc : 0.8650 | Best train acc : 0.8897 | GAP: 0.0247 | Epoch: 59
Average epoch time: 9.6990

**Conclusion: in my experiments, to keep improving the model I needed more depth. Increasing capacity (channels) or tweaking other parameters did not help, and sometimes made results worse.**


