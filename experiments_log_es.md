# Experimentos CNN

### Experimento base

- 50 epochs

- Adam, lr = 0.01, batch-size = 128

- eatures: 2 conv (3x3) + 2 batchnorm + 2 ReLU + 2 maxpool (2x2) (3→16→32)

- 2 capas ocultas en classifier (flat_size = 2048 last_neurons =  512)

- sin dropout en cada capa oculta

a partir de 1.1 de loss, overfitting constante, el eval no mejora

a partir del 0.8 de loss (20 epoch) la mejora es muy pequeña hasta llegar al ultimo (30 epochs solo 0.13 menos)

 **baja de 2 a 0.67 de loss y la acc se mantiene entre 0.6-0.8 en train como en eval (2+0.3segs por epoch)**

### Experimento 1 (dropout 0.1 en capas ocultas)

- dropout 0.1 en cada capa oculta

Se reduce el overfitting en loss y en acc

A partir del epoch 20 hay menos mejora en el loss que en el base (0.1 de mejora en 30 epochs)

Tambien diria que a partir del epoch 20 la **mejora de overfitting** es más irregular

**baja de 2 a 0.95 de loss y la acc se mantiene entre 0.6-0.7 (2+0.3segs por epoch)**

### Experimento 2 (dropout 0.1 + 3 capas ocultas (1+))

- dropout 0.1 en cada capa oculta

- 3 capas ocultas en classifier (flat_size = 2048 last_neurons =  256)

Empeora el entrenamiento, a partir del epoch 10 el bajon es de 1 a 0.87 y el overfitting es mayor

**baja de 2 a 0.87 de loss y la acc se mantiene entre 0.6-0.7 (2+0.3segs por epoch)**

### Experimento 3 (sin dropout + 3 capas ocultas)

- sin dropout en cada capa oculta

Muy similar al anterior, sin cambios grandes notables en el overfitting, considero que el dropout debe ser muy bajo

**baja de 2 a 0.8 de loss y la acc se mantiene entre 0.6-0.7 (2+0.3segs por epoch)**

### Experimento 4 (sin dropout + 4 capas ocultas)

- 4 capas ocultas en classifier (flat_size = 2048 last_neurons =  128)

Probar si el modelo puede memorizar más en 50 epochs, si es el caso miramos a arreglar el overfitting o si no modificar la parte de features para ver si asi aprende mejor. Considerar tambien implementar Weight Decay

Overfitting crece, considerablemente a diferencia del anterior a partir del epoch 10 se sale del train totalmente y a partir del epoch 20 crece considerablemente el loss de val en vez de seguir subiendo. En accuracy se va pasando del 0.8 en train pero va bajando del 0.7 en eval cuanto más se entrena (overfitting fuerte)

Entre el epoch 20 y 50 se da un bajon de loss del 0.14. 

**baja de 1.8 a 0.49 de loss y la acc se llega a 0.85 en train pero 0.68 en eval  (2+0.3segs por epoch)**

### Experimento 5 (sin dropout + 5 capas ocultas)

- 5 capas ocultas en classifier (flat_size = 2048 last_neurons =  64)

Un poco innecesario, ya no mejora más respecto al 4 y tiene un leve empeoro.

### Experimento 6 (lr = 0.001 + 4 capas ocultas)

- 4 capas ocultas en classifier (flat_size = 2048 last_neurons =  128)

- lr = 0.001

La diferencia es abismal, memoriza en practicamente 10 epochs, a partir de aqui vamos a centrarnos en hacer que el modelo sea capaz de aprender y evitar overfitting.

Best eval loss: 0.8517 | Best train loss: 0.6096 | GAP: 0.2422 | Epoch: 4
Best eval acc : 0.7392 | Best train acc : 0.9060 | GAP: 0.1668 | Epoch: 8
Average epoch time: 2.6889

### Experimento 7 (dropout dinamico)

- dropout 0.1*(nlayers-i) en cada capa oculta

No parece haber una mejora significativa en eval acc y un 0.1 en eval loss.

Best eval loss: 0.7455 | Best train loss: 0.5348 | GAP: 0.2107 | Epoch: 11
Best eval acc : 0.7557 | Best train acc : 0.8571 | GAP: 0.1014 | Epoch: 15
Average epoch time: 2.5636

### Experimento 8 (más dropout dinamico)

- dropout 0.15*(nlayers-i) en cada capa oculta

Mucho peor resultado que antes, lo dejare con 0.1 dinamico

Best eval loss: 0.7492 | Best train loss: 0.6217 | GAP: 0.1275 | Epoch: 17
Best eval acc : 0.7529 | Best train acc : 0.8502 | GAP: 0.0973 | Epoch: 31
Average epoch time: 2.5137

### Experimento 9 (dropout dinamico + 2 Transforms)

- dropout 0.1*(nlayers-i) en cada capa oculta

- transforms.RandomCrop(32, padding=4),
- transforms.RandomHorizontalFlip() (En el dataset de train)

El eval ha mejorado considerablemente, y va todo el rato mejor que el train.

A partir del epoch 20 la mejora es muy pequeña pero si que es constante

Voy a proceder a añadir 1-2 transformaciones más, a ver si hay una mejora considerable, si no procuramos aplicar weight-decay o mirar otros factores.

Best eval loss: 0.6461 | Best train loss: 0.7739 | GAP: -0.1278 | Epoch: 40
Best eval acc : 0.7842 | Best train acc : 0.7392 | GAP: -0.0450 | Epoch: 45
Average epoch time: 2.6433

### Experimento 10 (4 Transforms)

- transforms.ColorJitter(brightness=0.3, hue=0.1, contrast=0.3,saturation=0.3),
- transforms.RandomRotation(degrees=(0,5)),

No hay una mejora en los 50 epochs, solo un incremento considerable en el tiempo de ejecucion y cpu. Las descartamos de momento para proceder con WD.

Best eval loss: 0.6700 | Best train loss: 0.8709 | GAP: -0.2009 | Epoch: 45
Best eval acc : 0.7739 | Best train acc : 0.7025 | GAP: -0.0714 | Epoch: 45
Average epoch time: 4.2015 (pase de 6→10 num_workers en comparacion del experimento anterior, si no seria el doble)

### Experimento 11 (WD = 1e-4 + AdamW)

- Opt: AdamW

- WD = 1e-4

Mejora minima respecto al experimento 9, voy a probar con algo más de WD.

Best eval loss: 0.6355 | Best train loss: 0.7496 | GAP: -0.1140 | Epoch: 47
Best eval acc : 0.7854 | Best train acc : 0.7438 | GAP: -0.0416 | Epoch: 47
Average epoch time: 2.7123

### Experimento 12 (WD = 1e-3)

- WD = 1e-3

No parece haber mejora……

Best eval loss: 0.6345 | Best train loss: 0.7529 | GAP: -0.1184 | Epoch: 47
Best eval acc : 0.7895 | Best train acc : 0.7429 | GAP: -0.0466 | Epoch: 47
Average epoch time: 2.6438

### Experimento 13 (WD = 3e-3)

- WD = 3e-3

Sigue sin haber mejora….

Toca pasar a classifiers y se deja el WD = 1e-3

Best eval loss: 0.6353 | Best train loss: 0.7594 | GAP: -0.1241 | Epoch: 47
Best eval acc : 0.7862 | Best train acc : 0.7422 | GAP: -0.0440 | Epoch: 48
Average epoch time: 2.7519

### Experimento 14 (WD = 1e-3 y nueva capa convolucional en features)

- features = (1 Conv2D + 1 Pooling + 1 Activations 3 TIMES)

- convs ⇒ ( 3→16→32→64)

- pooling ⇒ (16*16→8*8→4*4)

Mejora notable (2% en acc), capaz 3 capaz de pooling es demasiado agresivo, sugerencia de hacer 2 convs antes de la ultima capa de pooling. 

Best eval loss: 0.5556 | Best train loss: 0.6141 | GAP: -0.0585 | Epoch: 47
Best eval acc : 0.8091 | Best train acc : 0.7941 | GAP: -0.0150 | Epoch: 47
Average epoch time: 2.5095

### Experimento 15 (Quitamos 1 de pooling)

- features = (1 Conv2D + 1Pooling + 2Conv2D + 1Pooling + 3 Activations after each conv) 

- convs ⇒ (3→16→32→64)

- pooling ⇒ (16*16→8*8)

Mejora notable (3% en acc)

Best eval loss: 0.5098 | Best train loss: 0.5423 | GAP: -0.0325 | Epoch: 48
Best eval acc : 0.8306 | Best train acc : 0.8178 | GAP: -0.0128 | Epoch: 48
Average epoch time: 5.4977

### Experimento 16 (Probamos primer scheduler: MultiStepLR)

- MultiStepLR(opt, milestones=[20, 35], gamma=0.1)

Decidi ponerle la milestone en el 20 por que es donde el modelo sufre una desaceleración de mejora considerable y la 35 para que siga afinando.

Hubo un bajon, reintento entranamiento con menos gamma, para bajon más suave y atraso las milestones por posibilidad de scheduler temprano.

Best eval loss: 0.5242 | Best train loss: 0.5658 | GAP: -0.0416 | Epoch: 41
Best eval acc : 0.8221 | Best train acc : 0.8048 | GAP: -0.0173 | Epoch: 37
Average epoch time: 5.4845

### Experimento 17 (cambiamos MultiStepLR)

- MultiStepLR(opt, milestones=[30, 40], gamma=0.3)

Mejora considerable en eval acc (casi un 1%). Se queda en el epoch 49/48 de 50, asi que voy a a hacer un incremento de 20 epochs para observar hasta que punto sigue mejorando.

Best eval loss: 0.4748 | Best train loss: 0.4856 | GAP: -0.0108 | Epoch: 49
Best eval acc : 0.8403 | Best train acc : 0.8340 | GAP: -0.0063 | Epoch: 48
Average epoch time: 5.4731

### Experimento 18 (aumentamos epochs +20)

- Epochs = 70

Mejora minima (0,5%), no considero que valga mucho la pena subirle 20 epochs. Voy a dejarlo asi y hacer dos ultimos experimentos aumentando la capacidad/profundidad de features para comprobar lo que hay que escalar para alcanzar valores de +90% en val acc.

Best eval loss: 0.4661 | Best train loss: 0.4628 | GAP: 0.0033 | Epoch: 67
Best eval acc : 0.8450 | Best train acc : 0.8411 | GAP: -0.0039 | Epoch: 64
Average epoch time: 5.4871

### Experimento 19 (Aumentamos capacidad features)

- La ultima capa antes del ultimo pooling paso de 32→64 a 32→96. 

Esto con el objetivo de ver si para seguir mejorando el modelo indefinidamente necesitariamos más capacidad o más profundidad. Este primer experimento consiste en la capacidad:

No hay mejora, de hecho hay un leve empeoramiento (**≈**0,2%), y el tiempo es casi el doble que el anterior experimento.

Best eval loss: 0.4668 | Best train loss: 0.4557 | GAP: 0.0110 | Epoch: 63
Best eval acc : 0.8434 | Best train acc : 0.8442 | GAP: 0.0008 | Epoch: 63
Average epoch time: 9.9719

### Experimento 20 (Aumentamos profundidad features)

- Capa antes del pooling de 32→96 a 32→64 (revertimos experimento 19)

Añadimos una capa despues del ultimo pooling de 64→92

Siguiendo la filosofia del anterior, vamos a ver si hay una mejora al aumentar profundidad.

Podemos ver una mejora considerable del 2%, aunque en este experimento tuve un pequeño despiste y no puse ni capa de activacion al final ni BatchNorm.

Best eval loss: 0.4214 | Best train loss: 0.3272 | GAP: 0.0942 | Epoch: 63
Best eval acc : 0.8654 | Best train acc : 0.8875 | GAP: 0.0221 | Epoch: 68
Average epoch time: 9.6658

### Experimento 21 (BN + Activation en la ultima capa)

- Capa despues del ultimo pooling de 64→92 + BN + ReLU

Mejora en eval loss y un minimo empeoramiento en eval acc (0.04%).

Tarda 9 epochs menos en obtener el mejor evall acc que el anterior aunque tarde 4 epochs más en obtener el mejor eval loss que el anterior.

Best eval loss: 0.4111 | Best train loss: 0.3121 | GAP: 0.0989 | Epoch: 67
Best eval acc : 0.8650 | Best train acc : 0.8897 | GAP: 0.0247 | Epoch: 59
Average epoch time: 9.6990

**En conclusión, en mis experimentos, para que el modelo siguiese mejorar hay que añadir más profundidad ya que mejoras en la capacidad o en otros par**á**metros no parece haber mejora o incluso empeora.**