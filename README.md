# deep-learning-early-stopping

A minimal utility class for an early-stopping of your deep learning model training. No additional packages needed.

## Quickstart

```
from earlystopping import EarlyStopping

early_stopping = EarlyStopping(patience=5, delta=1e-2, less_is_better=True)

for epoch in range(num_epoch):
    train_loss = train(...)
    val_loss = val(...)

    stop = early_stopping.step()
    if stop:
        print(f'Early stopping! Best validation loss: {early_stopping.get_best_score()}')
        break
```


