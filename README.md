# SGANet

The implementation of "SGANet: Hyperspectral Image Super-Resolution via Spectral Group Attention Network"

## Environment

- Python 3.9.18
- PyTorch 2.1.1+cu121
- NumPy 1.25.2

## Datasets

- CAVE: [http://www1.cs.columbia.edu/CAVE/databases/multispectral/](http://www1.cs.columbia.edu/CAVE/databases/multispectral/)
- Harvard: [https://vision.seas.harvard.edu/hyperspec/download.html](https://vision.seas.harvard.edu/hyperspec/download.html)
- PaviaU: [http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)
- Urban: [http://www.erdc.usace.army.mil/Media/Fact-Sheets/Fact-Sheet-Article-View/Article/610433/hypercube/#](http://www.erdc.usace.army.mil/Media/Fact-Sheets/Fact-Sheet-Article-View/Article/610433/hypercube/#)

## Directory Structure

```
datasets  # Training and testing data
models    # Implementation of models
data      # Implementation of data classes
main.py   # Main entry point for the model
util      # Collection of utility functions
```

## Training Command

To train the model, use the following command:
```
python main --model_name SGANet --dataset Urban
```

## Other Parameters

Additional parameters that can be used:
- `--model_name`: Name of the model
- `--dataset`: Name of the dataset
- `--check_point`: Path to resume training from a saved model checkpoint
- `--lr`: Learning rate (default: 4e-4)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 1000)

