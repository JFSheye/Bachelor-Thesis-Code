import matplotlib.pyplot as plt
from utils import extract_results

mnist_models = ['mnist_linear_1x8_1/results/B28_F2.txt',
                'mnist_sinusoidal_8x8_1/results/B32_F4.txt',
                'mnist_cnn_8x8_1/results/B24_F32.txt',
                'mnist_positionalEncoding_8x8_1/results/B32_F4.txt',
                ]
fashionmnist_models = ['fashionmnist_linear_1x8_1/results/B28_F2.txt',
                       'fashionmnist_sinusoidal_8x8_1/results/B24_F4.txt',
                       'fashionmnist_cnn_8x8/results/B28_F20.txt',
                       'fashionmnist_positionalEncoding_8x8_1/results/B12_F4.txt'
                       ]

names = ['linear', 'sinusoidal', 'learnable', 'positional encoding']

for model, name in zip(fashionmnist_models, names):
    val_loss = extract_results(model, f"validation loss")
    epochs = [i for i in range(1, len(val_loss) + 1)]
    plt.plot(epochs, val_loss, "-", marker=".", linewidth=1, label=name)

    print(max(epochs))

plt.title(f'Evolution of validation loss over epochs')
plt.xlabel('Epochs')
plt.ylabel(f'Validation loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_over_epochs_fashionmnist.png')
plt.show()


