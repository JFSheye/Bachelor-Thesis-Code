import torch
import torch.utils.data
import pickle
from torchvision import datasets, transforms
from utils import fit, test, create_loaders, create_samplers, create_num_batches


class CNN(torch.nn.Module):
    '''
        Convolutional Neural Network (CNN)
    '''

    def __init__(self, dims=2):
        super(CNN, self).__init__()

        # GPU usage
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        self.d = dims
        self.bond_dim = 0
        self.feature_dim = 0

        # two layered network
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=self.d // 2, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.d // 2, out_channels=self.d, kernel_size=(3, 3), padding=1),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=28 ** 2 * self.d, out_features=10)
        ).to(device)

    def forward(self, img):
        outputs = self.network(img)
        return outputs


# Random seed
torch.manual_seed(1)

# GPU usage
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Load and split data
# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])
# Data sets
# data_set = datasets.MNIST("./mnist", download=True, transform=transform)
# train_set, val_set = torch.utils.data.random_split(data_set, [50000, 10000])
# test_set = datasets.MNIST("./mnist", download=True, transform=transform, train=False)

data_set = datasets.FashionMNIST("./fashionmnist", download=True, transform=transform)
train_set, val_set = torch.utils.data.random_split(data_set, [50000, 10000])
test_set = datasets.FashionMNIST("./fashionmnist", download=True, transform=transform, train=False)

# Training parameters
num_train = len(train_set)
num_val = len(val_set)
batch_size = 100
l2_reg = 0.0
early_stopping = 10

# Samplers, loaders and number of batches
samplers = create_samplers(num_train, num_val)
loaders = create_loaders(train_set, val_set, batch_size, samplers)
num_batches = create_num_batches(num_train, num_val, batch_size)

myCNN = CNN()
fit(myCNN, 1e-3, 0.0, 10, loaders, 100, num_batches, device, False, "fashionmnist_cnn")

num_test = len(test_set)
samplers = {
    "test": torch.utils.data.SubsetRandomSampler(range(num_test))
}
loaders = {
    name: torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=samplers[name], drop_last=True, pin_memory=True
    )
    for (name, dataset) in [("test", test_set)]
}
num_batches = {
    name: total_num // batch_size
    for (name, total_num) in [("test", num_test)]
}

model = pickle.load(open(f"fashionmnist_cnn/models/B0_F0.sav", 'rb'))
test(model, loaders, batch_size, num_batches, device, False, "fashionmnist_cnn")
