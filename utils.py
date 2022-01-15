import os
import re
import time
import torch
import torch.utils.data
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torchmps import MPS


def set_lr(N, d, b):
    """ Function for computing a proportional learning rate """
    num_params = N * d * np.power(b, 2)
    if num_params <= np.power(10, 4):
        return 1e-3
    elif num_params <= np.power(10, 5):
        return 1e-4
    else:
        return 1e-5


def create_samplers(num_train, num_val):
    """ Create a dictionary of samplers that randomly
        sample subsets (indices) of the train and val
        data splits without replacement """
    samplers = {
        "train": torch.utils.data.SubsetRandomSampler(range(num_train)),
        "val": torch.utils.data.SubsetRandomSampler(range(num_val))
    }
    return samplers


def create_loaders(train_set, val_set, batch_size, samplers):
    """ Create a dictionary of data loaders that load
        the randomly sampled subsets in batches """
    loaders = {
        name: torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=samplers[name], drop_last=True, pin_memory=True
        )
        for (name, dataset) in [("train", train_set), ("val", val_set)]
    }
    return loaders


def create_num_batches(num_train, num_val, batch_size):
    """ Create a dictionary of number of batches to use """
    num_batches = {
        name: total_num // batch_size
        for (name, total_num) in [("train", num_train), ("val", num_val)]
    }
    return num_batches


def fit(model, learn_rate, l2_reg, early_stopping, loaders, batch_size, num_batches, device, flatten, dir_to_save_to):
    """ Fit a model. Saves the model with lowest observed validation loss. Fitting is stopped if validation loss
        has not increased for 10 consecutive epochs. Runs for a maximum of 100 epochs """
    # Loss function and optimizer
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_reg)

    # These are the file names we use for our results and models
    results_file = f"{dir_to_save_to}/results/B{model.bond_dim}_F{model.feature_dim}.txt"
    model_file = f'{dir_to_save_to}/models/B{model.bond_dim}_F{model.feature_dim}.sav'

    # Removes any pre-existing files with the same name as the file we want to write to
    if os.path.exists(results_file):
        os.remove(results_file)

    # Open a document to save learning rate to
    results = open(results_file, 'a')
    results.write(f"learning_rate: {learn_rate}\n\n")
    results.close()

    # Start timer
    start_time = time.time()

    # Counter for when to invoke early stopping
    early_stopping_counter = 0

    # Training and validation
    for epoch_num in range(1, 101):
        running_train_loss = 0.0
        running_train_acc = 0.0

        # Training
        for inputs, labels in loaders["train"]:
            if flatten:
                inputs, labels = inputs.view([batch_size, 28 ** 2]), labels.data
            else:
                inputs, labels = inputs.view([batch_size, 1, 28, 28]), labels.data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Call the model to get logit scores and predictions
            scores = model(inputs)

            _, preds = torch.max(scores, 1)

            # Compute the loss and accuracy, add them to the running totals
            loss = loss_fun(scores, labels)
            with torch.no_grad():
                accuracy = torch.sum(preds == labels).item() / batch_size
                running_train_loss += loss
                running_train_acc += accuracy

            # Backpropagate and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_acc = 0.0

            for inputs, labels in loaders["val"]:
                if flatten:
                    inputs, labels = inputs.view([batch_size, 28 ** 2]), labels.data
                else:
                    inputs, labels = inputs.view([batch_size, 1, 28, 28]), labels.data

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Call the model to get logit scores and predictions
                scores = model(inputs)

                _, preds = torch.max(scores, 1)

                # Compute the loss and accuracy, add them to the running totals
                loss = loss_fun(scores, labels)

                running_val_loss += loss
                running_val_acc += torch.sum(preds == labels).item() / batch_size

        # Keeping track of early stopping and saving the model
        if epoch_num == 1 or running_val_loss < lowest_observed_val_loss:
            lowest_observed_val_loss = running_val_loss
            early_stopping_counter = 0

            # Save the best model to disc (overwrites earlier models)
            pickle.dump(model, open(model_file, 'wb'))
        else:
            early_stopping_counter += 1

        # These lines are to be written to a document of results
        lines = [f"### Epoch {epoch_num} ###\n",
                 f"Average train loss:          {running_train_loss / num_batches['train']:.4f}\n",
                 f"Average train accuracy:      {running_train_acc / num_batches['train']:.4f}\n",
                 f"Average validation loss:     {running_val_loss / num_batches['val']:.4f}\n",
                 f"Average validation accuracy: {running_val_acc / num_batches['val']:.4f}\n",
                 f"Runtime so far:              {int(time.time() - start_time)} sec\n\n"]

        # Open a document to save training results to
        results = open(results_file, 'a')
        results.writelines(lines)
        results.close()

        # Invoke early stopping
        if early_stopping_counter == early_stopping:
            break


def fit_models(data_set, feature_map, fd_min, fd_max, bd_min, bd_max,
               fd_step_size, bd_step_size, flatten_data, dir_to_save_to):
    """ Set up training parameters and fit one or more models to the data. The user chooses which local feature map to
        apply. Results and models are saved to a user-specified directory. """
    # Create directory for results if needed
    if not os.path.isdir(dir_to_save_to):
        os.makedirs(dir_to_save_to)
        os.makedirs(dir_to_save_to + "/models")
        os.makedirs(dir_to_save_to + "/results")

    # Random seed
    torch.manual_seed(1)

    # GPU usage
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load and split data
    train_set, val_set = torch.utils.data.random_split(data_set, [50000, 10000])

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

    # Define grid
    feature_dims = [i for i in range(fd_min, fd_max+1, fd_step_size)]
    bond_dims = [i for i in range(bd_min, bd_max+1, bd_step_size)]

    for d in feature_dims:
        for b in bond_dims:
            mps = MPS(
                input_dim=28**2,
                output_dim=10,
                feature_dim=d,
                bond_dim=b,
                adaptive_mode=False,
                periodic_bc=False,
            )

            # if the feature map is None, then torchmps defaults to a linear feature map with d=2
            if feature_map is not None:
                mps.register_feature_map(feature_map(dims=d).execute)

            mps = mps.to(device)

            fit(model=mps, learn_rate=set_lr(28 ** 2, d, b), l2_reg=l2_reg, early_stopping=early_stopping,
                loaders=loaders, batch_size=batch_size, num_batches=num_batches, device=device, flatten=flatten_data,
                dir_to_save_to=dir_to_save_to)
    print("Finished running")


def test(model, loaders, batch_size, num_batches, device, flatten, dir_with_models_and_results):
    """Test a single pre-trained model against some test set and save the results to a directory"""
    b = model.bond_dim
    d = model.feature_dim

    running_test_acc = 0.0
    for inputs, labels in loaders["test"]:
        if flatten:
            inputs, labels = inputs.view([batch_size, 28 ** 2]), labels.data
        else:
            inputs, labels = inputs.view([batch_size, 1, 28, 28]), labels.data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Call our MPS to get logit scores and predictions
        scores = model(inputs)
        scores = scores.to(device)

        _, preds = torch.max(scores, 1)
        preds = preds.to(device)

        accuracy = torch.sum(preds == labels).item() / batch_size
        running_test_acc += accuracy

        # save test results to a file
    results = open(f"{dir_with_models_and_results}/test_results/B{b}_F{d}.txt", 'w')
    results.write(f"Test accuracy: {running_test_acc / num_batches['test']:.4f}")
    results.close()


def test_models(test_set, fd_min, fd_max, bd_min, bd_max, fd_step_size,
                bd_step_size, flatten_data, dir_with_models_and_results):
    """Test multiple pre-trained models against some test set and save the results to a directory"""
    os.makedirs(dir_with_models_and_results + "/test_results")

    # Random seed
    torch.manual_seed(1)

    # GPU usage
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Test parameters
    num_test = len(test_set)
    batch_size = 100
    l2_reg = 0.0
    early_stopping = 10

    # Samplers, loaders and number of batches
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

    # Define grid
    feature_dims = [i for i in range(fd_min, fd_max + 1, fd_step_size)]
    bond_dims = [i for i in range(bd_min, bd_max + 1, bd_step_size)]

    for d in feature_dims:
        for b in bond_dims:
            model = pickle.load(open(f"{dir_with_models_and_results}/models/B{b}_F{d}.sav", 'rb'))
            test(model=model, loaders=loaders, batch_size=batch_size, num_batches=num_batches,
                 device=device, flatten=flatten_data, dir_with_models_and_results=dir_with_models_and_results)
    print("Finished running")


def create_heatmap(dir_with_models_and_results, title, fd_min, fd_max, bd_min, bd_max, fd_step_size, bd_step_size):
    """Creates a heatmap of test results"""
    # Define the grid of feature dims and bond dims
    feature_dims = [i for i in range(fd_min, fd_max+1, fd_step_size)]
    bond_dims = [i for i in range(bd_min, bd_max+1, bd_step_size)]

    # Read all test results into a list of lists of values
    columns = []
    for d in feature_dims:
        row = []
        for b in bond_dims:
            f_name = f"{dir_with_models_and_results}/test_results/B{b}_F{d}.txt"
            f = open(f_name)
            test_acc = float(f.readline()[-6:])
            f.close()
            row.append(test_acc)
        columns.append(row)

    # Turn list of lists of values into a grid
    columns = np.array(columns)

    # Define figure and axes, arrange and add ticks, and set up axis-labels
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(bond_dims)))
    ax.set_yticks(np.arange(len(feature_dims)))
    ax.set_xticklabels(bond_dims)
    ax.set_yticklabels(feature_dims)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$d$', rotation='horizontal', labelpad=10)

    # Prettify the heatmap with colored and size-reduced text
    im = ax.imshow(columns)
    threshold = im.norm(columns.max()) / 2.
    for i in range(len(feature_dims)):
        for j in range(len(bond_dims)):
            text_color = ("white", "black")[int(im.norm(columns[i, j]) > threshold)]
            ax.text(j, i, "{:.4f}".format(columns[i, j]),
                    ha='center', va='center', color=text_color, fontsize='xx-small')

    # Set up a color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Test accuracy", rotation=-90, va="bottom")

    # Have the (bond_dim = min_val, feature_dim = min_val) be at the bottom left corner
    ax.invert_yaxis()

    # Title and tight layout
    plt.title(title)
    fig.tight_layout()
    plt.savefig(f'{dir_with_models_and_results}/{dir_with_models_and_results}.png')


def running_time(folders):
    """Compute the approximate running time of all models in a folder (e.g. mnist_cnn_8x8)"""
    for folder in folders:
        results = folder + '/results'
        total_secs = 0
        for file in os.listdir(results):
            f = open(results + "/" + file, 'r')
            last_line = f.readlines()[-2]
            total_secs += [int(s) for s in last_line.split() if s.isdigit()][0]
        print(folder, "ran for", total_secs // 60 // 60, "hours")
    print("\n")


def extract_results(results_dir, which_results):
    f = open(results_dir, 'r')
    lines = f.readlines()
    f.close()

    _string = f"{which_results}" + r":\s*([0-9].[0-9]+)"
    pattern = re.compile(_string)

    results = []
    for line in lines:
        match = pattern.search(line)
        if match:
            results.append(float(match.group(1)))

    return results


def create_plot(results_dir, which, fd_min, fd_max, bd_min, bd_max, fd_step_size, bd_step_size, vline=False):
    feature_dims = [i for i in range(fd_min, fd_max + 1, fd_step_size)]
    bond_dims = [i for i in range(bd_min, bd_max + 1, bd_step_size)]

    for d in feature_dims:
        for b in bond_dims:
            _dir = f"{results_dir}/B{b}_F{d}.txt"
            val_loss = extract_results(_dir, f"validation {which}")#[:-10]
            epochs = [i for i in range(1, len(val_loss) + 1)]
            plt.plot(epochs, val_loss, "-", marker=".", linewidth=1, label=f"B{b} F{d}")
            if vline:
                plt.axvline(max(epochs)-10, color='r', linestyle='--', label="Cut-off", zorder=0)
                plt.xticks(epochs[0::3])

    plt.title(f'Evolution of validation {which} over epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'Validation {which}')

    #plt.legend(bbox_to_anchor=(1.1, 1))
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_model(model, X_train, y_train, X_val, y_val):
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).type(torch.LongTensor)
    X_val = torch.tensor(X_val).float()
    y_val = torch.tensor(y_val).type(torch.LongTensor)

    for epoch_num in range(1, 101):
        # Call the model to get logit scores and predictions
        scores = model(X_train)
        _, preds = torch.max(scores, 1)

        # Compute the loss and accuracy, add them to the running totals
        loss = loss_fun(scores, y_train)

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Call the model to get logit scores and predictions
            scores = model(X_val)
            _, preds = torch.max(scores, 1)

            # Compute the validation loss
            val_loss = loss_fun(scores, y_val)

        # Keeping track of early stopping and saving the model
        if epoch_num == 1 or val_loss < lowest_observed_val_loss:
            lowest_observed_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Invoke early stopping
        if early_stopping_counter == 10:
            break

    return model


def score(model, X_test, y_test=None):
    X_test = torch.tensor(X_test).float()

    # Call our MPS to get logit scores and predictions
    scores = model(X_test)
    _, preds = torch.max(scores, 1)

    if y_test is not None:
        y_test = torch.tensor(y_test).type(torch.LongTensor)
        accuracy = torch.sum(preds == y_test).item() / len(X_test)
        return accuracy

    else:
        return preds
