import os
import re
import numpy as np
import matplotlib.pyplot as plt


def extract_results(results_dir, which_results):
    """Extract epoch numbers"""
    f = open(results_dir, 'r')
    lines = f.readlines()
    f.close()

    _string = f"{which_results}" + r"\s*([0-9]+)"
    pattern = re.compile(_string)

    results = []
    for line in lines:
        match = pattern.search(line)
        if match:
            results.append(float(match.group(1)))

    return results


def running_time(folder):
    """Create list of tuples: (running time, [beta, d]) for all models in a folder"""
    results = folder + '/results'
    seconds = []
    epochs = []
    for file in os.listdir(results):
        f = open(results + "/" + file, 'r')
        last_line = f.readlines()[-2]

        epochs.append(int(extract_results(results + "/" + file, "Epoch")[-1]))

        tup = ['', '']
        for i, sub_string in enumerate(file.split('_')):
            dim = ''
            for s in sub_string:
                if s.isdigit():
                    dim += s
            tup[i] = dim

        seconds.append(([int(s) for s in last_line.split() if s.isdigit()][0], tup))
    return seconds, epochs


def find_specific(a_list, B, d):
    """Finds the running time for a specific model given beta and d"""
    for tup in a_list:
        if (tup[1][0] == B) & (tup[1][1] == d):
            minutes = tup[0] // 60
            seconds = tup[0] % 60
            print(str(tup[0]) + " seconds is equal to " + str(minutes) + " minutes and " + str(seconds) + " seconds")
            assert 60 * minutes + seconds == tup[0]


#a, e = running_time('mnist_linear_1x8_1')
#fa, fe = running_time('fashionmnist_linear_1x8_1')

#a, e = running_time('mnist_sinusoidal_8x8_1')
#fa, fe = running_time('fashionmnist_sinusoidal_8x8_1')

#a, e = running_time('mnist_cnn_8x8_1')
#fa, fe = running_time('fashionmnist_cnn_8x8')

a, e = running_time('mnist_positionalEncoding_8x8_1')
fa, fe = running_time('fashionmnist_positionalEncoding_8x8_1')

print("MNIST, average running time:", int(np.average(np.array(a, dtype=object)[:, 0])) // 60, "minutes")
print("FashionMNIST, average running time:", int(np.average(np.array(fa, dtype=object)[:, 0])) // 60, "minutes")

find_specific(fa, B='12', d='4')
find_specific(fa, B='8', d='4')


def group_by_time(times):
    """groups running time by intervals"""
    b = np.zeros(7, dtype=int)
    for (time, _) in times:
        if time <= 60 * 10:  # 10 minutes or less
            b[0] += 1
        elif time <= 60 * 20:  # 20 minutes or less
            b[1] += 1
        elif time <= 60 * 30:  # 30 minutes or less
            b[2] += 1
        elif time <= 60 * 40:  # 40 minutes or less
            b[3] += 1
        elif time <= 60 * 50:  # 50 minutes or less
            b[4] += 1
        elif time <= 60 * 60:  # 60 minutes or less
            b[5] += 1
        else:  # More than 60 minutes
            b[6] += 1
    return b


step_size = 4

labels = [r'$\leq$ 10', r'$\leq$ 20', r'$\leq$ 30', r'$\leq$ 40', r'$\leq$ 50', r'$\leq$ 60', r'> 60']
indices = range(len(labels))
width = np.min(np.diff(indices))/2.8

plt.bar(indices-width/2, group_by_time(a), width=0.35, label='MNIST')
plt.bar(indices+width/2, group_by_time(fa), width=0.35, label='FashionMNIST')
plt.xticks(ticks=range(len(labels)), labels=labels)
plt.yticks(np.arange(0, len(a)+1, step_size))
plt.xlabel('minutes')
plt.ylabel('count')
plt.title('distribution of running times')
plt.legend()
plt.show()


def group_by_epoch(epochs):
    """group epochs by intervals"""
    c = np.zeros(6, dtype=int)
    for epoch in epochs:
        if epoch <= 20:
            c[0] += 1
        elif epoch <= 40:
            c[1] += 1
        elif epoch <= 60:
            c[2] += 1
        elif epoch <= 80:
            c[3] += 1
        elif epoch <= 99:
            c[4] += 1
        else:
            c[5] += 1
    return c


labels = [r'$\leq$ 20', r'$\leq$ 40', r'$\leq$ 60', r'$\leq$ 80', r'$\leq$ 99', '100']
indices = range(len(labels))
width = np.min(np.diff(indices))/2.8

plt.bar(indices-width/2, group_by_epoch(e), width=0.35, label='MNIST')
plt.bar(indices+width/2, group_by_epoch(fe), width=0.35, label='FashionMNIST')
plt.xticks(ticks=range(len(labels)), labels=labels)
plt.yticks(np.arange(0, len(a)+1, step_size))
plt.xlabel('epochs')
plt.ylabel('count')
plt.title('distribution of epochs')
plt.legend()
plt.tight_layout()
plt.show()





