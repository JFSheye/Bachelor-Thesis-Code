import matplotlib.pyplot as plt

bond_dims = [i for i in range(4, 33, 4)]
feature_dims = [i for i in range(4, 33, 4)]

for feature_dim in feature_dims:
    avg_runtime = []
    for bond_dim in bond_dims:
        f = open(f'mnist_sinusoidal_8x8_1/results/B{bond_dim}_F{feature_dim}.txt')
        lines = f.readlines()

        epoch = lines[-7]
        epoch = [int(s) for s in epoch.split() if s.isdigit()][0]

        runtime = lines[-2]
        runtime = [int(s) for s in runtime.split() if s.isdigit()][0]

        avg_runtime.append(runtime/epoch)

    plt.plot(bond_dims, avg_runtime, "-", marker=".", label=f"F{feature_dim}")

plt.xlabel(r'$\beta$')
plt.ylabel("Time per epoch (sec)")
plt.xticks(bond_dims)
plt.legend()
plt.show()

