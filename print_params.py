import pickle


N = 28 * 28
d = 2
B = 4

theoretical = N * d * (B**2)
print("Theoretical:", theoretical)

model = pickle.load(open(f"mnist_linear_1x8_1/models/B{B}_F{d}.sav", 'rb'))
nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Torch:      ", nParam)

print("Difference: ", nParam - theoretical)

