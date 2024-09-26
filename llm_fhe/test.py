import time

import numpy as np
from concrete.fhe import compiler
from concrete import fhe

# The function that implements our model
@compiler({"x": "encrypted", "y": "encrypted"})
def linear_model(x, y):
    return x @ y

# A representative input-set is needed to compile the function (used for tracing)
n_bits_input = 6
N_DIM = 32
inputset = [
    (
        np.random.randint(2**n_bits_input, size=(N_DIM,N_DIM)),
        np.random.randint(2**n_bits_input, size=(N_DIM,N_DIM)),
    )
    for _ in range(100)
]
circuit = linear_model.compile(inputset)
print("Compilation OK")
# Test our FHE inference
input_0 = np.random.randint(2**n_bits_input, size=(N_DIM,N_DIM))
input_1 = np.random.randint(2**n_bits_input, size=(N_DIM,N_DIM))

circuit.keygen()
print("Key generated")
encrypted_x, encrypted_y = circuit.encrypt(input_0, input_1)
print("Start execution")
start = time.time()
for _ in range(1):    
    encrypted_result = circuit.run(encrypted_x, encrypted_y)
end = time.time()

execution_time = end - start
print(f"Execution time: {execution_time:.2f} seconds")

result = circuit.decrypt(encrypted_result)