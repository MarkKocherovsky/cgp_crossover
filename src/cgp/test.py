import numpy as np

dtype = [('NodeType', 'U10'), ('Value', 'f8'), ('Operator', 'U10'),
         ('Operand0', 'i4'), ('Operand1', 'i4'), ('Active', 'i4')]

original = np.array([
    ('Input', 0.0, '', 0, 0, 0),
    ('Function', 0.0, 'add', 0, 0, 0),
], dtype=dtype)

def safe_copy_structured(array: np.ndarray) -> np.ndarray:
    copy = np.empty(array.shape, dtype=array.dtype)
    for field in array.dtype.names:
        copy[field] = np.array(array[field], copy=True)
    return copy

copy = safe_copy_structured(original)

print("np.shares_memory(copy, original):", np.shares_memory(copy, original))
for name in original.dtype.names:
    print(f"Field {name} shared? ", np.shares_memory(copy[name], original[name]))

