import numpy as np

data0 = np.fromfile("datasets/shapenet_airplane_l1only/airplane_001091_manifold_rootNum256_FPSInit1000_vecs.bin", dtype=np.float32)
data1 = np.fromfile("./test.bin", dtype=np.float32)

data0 = data0[3:]
data1 = data1[3:]

for i in range(256):
    data0_ = data0[i * 361: (i + 1) * 361]
    data1_ = data1[i * 361: (i + 1) * 361]
    j = 2
    if np.sum(np.abs(data0_[j:j + 7 ** 3] - data1_[j:j + 7 ** 3])) > 1e-5:
        print("Checking grid.")
        import pdb; pdb.set_trace()
    j += 7 ** 3
    if np.sum(np.abs(data0_[j:j + 6] - data1_[j:j + 6])) > 1e-5:
        print("Checking orientation.")
        import pdb; pdb.set_trace()
    j += 6
    if np.sum(np.abs(data0_[j:j + 3] - data1_[j:j + 3])) > 1e-5:
        print("Checking rhs.")
        import pdb; pdb.set_trace()
    j += 3
    if np.sum(np.abs(data0_[j:j + 1] - data1_[j:j + 1])) > 1e-5:
        print("Checking abs.")
        import pdb; pdb.set_trace()
    j += 1
    if np.sum(np.abs(data0_[j:j + 3] - data1_[j:j + 3])) > 1e-5:
        print("Checking rlp.")
        import pdb; pdb.set_trace()
    j += 3
    if np.sum(np.abs(data0_[j:j + 3] - data1_[j:j + 3])) > 1e-5:
        print("Checking abp.")
        import pdb; pdb.set_trace()

data0 = data0[256 * 361:]
data1 = data1[256 * 361:]

for i in range(2048):
    data0_ = data0[i * 139: (i + 1) * 139]
    data1_ = data1[i * 139: (i + 1) * 139]
    j = 2
    if np.sum(np.abs(data0_[j:j + 5 ** 3] - data1_[j:j + 5 ** 3])) > 1e-5:
        print("Checking grid.")
        import pdb; pdb.set_trace()
    j += 5 ** 3
    if np.sum(np.abs(data0_[j:j + 6] - data1_[j:j + 6])) > 1e-5:
        print("Checking orientation.")
        import pdb; pdb.set_trace()
    j += 6
    if np.sum(np.abs(data0_[j:j + 3] - data1_[j:j + 3])) > 1e-5:
        print("Checking rhs.")
        import pdb; pdb.set_trace()
    j += 3
    if np.sum(np.abs(data0_[j:j + 3] - data1_[j:j + 3])) > 1e-5:
        print("Checking rlp.")
        import pdb; pdb.set_trace()
