import numpy as np

def tree_to_img_mnist(data0: np.ndarray, data1: np.ndarray):
    """
        Input:
            (parIdx, curIdx, value)
            data0: [16, 3], the first level node information
            data1: [64, 3], the second level node information
        Output:
            img0: First level image
            img1: Second level image
    """
    img0 = np.zeros((4, 4))
    img1 = np.zeros((8, 8))

    # import pdb; pdb.set_trace()
    for j in range(16):
        if data0[j * 3 + 2] > 0:
            curIdx = int(data0[j * 3 + 1] * 16)
            x = curIdx % 4
            y = curIdx // 4
            img0[y, x] = data0[j * 3 + 2]
    
    for j in range(64):
        if data1[j * 3 + 2] > 0:
            parIdx = int(data0[int(data1[j * 3] * 16) * 3 + 1] * 16)
            curIdx = int(data1[j * 3 + 1] * 4)
            x_j = parIdx % 4
            y_j = parIdx // 4
            x_k = curIdx % 2
            y_k = curIdx // 2
            x = x_j * 2 + x_k
            y = y_j * 2 + y_k
            img1[y, x] = data1[j * 3 + 2]
    
    img0 = (img0 * 255.0).astype(np.uint8)
    img1 = (img1 * 255.0).astype(np.uint8)

    return (img0, img1)


if __name__ == "__main__":
    import cv2

    IMG_NUM = 60000
    flatten_data = np.fromfile("../datasets/data.bin", dtype=np.float32)
    batched_data = flatten_data.reshape(IMG_NUM, 240)
    for i in range(10):
        data = batched_data[i]
        data0 = data[:48]
        data1 = data[48:]
        img0, img1 = tree_to_img_mnist(data0, data1) 
        cv2.imwrite(f"../test_imgs/img{i}_0_p.png", img0)
        cv2.imwrite(f"../test_imgs/img{i}_1_p.png", img1)

