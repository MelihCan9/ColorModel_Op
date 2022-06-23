import numpy as np
import cv2



def red_highlight(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Specify 2 lower and upper bound for red color in hsv color model. Bounds includes hue, saturation, value range.
    lbound1 = np.array([0, 195, 50])
    ubound1 = np.array([10, 255, 255])

    lbound2 = np.array([169, 195, 50])
    ubound2 = np.array([180, 255, 255])

    # Two masks for two seperate bounds.
    mask = cv2.inRange(hsv, lbound1, ubound1)
    mask2 = cv2.inRange(hsv, lbound2, ubound2)

    # Combining the masks.
    full_mask = mask + mask2

    # Specify 2 arrays,first gray array for 'for loop' and the second array is our whitelist which will specify via loop
    # but here is the important part of this 'for loop'. This loop creates a grayscale image with 3 channels.
    arr = np.array(gray)
    wl = img[:] * 0
    for i, val1 in enumerate(arr):
        for j, val2 in enumerate(val1):
            wl[i, j][:] = val2

    # Applying the mask for the result.
    detect2 = cv2.bitwise_not(img, wl, mask=full_mask)
    result = cv2.bitwise_not(wl, wl, mask=full_mask)

    cv2.imshow("RESULT", result)
    cv2.imshow("ORIGINAL", img)

    # Press Esc for close this methods outputs.
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()


def yiq_histeq(img):
    arr = np.array(img)

    # OpenCV read images in BGR color model in default. Here make BGR to RGB
    b, g, r = cv2.split(img)
    rgb = cv2.merge((r, g, b))

    # Transformation matrices
    mat = [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]]
    mat_2 = [[1.000, 0.956, 0.621], [1.000, -0.272, -0.647], [1.000, -1.106, 1.703]]

    # Creating an empty and same shape array which will specify in the loop via calculation process.
    yiq = np.array(np.zeros(img.shape))
    for i, val1 in enumerate(rgb):
        for j, val2 in enumerate(val1):
            rgb_t = np.array(val2).reshape(3, 1)
            calc = np.array(np.dot(mat, rgb_t))

            yiq[i, j][0] = calc[0]
            yiq[i, j][1] = calc[1]
            yiq[i, j][2] = calc[2]

    # Round the values of the array and conformate the  array types for equalization and one more transformation process
    y_1, i_1, q_1 = np.array(cv2.split(yiq))
    off_1 = np.round(y_1)
    int_array_1 = off_1.astype(int)
    int_array_1 = int_array_1.astype(np.uint8)

    # Equalization process in just Y component of the YIQ image.
    for a, v_1 in enumerate(int_array_1):
        for b, v_2 in enumerate(v_1):
            int_array_1[a, b] = ((int_array_1[a, b] ** 2) / 255)

    # Again edit the array type.
    int_array_1 = int_array_1.astype(np.uint8).astype(y_1.dtype)

    # Merging the other YIQ components with the new Y.
    yiq = cv2.merge((int_array_1, i_1, q_1))

    # Creating empty arrays for YIQ to RGB transformation.
    ytr = np.array(np.zeros(img.shape))
    r_1 = np.array(np.zeros((img.shape[0], img.shape[1])))
    g_1 = np.array(np.zeros((img.shape[0], img.shape[1])))
    b_1 = np.array(np.zeros((img.shape[0], img.shape[1])))

    # YIQ to RGB
    for a, k1 in enumerate(yiq):
        for b, k2 in enumerate(k1):
            yiq_t = np.array(k2).reshape(3, 1)

            mltp = np.array(np.dot(mat_2, yiq_t))

            r_1[a, b] = np.array(mltp[0])
            g_1[a, b] = np.array(mltp[1])
            b_1[a, b] = np.array(mltp[2])

    ytr = cv2.merge((b_1, g_1, r_1))
    off = np.round(ytr)

    # After several processes our image array may include values which are negative or bigger then 255. To handle that:
    for m, v1 in enumerate(off):
        for n, v2 in enumerate(v1):

            if v2[0] < 0:
                v2[0] = 0
            elif v2[0] > 255:
                v2[0] = 255

            if v2[1] < 0:
                v2[1] = 0
            elif v2[1] > 255:
                v2[1] = 255

            if v2[2] < 0:
                v2[2] = 0
            elif v2[2] > 255:
                v2[2] = 255

    # Final arrangement at array type.
    int_array = off.astype(arr.dtype)

    cv2.imshow("Original", arr)
    cv2.imshow("After YIQ", int_array)

    # Press Esc for close this methods outputs.
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    yiq_histeq(img)
    red_highlight(img)
