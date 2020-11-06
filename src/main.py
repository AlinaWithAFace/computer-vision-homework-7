import cv2 as cv
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim


def multi_resolution(img):
    """
    Generate and show four levels of multi-resolution. Use a Gaussian kernel of your choice.
    :param img:
    :return:
    """
    dst_0 = cv.resize(img, (0, 0), fx=.5, fy=.5)
    dst_1 = cv.resize(dst_0, (0, 0), fx=.5, fy=.5)
    dst_2 = cv.resize(dst_1, (0, 0), fx=.5, fy=.5)
    dst_3 = cv.resize(dst_2, (0, 0), fx=.5, fy=.5)
    return dst_0, dst_1, dst_2, dst_3


def multi_scale(img):
    """
    Generate and show four levels of multi-scale. Use the same Gaussian kernel as above.
    :param img:
    :return:
    """
    width, height, channel = img.shape
    temp0, temp1, temp2, temp3 = multi_resolution(img)
    dst_0 = cv.resize(temp0, (height, width), fx=1, fy=1)
    dst_1 = cv.resize(temp1, (height, width), fx=1, fy=1)
    dst_2 = cv.resize(temp2, (height, width), fx=1, fy=1)
    dst_3 = cv.resize(temp3, (height, width), fx=1, fy=1)
    return dst_0, dst_1, dst_2, dst_3


def laplacian_planes(img):
    """
    Generate Laplacian planes using a Laplacian kernel of your choice (can use Laplacian of Gaussian, or Laplacian).
    :param img:
    :return:
    """
    depth = cv.CV_16S
    dst = cv.Laplacian(img, depth)
    return dst


def approximation_multi_res(img):
    """
    Generate approximation to Laplacian using the difference of Gaussian planes from multi_resolution.
    Note, you need to do 'Expand' on images before taking the difference.
    :param img:
    :return:
    """
    width, height, channel = img.shape

    a, b, c, d = multi_resolution(img)
    a = cv.resize(a, (height, width), fx=1, fy=1)
    b = cv.resize(b, (height, width), fx=1, fy=1)
    c = cv.resize(c, (height, width), fx=1, fy=1)
    d = cv.resize(d, (height, width), fx=1, fy=1)

    diff_a = cv.subtract(img, a)
    diff_b = cv.subtract(img, b)
    diff_c = cv.subtract(img, c)
    diff_d = cv.subtract(img, d)

    temp_a = cv.add(diff_a, diff_b)
    temp_b = cv.add(diff_c, diff_d)
    dst = cv.add(temp_a, temp_b)

    return dst


def approximation_multi_scale(img):
    """
    Generate approximation to Laplacian using the difference of Gaussian planes from multi_scale
    :param img:
    :return:
    """

    a, b, c, d = multi_scale(img)

    diff_a = cv.subtract(img, a)
    diff_b = cv.subtract(img, b)
    diff_c = cv.subtract(img, c)
    diff_d = cv.subtract(img, d)

    temp_a = cv.add(diff_a, diff_b)
    temp_b = cv.add(diff_c, diff_d)
    dst = cv.add(temp_a, temp_b)

    return dst


def main():
    print("starting...")
    img = cv.imread("flower.jpg")
    a, b, c, d = multi_resolution(img)
    e, f, g, h = multi_scale(img)
    j = laplacian_planes(img)
    k = approximation_multi_res(img)
    l = approximation_multi_scale(img)
    results = [a, b, c, d, e, f, g, h, j, k, l]

    i = 0

    for image in results:
        cv.imwrite(str(i) + ".jpg", image)
        i += 1


main()
