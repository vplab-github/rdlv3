import cv2
import numpy as np
import argparse



def process_image(image):
    # covert to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary image
    mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

    # # add a closing operation
    # # morphological closing operation
    # kernel = np.ones((2, 2), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # do a min cluster filter and remove noise
    # do connected components
    connectivity = 8
    cc = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    num_labels = cc[0]
    label_mat = cc[1]
    stats = cc[2]

    thresh = 20
    for i in range(1, num_labels):
        if stats[i,4] <= thresh:
            mask[label_mat==i] = 0

    return mask

def calculate_num_clusters(mask):
    labeled_image = np.zeros((mask.shape[0], mask.shape[1], 3)).astype(np.uint8)
    connectivity = 8
    cc = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    num_labels = cc[0]
    labels = cc[1]

    for i in range(1, num_labels):
        mask = labels == i
        labeled_image[mask] = np.random.randint(0, 255, size=3)
    
    # cv2.imwrite("labelled.png", labeled_image)
    return num_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_mask", "-r", type=str, default="test_images/15188-result.tif")
    parser.add_argument("--gt_mask", "-g", type=str, default="test_images/15188.tif")
    

    opt_parser = parser.parse_args()
    image1 = opt_parser.gt_mask
    image2 = opt_parser.result_mask

    image_gt = cv2.imread(image1)
    image_pred = cv2.imread(image2)

    # pre process image
    mask_gt = process_image(image_gt)
    mask_pred = process_image(image_pred)


    # calculate total number of clusters
    num_gt = calculate_num_clusters(mask_gt)
    num_pred = calculate_num_clusters(mask_pred) 

    print(f"Total count of gt cells :{num_gt}")
    print(f"Total count of pred cells :{num_pred}")

    # for every gt cluster, calculate presece or absense
    connectivity = 8
    cc = cv2.connectedComponentsWithStats(mask_gt, connectivity, cv2.CV_32S)
    num_labels = cc[0]
    label_mat = cc[1]
    stats = cc[2]
    match = 0
    for i in range(1, num_labels):
        current_area = stats[i,4]
        current_mask = (label_mat == i)

        mask_p = mask_pred.copy()
        mask_p[current_mask == 0] = 0

        pred_area = np.sum(mask_p > 0)

        fill_score = pred_area / current_area
        # print(f"fille_acore = {fill_score}")
        fill_thresh = 0.5
        if fill_score > fill_thresh:
            match = match + 1


    print(f"Accuracy = {match / num_gt}")
    print("")


    # Display the binary image
    cv2.imshow('mask_gt', mask_gt)
    cv2.imshow('mask_pred', mask_pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
