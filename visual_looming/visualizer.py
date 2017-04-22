import os
import cv2
import numpy as np


def draw_output(all_matches, obs_matches, kp1, output):
    """ Writes output image to file.

    Params:
        all_matches: All matches found from Orb matcher.
        obs_matches: Matches after matches have been filtered.
        kp1: Keypoints from image 1 (current image).
        output: The keypoints will be drawn on output.
    """
    for m in all_matches:
        c, r = kp1[m.queryIdx].pt
        cv2.circle(output, (int(c), int(r)), int(
            kp1[m.queryIdx].size), (255, 255, 255), thickness=1)

    for m in obs_matches:
        c, r = kp1[m.queryIdx].pt
        cv2.circle(output, (int(c), int(r)), int(
            kp1[m.queryIdx].size), (0, 0, 255), thickness=3)

    cv2.imwrite("./output/output.png", output)


def show_kp(image1, kp1, image2, kp2, matches, tname, output_folder):
    """ Creates grid to show top 10 keypoints between two images.

    Params:
        image1: First image.
        kp1: Keypoints associated with image1.
        image2: Second image.
        kp2: Keypoints associated with image2.
        matches: All the matches between image1 and image2.
        tname: Name for the table (grid).
        output_folder: Folder name to output grid.
    """
    # create grid, to place 20 images, 10x2
    # each image is a square, dxd, dimensions of grid is d*10 x d*2

    d = 300

    y1bnd, x1bnd, _ = image1.shape
    y2bnd, x2bnd, _ = image2.shape

    grid = np.zeros((d * 10, d * 2, 3))

    # take top 10
    matches = matches[:10]

    for i, m in enumerate(matches):
        img1 = image1.copy()
        img2 = image2.copy()
        tmp1 = np.zeros((d, d))
        tmp2 = np.zeros((d, d))

        x1, y1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        x2, y2 = (int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1]))

        cv2.circle(img1, (x1, y1), 5, (0, 0, 255), thickness=-1)
        cv2.circle(img2, (x2, y2), 5, (0, 0, 255), thickness=-1)

        for channel in xrange(3):
            # calculate offsets for image 1
            t1 = abs(np.min([y1 - d / 2, 0]))
            b1 = np.max([y1 + d / 2 - y1bnd, 0])
            l1 = abs(np.min([x1 - d / 2, 0]))
            r1 = np.max([x1 + d / 2 - x1bnd, 0])

            # calculate offsets for image 2
            t2 = abs(np.min([y2 - d / 2, 0]))
            b2 = np.max([y2 + d / 2 - y2bnd, 0])
            l2 = abs(np.min([x2 - d / 2, 0]))
            r2 = np.max([x2 + d / 2 - x2bnd, 0])

            tmp1[t1:d - b1, l1:d - r1] = img1[np.max(
                [y1 - d / 2, 0]):np.min([y1 + d / 2, y1bnd]),
                np.max([x1 - d / 2, 0]):np.min([x1 + d / 2, x1bnd]),
                                         channel]

            tmp2[t2:d - b2, l2:d - r2] = img2[np.max(
                [y2 - d / 2, 0]):np.min([y2 + d / 2, y2bnd]),
                np.max([x2 - d / 2, 0]):np.min([x2 + d / 2, x2bnd]),
                                         channel]

            # add to main grid to return
            grid[(d * i):(d * i + d), :d, channel] = tmp1
            grid[(d * i):(d * i + d), d:, channel] = tmp2

            cv2.imwrite(
                os.path.join(output_folder, 'grid_{}.jpg'.format(tname)), grid)


def draw_matches(image_1, image_1_keypoints, image_2, image_2_keypoints, matches):
    """ Draws the matches between the image_1 and image_2.
    (Credit: GT CP2017 course provided source)

    Params:
        image_1: The first image (can be color or grayscale).
        image_1_keypoints: The image_1 keypoints.
        image_2: The image to search in (can be color or grayscale)
        image_2_keypoints: The image_2 keypoints.

    Returns:
        output: Image with a line drawn between matched keypoints.
    """
    # Compute number of channels.
    num_channels = 1
    if len(image_1.shape) == 3:
        num_channels = image_1.shape[2]
    # Separation between images.
    margin = 10
    # Create an array that will fit both images (with a margin of 10 to
    # separate the two images)
    joined_image = np.zeros((max(image_1.shape[0], image_2.shape[0]),
                             image_1.shape[1] + image_2.shape[1] + margin,
                             3))
    if num_channels == 1:
        for channel_idx in range(3):
            joined_image[:image_1.shape[0],
                         :image_1.shape[1],
                         channel_idx] = image_1
            joined_image[:image_2.shape[0],
                         image_1.shape[1] + margin:,
                         channel_idx] = image_2
    else:
        joined_image[:image_1.shape[0], :image_1.shape[1]] = image_1
        joined_image[:image_2.shape[0], image_1.shape[1] + margin:] = image_2

    for match in matches:
        image_1_point = (int(image_1_keypoints[match.queryIdx].pt[0]),
                         int(image_1_keypoints[match.queryIdx].pt[1]))
        image_2_point = (int(image_2_keypoints[match.trainIdx].pt[0] +
                             image_1.shape[1] + margin),
                         int(image_2_keypoints[match.trainIdx].pt[1]))

        rgb = (np.random.rand(3) * 255).astype(np.int)
        cv2.circle(joined_image, image_1_point, 5, rgb, thickness=-1)
        cv2.circle(joined_image, image_2_point, 5, rgb, thickness=-1)
        cv2.line(joined_image, image_1_point, image_2_point, rgb, thickness=3)

    return joined_image
