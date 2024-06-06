import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Function to apply affine transformation to a triangle
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


# Function to get the Delaunay triangulation of a set of points
def get_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for point in points:
        subdiv.insert((float(point[0]), float(point[1])))
    triangles = subdiv.getTriangleList()
    delaunay_triangles = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if (rect[0] <= pts[0][0] <= rect[2] and rect[1] <= pts[0][1] <= rect[3] and
                rect[0] <= pts[1][0] <= rect[2] and rect[1] <= pts[1][1] <= rect[3] and
                rect[0] <= pts[2][0] <= rect[2] and rect[1] <= pts[2][1] <= rect[3]):
            ind = []
            for i in range(3):
                for j in range(len(points)):
                    if abs(pts[i][0] - points[j][0]) < 1 and abs(pts[i][1] - points[j][1]) < 1:
                        ind.append(j)
            if len(ind) == 3:
                delaunay_triangles.append((ind[0], ind[1], ind[2]))
    return delaunay_triangles


# Load the image you want to swap with
swap_img = cv2.imread('putin.png')
swap_img_gray = cv2.cvtColor(swap_img, cv2.COLOR_BGR2GRAY)

# Detect face and landmarks in the swap image
rects = detector(swap_img_gray, 1)
if len(rects) > 0:
    swap_shape = predictor(swap_img_gray, rects[0])
    swap_points = face_utils.shape_to_np(swap_shape)

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        # Only swap if the face is large enough
        if rect.width() < 100 or rect.height() < 100:
            continue

        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        points = face_utils.shape_to_np(shape)

        # Create a convex hull for the face
        hull1 = cv2.convexHull(np.array(swap_points))
        hull2 = cv2.convexHull(np.array(points))

        # Find the bounding rectangle of the convex hull
        rect = cv2.boundingRect(hull2)

        # Find the Delaunay triangulation of the face
        delaunay_triangles = get_delaunay_triangles(rect, points)

        # Warp and blend each triangle
        for triangle in delaunay_triangles:
            x1, y1, z1 = triangle
            t1 = [swap_points[x1], swap_points[y1], swap_points[z1]]
            t2 = [points[x1], points[y1], points[z1]]

            # Find the bounding rectangle of each triangle
            rect1 = cv2.boundingRect(np.array(t1))
            rect2 = cv2.boundingRect(np.array(t2))

            # Ensure the rectangles are within image bounds
            rect1 = (max(rect1[0], 0), max(rect1[1], 0),
                     min(rect1[2], swap_img.shape[1] - rect1[0]),
                     min(rect1[3], swap_img.shape[0] - rect1[1]))
            rect2 = (max(rect2[0], 0), max(rect2[1], 0),
                     min(rect2[2], frame.shape[1] - rect2[0]),
                     min(rect2[3], frame.shape[0] - rect2[1]))

            # Offset points by left top corner of the respective rectangles
            t1_offset = [(t1[i][0] - rect1[0], t1[i][1] - rect1[1]) for i in range(3)]
            t2_offset = [(t2[i][0] - rect2[0], t2[i][1] - rect2[1]) for i in range(3)]

            # Get the mask for each triangle
            mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0), 16, 0)

            # Apply affine transformation
            img1_rect = swap_img[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
            img2_rect = apply_affine_transform(img1_rect, t1_offset, t2_offset, (rect2[2], rect2[3]))

            # Copy triangular region of the rectangular patch to the output image
            img2_rect = img2_rect * mask
            frame[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]] = frame[rect2[1]:rect2[1] + rect2[3],
                                                                                rect2[0]:rect2[0] + rect2[2]] * (
                                                                                            1 - mask) + img2_rect

        # Seamless clone
        center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
        output = cv2.seamlessClone(np.uint8(frame), frame,
                                   np.uint8(255 * cv2.fillConvexPoly(np.zeros_like(frame), np.int32(hull2), (1, 1, 1))),
                                   center, cv2.NORMAL_CLONE)

        # Display the output
        cv2.imshow("Face Swap", output)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

