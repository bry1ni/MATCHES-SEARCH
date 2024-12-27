import os
import cv2

queries_folder_path = "images/queries/"
cars_folder_path = "images/License Plates"

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()


def search():
    global best_matches_dict
    best_matches_dict = {}
    # loop through all query images
    for query_filename in os.listdir(queries_folder_path):
        if query_filename.endswith(".png"):

            imgMAT = cv2.imread(os.path.join(queries_folder_path, query_filename))
            gray2 = cv2.cvtColor(imgMAT, cv2.COLOR_BGR2GRAY)
            keypointsMAT, descriptorsMAT = sift.detectAndCompute(gray2, None)
            best_match = None
            best_match_ratio = 0
            # loop through all license plates images
            for car_filename in os.listdir(cars_folder_path):
                if car_filename.endswith(".jpg"):
                    img = cv2.imread(os.path.join(cars_folder_path, car_filename))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    keypoints, descriptors = sift.detectAndCompute(gray, None)
                    matches = bf.knnMatch(descriptorsMAT, descriptors, k=2)

                    good_matches = 0
                    for match in matches:
                        if len(match) > 1:
                            m, n = match
                            if m.distance < 0.75 * n.distance:
                                good_matches += 1
                    match_ratio = good_matches / len(matches) if len(matches) > 0 else 0

                    if match_ratio > best_match_ratio:
                        best_match_ratio = match_ratio
                        best_match = img  # best image hiya li kan fiha max ratio

            if best_match is not None:
                best_matches_dict[query_filename] = best_match

    return best_matches_dict


best_matches_dict = search()
if best_matches_dict:
    for query_filename, best_match in best_matches_dict.items():

        img = cv2.imread(os.path.join(queries_folder_path, query_filename))
        key, desc = sift.detectAndCompute(img, None)
        keymatch, descmatch = sift.detectAndCompute(best_match, None)
        matches = bf.knnMatch(desc, descmatch, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        img_matches = cv2.drawMatchesKnn(img, key, best_match, keymatch, None, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow(f"Matches for {query_filename}", img_matches)

cv2.waitKey(0)
cv2.destroyAllWindows()
