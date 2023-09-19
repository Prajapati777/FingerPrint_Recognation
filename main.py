import os          
import cv2

sample = cv2.imread("D:/codes/.vscode/FingerPrintRecognation/SOCOFing/Altered/Altered-Hard/184__M_Right_thumb_finger_CR.BMP")

best_score=0
filename = None
image = None
kp1, kp2, mp = None, None, None

counter = 0

for file in [file for file in os.listdir("D:/codes/.vscode/FingerPrintRecognation/SOCOFing/Real")][:1000]:
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter +=1
    fingerprint_image=cv2.imread( "D:/codes/.vscode/FingerPrintRecognation/SOCOFing/Real/" + file) 
    sift = cv2.SIFT_create()
    
    keypoint_1, description_1 = sift.detectAndCompute(sample, None)
    keypoint_2, description_2 = sift.detectAndCompute(fingerprint_image, None)
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},
                                    {}).knnMatch(description_1, description_2, k=2)
    
    match_point = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_point.append(p)
    
    keypoints = 0
    if len(keypoint_1) < len(keypoint_2):
        keypoints = len(keypoint_1)
    else:
        keypoints= len(keypoint_2)

    if len(match_point) / keypoints * 100 > best_score:
        best_score = len(match_point) / keypoints * 100
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoint_1, keypoint_2, match_point

print("Best match :" + filename)
print("Score: " + str(best_score))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=3, fy=3)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows