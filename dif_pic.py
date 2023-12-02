import cv2
import numpy as np

def find_differences(image1_path, image2_path):
    # 이미지 읽기
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 이미지 크기가 다르면 같게 조절
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 이미지를 그레이 스케일로 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 히스토그램 계산
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    # 히스토그램 비교
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # 유사도에 따라 경계 상자 그리기
    if similarity < 0.95:  # 임계값은 조절 가능
        print("이미지가 다릅니다.")
        diff_image = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
        # 결과 이미지 저장
        cv2.imwrite("result_image.png", img1)
    else:
        print("이미지가 유사합니다.")

if __name__ == "__main__":
    image1_path = "image1.jpg"
    image2_path = "image2.jpg"
    find_differences(image1_path, image2_path)
