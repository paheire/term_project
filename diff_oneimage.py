import cv2

#이미지 사이즈 변환
def resize_image(input_path, output_path, target_size=(800, 600)):
    img = cv2.imread(input_path)
    resized_img = cv2.resize(img, target_size, cv2.INTER_CUBIC)
    cv2.imwrite(output_path, resized_img)

resize_image('test.png', 'resized_img.jpg')
image = cv2.imread('resized_img.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape[:2]
split_line = int(width / 2)
left_part = gray[:, :split_line]
right_part = gray[:, split_line:]
'''
아래의 코드를 사용하여 구분된 이미지를 저장하여 확인 가능
left_img = image[:, :split_line]
right_img = image[:, split_line:]
cv2.imwrite('left_part.jpg', left_img)
cv2.imwrite('right_part.jpg', right_img)
'''

#차이 이미지 생성
diff = cv2.absdiff(left_part, right_part)
_, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
cv2.imwrite('Diff.jpg', thresholded)

#윤곽선 찾기
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#COLOR = 초록색
COLOR = (0,200,0)

#원본 이미지에 차이 이미지 윤곽선 그리기
for cnt in contours:
    x, y, width, height = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x,y), (x+width, y+height), COLOR, 2)

#결과 이미지 저장 및 출력
cv2.imwrite('result.jpg', image)

resized_image = cv2.resize(image, (800, 600))  # 출력할 크기 조절
cv2.imshow('difference', resized_image)
cv2.waitKey(0)