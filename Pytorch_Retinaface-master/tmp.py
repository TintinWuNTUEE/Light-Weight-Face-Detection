import cv2 
    
# path 
path = r'C:\Users\Rajnish\Desktop\geeksforgeeks\geeks.png'
    
image = cv2.imread(path, 0)
window_name = 'Image'
start_point = (100, 50)
end_point = (125, 80)
color = (0, 0, 0)
thickness = -1
image = cv2.rectangle(image, start_point, end_point, color, thickness)
cv2.imshow(window_name, image)