import cv2
import aircv as ac
# import pyscreenshot as ImageGrab
from PIL import Image, ImageGrab
import time
from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()


def tuple_f2i(t):
    res = []
    for i in t:
        res.append(int(i))
    return tuple(res)


# print circle_center_pos
def draw_circle(img, pos, circle_radius, color, line_width):
    cv2.circle(img, pos, circle_radius, color, line_width)
    cv2.imshow('objDetect', imsrc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        # imsrc = ac.imread('C:\\Users\\SakataKin\\Desktop\\src.jpg')
        im = ImageGrab.grab()
        im.save('d:\\grabtmp.jpg')
        imsrc = ac.imread('d:\\grabtmp.jpg')
        imobj = ac.imread('C:\\Users\\SakataKin\\Desktop\\tag.png')

        # find the match position
        pos = ac.find_template(imsrc, imobj, rgb=True, bgremove=False)
        print(pos)
        if not (pos is None):
            circle_center_pos = tuple_f2i(pos['result'])
            print(circle_center_pos)
            circle_radius = 50
            color = (0, 255, 0)
            line_width = 10

            draw_circle(imsrc, circle_center_pos, circle_radius, color, line_width)
            time.sleep(3)
        else:
            print("Cannot find the target")
            time.sleep(3)
