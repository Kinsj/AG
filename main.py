# coding=utf-8
import cv2
import aircv as ac

DEBUG = False


def mathc_img(image, target, value): # not good, need to fix
    import cv2
    import numpy as np
    img_rgb = cv2.imread(image)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(target, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
    threshold = value
    loc = np.where(res < 0.2)
    print(loc[0], loc[1])
    print(len(loc[0]))
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (7, 249, 151), 2)
        cv2.imshow('Detected', img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 这部分参照了aircv的代码
def find_template(im_source, im_search, threshold=0.5, rgb=False, bgremove=False):
    '''
    @return find location
    if not found; return None
    '''
    result = find_all_template(im_source, im_search, threshold, 1, rgb, bgremove)
    return result[0] if result else None


def find_all_template(im_source, im_search, threshold=0.5, maxcnt=0, rgb=False, bgremove=False):
    '''
    Locate image position with cv2.templateFind

    Use pixel match to find pictures.

    Args:
        im_source(string): 图像、素材
        im_search(string): 需要查找的图片
        threshold: 阈值，当相识度小于该阈值的时候，就忽略掉

    Returns:
        A tuple of found [(point, score), ...]

    Raises:
        IOError: when file read error
    '''
    # 模板匹配的匹配方式
    # 在OpenCv和EmguCv中支持以下6种对比方式：
    # CV_TM_SQDIFF    平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
    # CV_TM_CCORR    相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
    # CV_TM_CCOEFF    相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
    # CV_TM_SQDIFF_NORMED    归一化平方差匹配法
    # CV_TM_CCORR_NORMED    归一化相关匹配法
    # CV_TM_CCOEFF_NORMED    归一化相关系数匹配法
    # method = cv2.TM_CCORR_NORMED
    # method = cv2.TM_SQDIFF_NORMED
    method = cv2.TM_CCOEFF_NORMED

    if rgb:
        s_bgr = cv2.split(im_search) # Blue Green Red
        i_bgr = cv2.split(im_source)
        weight = (0.3, 0.3, 0.4)
        resbgr = [0, 0, 0]
        for i in range(3): # bgr
            resbgr[i] = cv2.matchTemplate(i_bgr[i], s_bgr[i], method)
            # print(resbgr[i])
        res = resbgr[0]*weight[0] + resbgr[1]*weight[1] + resbgr[2]*weight[2]
        # print("res:\n",res)
    else:
        s_gray = cv2.cvtColor(im_search, cv2.COLOR_BGR2GRAY)
        i_gray = cv2.cvtColor(im_source, cv2.COLOR_BGR2GRAY)
        # 边界提取(来实现背景去除的功能)
        if bgremove:
            s_gray = cv2.Canny(s_gray, 100, 200)
            i_gray = cv2.Canny(i_gray, 100, 200)

        res = cv2.matchTemplate(i_gray, s_gray, method)
    w, h = im_search.shape[1], im_search.shape[0]

    result = []
    while True:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        if DEBUG:
            print('templmatch_value(thresh:%.1f) = %.3f' %(threshold, max_val)) # not show debug
        if method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCOEFF] and max_val < 0.4:
            break
        elif method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and min_val > 0.5:
            break
        # calculator middle point
        middle_point = (top_left[0]+w/2, top_left[1]+h/2)
        if method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCOEFF]:
            result.append(dict(
                result=middle_point,
                rectangle=(top_left, (top_left[0], top_left[1] + h), (top_left[0] + w, top_left[1]), (top_left[0] + w, top_left[1] + h)),
                confidence=max_val
            ))
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            result.append(dict(
                result=middle_point,
                rectangle=(top_left, (top_left[0], top_left[1] + h), (top_left[0] + w, top_left[1]), (top_left[0] + w, top_left[1] + h)),
                confidence=min_val
            ))
        if maxcnt and len(result) >= maxcnt:
            break
        # floodfill the already found area
        cv2.floodFill(res, None, max_loc, (-1000,), max_val-threshold+0.1, 1, flags=cv2.FLOODFILL_FIXED_RANGE)
    return result


def tuple_f2i(tuple_arr):
    res = []
    for i in tuple_arr:
        res.append(int(i))
    return tuple(res)


# print circle_center_pos
def draw_circle(img, pos, circle_radius, color=(7, 249, 151), line_width=2):
    cv2.circle(img, pos, circle_radius, color, line_width)
    cv2.imshow('objDetect', imsrc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_rectangle(img, left_top, w, h, l_color=(7, 249, 151), l_width=2):
    cv2.rectangle(img, left_top, (left_top[0] + w, left_top[1] + h), l_color, l_width)
    cv2.imshow('Detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = '/home/kin/bg3.png'
    target = '/home/kin/obj.png'

    imsrc = ac.imread(image)
    imobj = ac.imread(target)
    # find the match position
    pos = find_template(imsrc, imobj, rgb=True, bgremove=False)
    print(pos)
    if not (pos is None):
        circle_center_pos = pos['result']
        circle_center_pos = tuple_f2i(circle_center_pos)
        print(circle_center_pos)
        w, h = imobj.shape[:-1]
        # draw circle
        draw_circle(imsrc, circle_center_pos, 4, line_width=7)
    else:
        print("Cannot find the target")
