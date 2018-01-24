#coding=utf-8
from PIL import Image, ImageFilter

im = Image.open(r'/home/kin/bg4.png')
im.filter(ImageFilter.BLUR).save(r'/home/kin/bg4.png')
