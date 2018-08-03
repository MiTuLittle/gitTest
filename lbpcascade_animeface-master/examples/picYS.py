from PIL import Image
import os


class ImageCompressUtil(object):
    # 等比例压缩
    def resizeImg(self, **args):
        try:
            args_key = {'ori_img': '', 'dst_img': '',
                        'dst_w': '', 'dst_h': '', 'save_q': 100}
            arg = {}
            for key in args_key:
                if key in args:
                    arg[key] = args[key]
            im = Image.open(arg['ori_img'])
            if im.format in ['gif', 'GIF', 'Gif']:
                return
            ori_w, ori_h = im.size
            widthRatio = heightRatio = None
            ratio = 1
            if (ori_w and ori_w > arg['dst_w']) or (ori_h and ori_h > arg['dst_h']):
                if arg['dst_w'] and ori_w > arg['dst_w']:
                    widthRatio = float(arg['dst_w']) / ori_w  # 正确获取小数的方式
                if arg['dst_h'] and ori_h > arg['dst_h']:
                    heightRatio = float(arg['dst_h']) / ori_h
                if widthRatio and heightRatio:
                    if widthRatio < heightRatio:
                        ratio = widthRatio
                    else:
                        ratio = heightRatio
                if widthRatio and not heightRatio:
                    ratio = widthRatio
                if heightRatio and not widthRatio:
                    ratio = heightRatio
                newWidth = int(ori_w * ratio)
                newHeight = int(ori_h * ratio)
            else:
                newWidth = ori_w
                newHeight = ori_h
            if len(im.split()) == 4:
                # prevent IOError: cannot write mode RGBA as BMP
                r, g, b, a = im.split()
                im = Image.merge("RGB", (r, g, b))
            im.resize((newWidth, newHeight), Image.ANTIALIAS).save(
                arg['dst_img'], quality=arg['save_q'])
        except Exception as e:
            #LogDao.warn(u'压缩失败' + str(e), belong_to='resizeImg')
            pass


# 目标图片大小
dst_w = 790
dst_h = 0
# 保存的图片质量
save_q = 100
srcPath = 'D:\\picSet\\cutPic\\src\\1905.jpg'
savePath = 'D:\\picSet\\cutPic\\src\\save1.jpg'
ImageCompressUtil().resizeImg(
    ori_img=srcPath,
    dst_img=savePath,
    dst_w=dst_w,
    dst_h=dst_h,
    save_q=save_q
)
