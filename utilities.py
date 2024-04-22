class GrayscaleToRGB(object):
    def __call__(self, img):
        return img.convert('RGB')