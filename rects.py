__author__ = 'mohamed'

import cv2


def outline_rect(image, rect, color):
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), color)


# Copy part of the source to part of the destination, defined by rectangles
def copy_rect(src, dst, srcRect, dstRect,
              interpolation=cv2.INTER_LINEAR):
    xs, ys, ws, hs = srcRect
    xd, yd, wd, hd = dstRect

    # Resize the content of the source rectangle, puts it in the destination rectangle

    dst[yd:yd + hd, xd:xd + wd] = cv2.resize(src[ys:ys + hs, xs:xs + ws], (wd, hd), interpolation)