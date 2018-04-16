import cv2


def resize(img, w, h):
	img = cv2.resize(img, (w, h), interpolation = cv2.INTER_CUBIC)
	return img

def save(filename, img):
	cv2.imwrite(filename, img)

def showImg(img, name = 'test'):
	cv2.imshow(name, img)
	cv2.waitKey()
	cv2.destroyAllWindows()


def showFromFilename(filename, name = 'test'):
	img = cv2.imread(filename, 1)
	cv2.imshow(name, img)
	cv2.waitKey()
	cv2.destroyAllWindows()


def read(filename, channel):
	return cv2.imread(filename, channel)