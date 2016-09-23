import sys
from PIL import Image

if not len(sys.argv) == 2:
	print('usage: python3 %s <file name>' % sys.argv[0])
	exit()

try:
	im = Image.open(sys.argv[1])
	im.rotate(180).save('ans2.png', 'png')
except:
	print('rotating image failed, maybe file not exist?')
