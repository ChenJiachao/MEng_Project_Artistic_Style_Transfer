
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=255,height=255):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("content/me3/*.jpg"):
    convertjpg(jpgfile,"content/me3/me3255/")
