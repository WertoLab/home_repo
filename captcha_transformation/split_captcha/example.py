import os
import cv2
from matplotlib import pyplot as plt
from captcha_transformation.split_captcha.split_captcha import *

YOUR_PATH_DATASET = "split_captcha/datasets/out"
YOUR_PATH_SAVE_RESULT = "split_captcha/datasets/result/icons"

YOUR_NAME_CAPTCHA = "captcha_split"
YOUR_NAME_ICON = "icon"

def init_dataset():
    images = []
    for i in range(1, 2):
        img = cv2.imread(f"{YOUR_PATH_DATASET}/small_icons1.png")
        images.append(img)
    return images


def save_captcha(image,index):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(image)
        os.makedirs(
            os.path.dirname(f"{YOUR_PATH_SAVE_RESULT}/captcha/"),
            exist_ok=True)
        plt.savefig(f"{YOUR_PATH_SAVE_RESULT}/captcha/{YOUR_NAME_CAPTCHA}_{index}.png",
                    bbox_inches='tight',
                    pad_inches=0.0,
                )
        plt.close()


def save_icon(icon, index_image, index_icon):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(icon)
    os.makedirs(
        os.path.dirname(f"{YOUR_PATH_SAVE_RESULT}/icons/{index_image}/"),
        exist_ok=True)
    plt.savefig(f"{YOUR_PATH_SAVE_RESULT}/icons/{index_image}/{YOUR_NAME_ICON}_{index_icon}.png",
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300)
    plt.close()

def save_icon_V2(icon, index_image, index_icon):
    os.makedirs(
        os.path.dirname(f"{YOUR_PATH_SAVE_RESULT}/icons/{index_image}/"),
        exist_ok=True)
    cv2.imwrite(f"{YOUR_PATH_SAVE_RESULT}/icons/{index_image}/{YOUR_NAME_ICON}_{index_icon}.png",
                icon)

if __name__ == "__main__":
    images=init_dataset()
    for i in range (1,len(images)+1):
        img=images[i-1]
        icons = split_icons(img)
        ans = create_icons(image=img, icons=icons)
        for j in range (1,len(ans)+1):
            save_icon_V2(index_icon=j, index_image=i, icon=ans[j-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
