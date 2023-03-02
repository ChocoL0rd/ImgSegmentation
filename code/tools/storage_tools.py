import torchvision.transforms as tfms
import os
import shutil
from PIL import Image
import logging

__all__ = [
    "img2mask"
]


def img2mask(in_dir: str, out_dir: str, clean: bool = False):
    """
        Creates if not exists directory with path out_dir.
        For every image in in_dir creates mask in out_dir.
    """

    logger = logging.getLogger(__name__)
    #
    # in_dir = os.path.abspath(in_dir)
    # out_dir = os.path.abspath(out_dir)

    logger.info(f"img2mask started.")
    if not os.path.isdir(in_dir):
        logger.error(f"Input directory {in_dir} doesn't exist.")
        raise Exception(f"{in_dir} doesn't exist.")

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        logger.info(f"Output directory {out_dir} didn't exist. So, created.")

    if clean:
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        logger.info(f"Output directory {out_dir} cleaned and recreated.")

    non_png_count = 0
    saved_count = 0
    non_rgba_count = 0

    for full_file_name in os.listdir(in_dir):
        file_path = os.path.join(in_dir, full_file_name)
        file_name, ext = os.path.splitext(full_file_name)
        if not ext == ".png":
            logger.warning(f"Extension of {file_path} is not '.png'.")
            non_png_count += 1
            continue

        with Image.open(file_path) as img:
            if not img.mode == 'RGBA':
                logger.warning(f"{file_path} is not RBA format")
                non_rgba_count += 1
                continue

            mask = img.split()[-1]
            mask_path = os.path.join(out_dir, f"{file_name}.jpeg")
            mask.save(mask_path)

        logger.info(f"Mask {mask_path} is saved.")
        saved_count += 1

    logger.info(f"img2mask finished. Not PNG: {non_png_count}, Not RGBA: {non_rgba_count}, SAVED: {saved_count}")
