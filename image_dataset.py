from ai2thor.controller import Controller
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
from PIL import Image
import json
from tqdm import tqdm
from time import clock
from sys import maxsize


class Wnid:
    """
    The wnid, and the number of images per wnid.
    """

    def __init__(self, wnid: str, object_types: List[str], count: int = 0):
        """
        :param object_types: The AI2Thor object types.
        :param wnid: The ImageNet wnid.
        :param count: The current number of images in this wnid that have been saved to disk.
        """

        self.object_types = object_types
        self.wnid = wnid
        self.count = count


class ImageDataset:
    """
    Generate an object image dataset using AI2-Thor.
    """

    RNG = np.random.RandomState(0)
    NUM_PIXELS = 300.0 * 300

    # Every valid AI2-Thor scene name.
    SCENES: List[str] = []
    for i in range(1, 31):
        SCENES.append(f"FloorPlan{i}")
    for i in range(201, 231):
        SCENES.append(f"FloorPlan{i}")
    for i in range(301, 331):
        SCENES.append(f"FloorPlan{i}")
    for i in range(401, 431):
        SCENES.append(f"FloorPlan{i}")

    # The avatar will randomly choose one of these actions per step.
    ACTIONS = ["RotateRight", "RotateLeft", "LookUp", "LookDown", "MoveAhead", "MoveRight", "MoveLeft", "MoveBack"]

    def __init__(self, root_dir: Union[str, Path], train: int = 1300000, val: int = 50000):
        """
        :param root_dir: The root output directory.
        :param train: The total number of train images.
        :param val: The total number of val images.
        """

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.progress_filepath = root_dir.joinpath("progress.json")

        # Create an images/ directory.
        root_dir = root_dir.joinpath("images")
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        self.root_dir = root_dir
        self.train = train
        self.val = val

        self.pbar = tqdm(total=train + val)

        self.wnids: Dict[str, Wnid] = {}
        # Load existing progress.
        if self.progress_filepath.exists():
            data = json.loads(self.progress_filepath.read_text(encoding="utf-8"))
            self.scene_index = data["scene_index"]
            for key in data["progress"]:
                w = Wnid(object_types=data["progress"][key]["object_types"], wnid=data["progress"][key]["wnid"], count=data["progress"][key]["count"])
                self.wnids.update({key: w})
                # Increment the progress bar by the image count so far.
                self.pbar.update(w.count)
        # Create new progress.
        else:
            self.scene_index = 0
            # Parse the AI2Thor/wnid spreadsheet.
            csv = Path("object_types.csv").read_text()
            for line in csv.split("\n"):
                if line == "":
                    continue
                row = line.split(",")
                wnid = row[1]
                if wnid not in self.wnids:
                    self.wnids.update({wnid: Wnid(wnid=wnid, object_types=[])})
                self.wnids[wnid].object_types.append(row[0])

        self.train_per_wnid = self.train / len(self.wnids)
        self.val_per_wnid = self.val / len(self.wnids)

    def done(self) -> bool:
        """
        Returns true if the dataset is done.
        """

        for object_type in self.wnids:
            # If any wnid is missing images, the dataset is not done.
            if self.wnids[object_type].count < self.train_per_wnid + self.val_per_wnid:
                return False
        return True

    def save_image(self, image: np.array, object_type: str, count: int, wnid: str) -> None:
        """
        Save an image.

        :param image: The image as a numpy array.
        :param object_type: The typwe of the object.
        :param count: Append this number to the filename.
        :param wnid: The wnid directory.
        """

        if count < self.train_per_wnid:
            dest = self.root_dir.joinpath("train").joinpath(wnid)
            object_count = count
        else:
            dest = self.root_dir.joinpath("val").joinpath(wnid)
            object_count = count - int(self.train_per_wnid)
        if not dest.exists():
            dest.mkdir(parents=True)

        # Resize the image and save it.
        Image.fromarray(image).resize((256, 256), Image.LANCZOS).save(
            str(dest.joinpath(f"{object_type}_{str(object_count).zfill(4)}.jpg").resolve()))
        # Increment the progress bar.
        self.pbar.update(1)

    def increment_scene_index(self) -> None:
        """
        Increment the scene index value and loop back to 0 if needed.
        """

        self.scene_index += 1
        if self.scene_index >= len(ImageDataset.SCENES):
            self.scene_index = 0

    def get_wnid(self, object_type: str) -> Optional[Wnid]:
        """
        Returns the wnid associated with the AI2Thor object type.

        :param object_type: The AI2Thor object type.
        """

        for w in self.wnids:
            if object_type in self.wnids[w].object_types:
                return self.wnids[w]
        return None

    def run(self, grid_size: float = 0.25, images_per_scene: int = 100, pixel_percent_threshold: float = 0.01) -> None:
        """
        Generate an image dataset.

        :param grid_size: The AI2Thor room grid size (see AI2Thor documentation).
        :param images_per_scene: Capture this many images before loading a new scene.
        :param pixel_percent_threshold: Objects must occupy >= this percentage of pixels in the segmentation mask to be saved to disk as an image.
        """

        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True)
        # Load a new scene.
        controller = Controller(scene=ImageDataset.SCENES[self.scene_index], gridSize=grid_size, renderObjectImage=True)

        t0 = clock()
        # The number of times images were acquired very slowly.
        num_slow_images = 0

        while not self.done():
            # Load the next scene and populate it.
            controller.reset(scene=ImageDataset.SCENES[self.scene_index])
            controller.step(action='InitialRandomSpawn', randomSeed=ImageDataset.RNG.randint(-maxsize, maxsize),
                            forceVisible=True, numPlacementAttempts=5)

            accept_all_images = False

            for i in range(images_per_scene):
                # Step through the simulation with a random movement or rotation.
                event = controller.step(action=ImageDataset.ACTIONS[
                    ImageDataset.RNG.randint(0, len(ImageDataset.ACTIONS))])

                # Get the segmentation colors to object IDs map.
                object_colors = event.color_to_object_id

                # Get the unique colors in the image and how many pixels per color.
                colors, counts = np.unique(event.instance_segmentation_frame.reshape(-1, 3),
                                           return_counts=True,
                                           axis=0)
                saved_image = False
                for color, count in zip(colors, counts):
                    for object_color in object_colors:
                        # Match an object to the segmentation mask.
                        if object_color == tuple(color):
                            # Save an image tagged as an object if there are enough pixels in the segmentation pass.
                            percent = count / ImageDataset.NUM_PIXELS
                            if percent > pixel_percent_threshold:
                                for obj in event.metadata["objects"]:
                                    if obj["objectId"] == object_colors[object_color]:
                                        wnid = self.get_wnid(obj["objectType"])
                                        # If this is an object type we don't care about, skip it.
                                        if wnid is None:
                                            continue
                                        # If we already have enough images in this category, skip it.
                                        elif wnid.count >= self.train_per_wnid + self.val_per_wnid:
                                            continue
                                        # Save the image.
                                        else:
                                            self.save_image(event.frame, obj["objectType"], wnid.count, wnid.wnid)
                                            saved_image = True

                                            w = wnid.wnid
                                            self.wnids[w].count += 1
                if saved_image and not accept_all_images:
                    t1 = clock()
                    dt = t1 - t0
                    t0 = t1
                    # If this image was acquired slowly, increment the total.
                    if dt > 3:
                        num_slow_images += 1
                    # If there haven't been new images in a while, accept all images.
                    if num_slow_images >= 100:
                        pixel_percent_threshold = 0
                        accept_all_images = True
                        print("There haven't been new images in a while... "
                              "Reducing pixel percent threshold to 0.")
            # Next scene.
            self.increment_scene_index()

    def end(self) -> None:
        """
        End the script for now. Save a progress file. This file will be used to avoid overwriting existing progress.
        """

        # Create a "save file".
        progress = dict()
        for wnid in self.wnids:
            progress.update({wnid: self.wnids[wnid].__dict__})
        save_file = {"scene_index": self.scene_index, "progress": progress}
        self.progress_filepath.write_text(json.dumps(save_file), encoding="utf-8")

        # Stop the progress bar.
        self.pbar.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    from distutils import dir_util
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="ai2thor_image_dataset", help="Root output directory in <home>/")
    parser.add_argument("--new", action="store_true", help="Delete an existing dataset at the output directory.")
    args = parser.parse_args()

    output_dir = Path.home().joinpath(args.dir)
    # Delete an existing dataset.
    if args.new:
        dir_util.remove_tree(str(output_dir.resolve()))

    image_dataset = ImageDataset(output_dir)
    try:
        image_dataset.run()
    finally:
        image_dataset.end()
