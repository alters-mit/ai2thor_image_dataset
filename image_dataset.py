from ai2thor.controller import Controller
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from PIL import Image
import io
import json


class Wnid:
    """
    The wnid, and the number of images per wnid.
    """
    def __init__(self, object_type: str, wnid: str, count: int = 0):
        """
        :param object_type: The AI2Thor object type.
        :param wnid: The ImageNet wnid.
        :param count: The current count of images.
        """

        self.object_type = object_type
        self.wnid = wnid
        self.count = count


class ImageDataset:
    RNG = np.random.RandomState(0)
    NUM_PIXELS = 300.0 * 300

    SCENES: List[str] = []
    for i in range(1, 31):
        SCENES.append(f"FloorPlan{i}")
    for i in range(201, 231):
        SCENES.append(f"FloorPlan{i}")
    for i in range(301, 331):
        SCENES.append(f"FloorPlan{i}")
    for i in range(401, 431):
        SCENES.append(f"FloorPlan{i}")

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

        self.object_types: Dict[str, Wnid] = {}
        # Load existing progress.
        if self.progress_filepath.exists():
            data = json.loads(self.progress_filepath.read_text(encoding="utf-8"))
            for key in data:
                w = Wnid(data[key]["object_type"], data[key]["wnid"], data[key]["count"])
                self.object_types.update({key: w})
        # Create new progress.
        else:
            # Parse the AI2Thor/wnid spreadsheet.
            csv = Path("object_types.csv").read_text()
            for line in csv.split("\n"):
                if line == "":
                    continue
                row = line.split(",")
                self.object_types.update({row[0]: Wnid(object_type=row[0], wnid=row[1])})

        self.train_per_wnid = self.train / len(self.object_types)
        self.val_per_wnid = self.val / len(self.object_types)

    def done(self) -> bool:
        """
        Returns true if the dataset is done.
        """

        for object_type in self.object_types:
            if self.object_types[object_type].count < self.train_per_wnid + self.val_per_wnid:
                return False
        return True

    def save_image(self, image: np.array, object_name: str, count: int, wnid: str) -> None:
        """
        Save an image.

        :param image: The image as a numpy array.
        :param object_name: The name of the object.
        :param count: Append this number to the filename.
        :param wnid: The wnid directory.
        """

        if count < self.train_per_wnid:
            dest = self.root_dir.joinpath("train").joinpath(wnid)
            object_count = count
        else:
            dest = self.root_dir.joinpath("val").joinpath(wnid)
            object_count = count - self.train_per_wnid
        if not dest.exists():
            dest.mkdir(parents=True)

        dest = dest.joinpath(f"{object_name}_{str(object_count).zfill(4)}.jpg")
        # Save the image.
        Image.fromarray(image).resize((256, 256), Image.LANCZOS).save(str(dest.resolve()))

    def run(self, grid_size: float = 0.25, images_per_position: int = 10, pixel_percent_threshold: float = 0.05) -> None:
        """
        Generate an image dataset.

        :param grid_size: The AI2Thor room grid size (see AI2Thor documentation).
        :param images_per_position: Every time the avatar teleports, capture this many images.
        :param pixel_percent_threshold: Objects must occupy this percentage of pixels in the segmentation color pass or greater to be saved to disk as an image.
        """

        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True)

        scene_index = 0
        # Load a new scene.
        controller = Controller(scene=ImageDataset.SCENES[scene_index], gridSize=grid_size, renderObjectImage=True)
        first_time_only = True
        while not self.done():
            # Load a new scene.
            if first_time_only:
                first_time_only = False
            else:
                controller.reset(scene=ImageDataset.SCENES[scene_index])

            # Next scene.
            scene_index += 1
            if scene_index >= len(ImageDataset.SCENES):
                scene_index = 0

            event = controller.step(action='GetReachablePositions')
            positions = event.metadata["actionReturn"]

            for position in positions:
                # Teleport to the position.
                controller.step(action='Teleport', x=position["x"], y=position["y"], z=position["z"])
                for i in range(images_per_position):
                    # Step through the simulation with a random movement or rotation.
                    event = controller.step(action=ImageDataset.ACTIONS[
                        ImageDataset.RNG.randint(0, len(ImageDataset.ACTIONS))])

                    # Segmentation colors to object IDs map.
                    object_colors = event.color_to_object_id

                    # Get the unique colors in the image and how many pixels per color.
                    colors, counts = np.unique(event.instance_segmentation_frame.reshape(-1, 3),
                                               return_counts=True,
                                               axis=0)
                    for color, count in zip(colors, counts):
                        for object_color in object_colors:
                            if object_color == tuple(color):
                                percent = count / ImageDataset.NUM_PIXELS
                                if percent > pixel_percent_threshold:
                                    for obj in event.metadata["objects"]:
                                        if obj["objectId"] == object_colors[object_color]:
                                            obj_type = obj["objectType"]
                                            # If this is an object type we don't care about, skip it.
                                            if obj_type not in self.object_types:
                                                continue
                                            # If we already have enough images in this category, skip it.
                                            elif self.object_types[obj_type].count >= self.train_per_wnid + self.val_per_wnid:
                                                continue
                                            # Save the image.
                                            else:
                                                self.save_image(event.frame, obj["name"], self.object_types[obj_type].count, self.object_types[obj_type].wnid)
                                                self.object_types[obj_type].count += 1

    def end(self) -> None:
        """
        End the script for now. Save a progress file.
        """

        progress = dict()
        for object_type in self.object_types:
            progress.update({object_type: self.object_types[object_type].__dict__})
        self.progress_filepath.write_text(json.dumps(progress), encoding="utf-8")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="ai2thor_image_dataset", help="Root output directory in <home>/")
    image_dataset = ImageDataset(Path.home().joinpath("ai2thor_image_dataset"))
    try:
        image_dataset.run()
    finally:
        image_dataset.end()
