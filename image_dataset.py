from ai2thor.controller import Controller
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
from PIL import Image
import json


class Wnid:
    """
    The wnid, and the number of images per wnid.
    """
    def __init__(self, wnid: str, object_types: List[str], count: int = 0):
        """
        :param object_types: The AI2Thor object types.
        :param wnid: The ImageNet wnid.
        :param count: The current count of images.
        """

        self.object_types = object_types
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

        self.wnids: Dict[str, Wnid] = {}
        # Load existing progress.
        if self.progress_filepath.exists():
            data = json.loads(self.progress_filepath.read_text(encoding="utf-8"))
            self.scene_index = data["scene_index"]
            for key in data["progress"]:
                w = Wnid(object_types=data["progress"][key]["object_types"], wnid=data["progress"][key]["wnid"], count=data["progress"][key]["count"])
                self.wnids.update({key: w})
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
            if self.wnids[object_type].count < self.train_per_wnid + self.val_per_wnid:
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

    def increment_scene_index(self) -> None:
        """
        Increment the scene index value and loop to 0 if needed.
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

    def run(self, grid_size: float = 0.25, images_per_position: int = 1, pixel_percent_threshold: float = 0.01) -> None:
        """
        Generate an image dataset.

        :param grid_size: The AI2Thor room grid size (see AI2Thor documentation).
        :param images_per_position: Every time the avatar teleports, capture this many images.
        :param pixel_percent_threshold: Objects must occupy this percentage of pixels in the segmentation color pass or greater to be saved to disk as an image.
        """

        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True)
        # Load a new scene.
        controller = Controller(scene=ImageDataset.SCENES[self.scene_index], gridSize=grid_size, renderObjectImage=True)

        while not self.done():
            event = controller.step(action='GetReachablePositions')
            positions = event.metadata["actionReturn"]

            if positions is None:
                self.increment_scene_index()
                continue

            for position in positions:
                # Reposition the objects.
                controller.reset(scene=ImageDataset.SCENES[self.scene_index])
                controller.step(action='InitialRandomSpawn', randomSeed=0, forceVisible=True, numPlacementAttempts=5)

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
                                            wnid = self.get_wnid(obj["objectType"])
                                            # If this is an object type we don't care about, skip it.
                                            if wnid is None:
                                                continue
                                            # If we already have enough images in this category, skip it.
                                            elif wnid.count >= self.train_per_wnid + self.val_per_wnid:
                                                continue
                                            # Save the image.
                                            else:
                                                self.save_image(event.frame, obj["name"], wnid.count, wnid.wnid)
                                                w = wnid.wnid
                                                self.wnids[w].count += 1
            # Next scene.
            self.increment_scene_index()

    def end(self) -> None:
        """
        End the script for now. Save a progress file.
        """

        progress = dict()
        for wnid in self.wnids:
            progress.update({wnid: self.wnids[wnid].__dict__})
        save_file = {"scene_index": self.scene_index, "progress": progress}
        self.progress_filepath.write_text(json.dumps(save_file), encoding="utf-8")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="ai2thor_image_dataset", help="Root output directory in <home>/")
    image_dataset = ImageDataset(Path.home().joinpath("ai2thor_image_dataset"))
    try:
        image_dataset.run()
    finally:
        image_dataset.end()
