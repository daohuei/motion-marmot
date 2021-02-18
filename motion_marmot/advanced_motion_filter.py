import cv2
import typer
from simple_scene_classifier import SimpleSceneClassifier


class AdvancedMotionFilter():
    """
    Automation of the motion filter to improve the FP rate.
    """

    def __init__(self, ssc_model: str):
        self.ssc = SimpleSceneClassifier("For Advanced Motion Filter", ssc_model)
        self.mog2_mf = cv2.createBackgroundSubtractorMOG2()
        

    def __str__(self):
        return f"AdvancedMotionFilter(SimpleSceneClassifier={self.ssc})"


app = typer.Typer()


@app.command()
def run():
    amf = AdvancedMotionFilter('model/scene_knn_model')
    print(amf)


def main():
    """Main program"""
    app()


if __name__ == "__main__":
    main()
