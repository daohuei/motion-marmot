import typer
from motion_marmot.advanced_motion_filter import AdvancedMotionFilter

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
