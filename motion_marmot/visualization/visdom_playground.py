import typer
import time
from visdom import Visdom


class VisdomPlayground():
    def __init__(self, hello):
        self.hello = hello
        self.viz = Visdom(port=8090)
        self.init_viz()

    def init_viz(self):
        self.viz.text(self.hello)

    def start(self):
        print("Visdom Playground Activating")
        while True:
            time.sleep(100)


app = typer.Typer()


@app.command()
def hello_world(hello: str):
    vp = VisdomPlayground(hello)
    vp.start()


def main():
    """Main program"""
    app()


if __name__ == "__main__":
    main()
