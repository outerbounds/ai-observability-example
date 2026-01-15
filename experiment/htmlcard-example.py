from metaflow import FlowSpec, step, IncludeFile, pypi, card, IncludeFile
import json


class HTMLCardExample(FlowSpec):

    card_template = IncludeFile("card_template", default="glow.html")

    @card(type="html")
    @step
    def start(self):
        from metaflow.plugins.cards.card_modules import chevron
        import time

        time.sleep(20)

        data = {"example": 123}
        self.html = chevron.render(
            self.card_template, dict(data=json.dumps(data), title="Hello HTML!")
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    HTMLCardExample()
