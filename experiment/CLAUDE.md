
# Defining a card 

Define cards in a separate module which is called in a Metaflow step.
Do not intermingle visualization code with other business logic.

Read the documentation for defining cards at

https://docs.metaflow.org/metaflow/visualizing-results

and see examples at

https://github.com/outerbounds/dynamic-card-examples

For charts, use the `VegaChart` component. Find examples of Vega charts at

https://vega.github.io/vega-lite/examples/

## Defining an advanced card

If you need to provide advanced functionality, e.g. custom visualization components or layouts
which can't be achieved with the default card components provided by Metaflow, leverage
HTML-type cards (`@card(type='html')`) which allow you to define an arbitrary self-contained
HTML file (a single file) using any Javascript and other libraries.

See `htmlcard-example.py` for an example.

# Testing a step

Execute a single step in Metaflow using the `spin` command:
```
python flow.py --environment=pypi spin start
```
This will update cards attached in the `start` step.

# Inspecting a card

You can view the latest card produced by a step, e.g. `start`, by
fetching it with the following command:

```
python flow.py --environment=pypi --mode=spin card get start > card.html
```

and then render it as a png, saved in `shot.png`

```
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --headless=new --window-size=1920,1080 --screenshot=shot.png card.html

```

open `shot.png` and ensure that the result looks correct.

