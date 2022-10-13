from parafields.field import load_schema, RandomField

import copy
import io
import numpy as np

try:
    # Check whether the extra requirements were installed
    import ipywidgets_jsonschema
    import ipywidgets
    import wrapt
    from IPython.display import display

    HAVE_JUPYYER_EXTRA = True
except ImportError:
    HAVE_JUPYYER_EXTRA = False


def return_proxy(creator):
    """A transparent proxy that can be returned from Jupyter UIs

    The created proxy object solves the general problem of needing to non-blockingly
    return from functions that display UI controls as a side effect. The returned object
    is updated whenever the widget state changes so that the return object would change.
    The proxy uses the :code:`wrapt` library to be as transparent as possible, allowing
    users to immediately work with the created object.
    :param creator:
        A callable that accepts no parameters and creates the object
        that should be returned. This is called whenever the widget
        state changes.
    :param widgets:
        A list of widgets whose changes should trigger a recreation of
        the proxy object.
    """

    class ObjectProxy(wrapt.ObjectProxy):
        def _update_(self):
            self.__wrapped__ = creator()

        def __copy__(self):
            return ObjectProxy(copy.copy(self.__wrapped__))

        def __deepcopy__(self, memo):
            return ObjectProxy(copy.deepcopy(self.__wrapped__, memo))

    # Create a new proxy object by calling the creator once
    proxy = ObjectProxy(creator())

    return proxy


def interactive_generate_field(comm=None, partitioning=None, dtype=np.float64):
    """Interactively explore field generation in a Jupyter notebook

    :param dtype:
        The floating point type to use. If the matching C++ type has not been
        compiled into the backend, an error is thrown.
    :type dtype: np.dtype

    :param comm:
        The mpi4py communicator that should be used to distribute this
        random field. Defaults to MPI_COMM_WORLD. Specifying this parameter
        when using sequential builds for parafields results in an error.

    :param partitioning:
        The tuple with processors per direction. The product of all entries
        is expected to match the number of processors in the communicator.
        Alternatively, a function can be provided that accepts the number of
        processors and the cell sizes as arguments.
    :type partitioning: list
    """

    # Return early if we do not have the extra
    if not HAVE_JUPYYER_EXTRA:
        print("Please re-run pip installation with 'parafields[jupyter]'")
        return

    # Create widgets for the configuration
    form = ipywidgets_jsonschema.Form(load_schema("stochastic.json"))

    # Set a default so that we immediately get a good output
    form.data = {
        "grid": {
            "cells": [512, 512],
            "extensions": [1.0, 1.0],
        },
        "stochastic": {
            "covariance": "exponential",
            "variance": 1.0,
            "corrLength": 0.05,
        },
    }

    # Output proxy object
    def _creator():
        return RandomField(form.data, comm=comm, partitioning=partitioning, dtype=dtype)

    proxy = return_proxy(_creator)

    # Image widget for output
    imagebox = ipywidgets.Box()

    # Add a button that displays a realization of the field
    realize = ipywidgets.Button(
        description="Show realization", layout=ipywidgets.Layout(width="100%")
    )

    def _realize(_):
        proxy._update_()

        # Maybe trigger visualization
        try:
            png = proxy._repr_png_()
        except RuntimeError as e:
            if e.args[0] == "negative eigenvalues in covariance matrix":
                imagebox.children = [
                    ipywidgets.Label(
                        "Negative eigenvalues in covariance matrix. Consider increasing"
                        " the embedding factor or explicitly allow approximation of results"
                        " (both in the 'Embedding' tab)."
                    )
                ]
                return
            else:
                raise e

        if png is None:
            imagebox.children = [
                ipywidgets.Label("This dimension cannot be visualized interactively.")
            ]
        else:
            imagebox.children = [ipywidgets.Image(value=png, format="png")]

    realize.on_click(_realize)

    # Start with a visualization
    realize.click()

    # Arrange the widgets into a grid layout
    app = ipywidgets.AppLayout(
        left_sidebar=form.widget,
        center=ipywidgets.VBox([realize, imagebox]),
        pane_widths=(1, 2, 0),
    )

    # Show the app as a side effect
    display(app)

    return proxy


def interactive_add_trend_component(field):
    # Return early if we do not have the extra
    if not HAVE_JUPYYER_EXTRA:
        print("Please re-run pip installation with 'parafields[jupyter]'")
        return

    # Create widgets for the configuration
    full_schema = load_schema("trend.json")
    dimschema = full_schema["anyOf"][field.dimension - 1]
    form = ipywidgets_jsonschema.Form(dimschema)
    added_component = False

    # Output proxy object
    def _creator():
        nonlocal added_component

        # Maybe remove previously added component
        if added_component:
            field._field.remove_trend_component()

        # Add the component
        field._add_trend_component(form.data)
        added_component = True

        return field

    proxy = return_proxy(_creator)

    # Image widget for output
    imagebox = ipywidgets.Box()

    # Add a button that displays a realization of the field
    realize = ipywidgets.Button(
        description="Show realization", layout=ipywidgets.Layout(width="100%")
    )

    def _realize(_):
        proxy._update_()
        png = proxy._repr_png_()
        if png is None:
            imagebox.children = [
                ipywidgets.Label("This dimension cannot be visualized interactively.")
            ]
        else:
            imagebox.children = [
                ipywidgets.Image(value=proxy._repr_png_(), format="png")
            ]

    realize.on_click(_realize)

    # Start with a visualization
    realize.click()

    # Arrange the widgets into a grid layout
    app = ipywidgets.AppLayout(
        left_sidebar=form.widget,
        center=ipywidgets.VBox([realize, imagebox]),
        pane_widths=(1, 2, 0),
    )

    # Show the app as a side effect
    display(app)

    return proxy
