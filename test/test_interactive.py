from parafields.interactive import *

from parafields.field import generate_field


def test_interactive_generate_field():
    app = interactive_generate_field()
    if not HAVE_JUPYYER_EXTRA:
        assert app is None


def test_interactive_generate_field():
    field = generate_field()
    app = interactive_add_trend_component(field)
    if not HAVE_JUPYYER_EXTRA:
        assert app is None
