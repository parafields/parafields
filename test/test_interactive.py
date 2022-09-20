from parafields.interactive import *


def test_interactive_generate_field():
    app = interactive_generate_field()
    if not HAVE_JUPYYER_EXTRA:
        assert app is None
