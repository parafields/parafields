import collections.abc
import jsonschema
import numpy as np
import parafields._parafields as _parafields
import time

from matplotlib import cm
from parafields.mpi import default_partitioning, MPI
from parafields.utils import is_iterable, load_schema
from PIL import Image


def dict_to_parameter_tree(data, tree=None, prefix=""):
    """Convert a (nested) dictionary to a C++ parameter tree structure"""
    if tree is None:
        tree = _parafields.ParameterTree()
    for k, v in data.items():
        if isinstance(v, dict):
            dict_to_parameter_tree(v, tree=tree, prefix=prefix + k + ".")
        else:
            if is_iterable(v):
                v = " ".join([str(x) for x in v])
            tree.set(prefix + k, str(v))
    return tree


def validate_config(config, schema="stochastic.json"):
    """Validate a given configuration against the provided schema"""

    # Validate the given config against the schema
    schema = load_schema(schema)
    jsonschema.validate(instance=config, schema=schema)

    return config


def generate_field(
    cells=(512, 512),
    extensions=(1.0, 1.0),
    covariance="exponential",
    variance=1.0,
    anisotropy="none",
    corrLength=0.05,
    periodic=False,
    embedding_factor=2,
    embedding_type="classical",
    sigmoid_function="smoothstep",
    threshold=1e-14,
    approximate=False,
    fftw_transpose=None,
    cacheInvMatvec=True,
    cacheInvRootMatvec=False,
    cg_iterations=100,
    cauchy_alpha=1.0,
    cauchy_beta=1.0,
    exp_gamma=1.0,
    transform=None,
    dtype=np.float64,
    seed=None,
    partitioning=None,
    comm=None,
):
    """Main entry point for generating parafields parameter fields

    :param cells:
        The number of cells in each direction in the grid that defines the
        random field resolution.
    :type cells: list

    :param extensions:
        The extent of the physical domain that the random field is defined
        on. This is only required if the random field is to be probed with
        global coordinates.
    :type list:

    :param covariance:
        The covariance structure that is used. `parafields` provides the
        following choices:

        * `exponential` (default choice)
        * `gammaExponential` (requires parameter `gammaExp`)
        * `separableExponential`
        * `matern` (requires parameter `maternNu`)
        * `matern32`
        * `matern52`
        * `gaussian`
        * `spherical`
        * `cauchy`
        * `generalizedCauchy`
        * `cubic`
        * `dampedOscillation`
        * `whiteNoise`

        Alternatively, you can pass a callable (e.g. a function or a class instance
        that defines __call__) to the covariance function. This allows you to use
        covariance functions defined in Python, but results in a significant performance
        penalty. This is currently limited to symmetric covariance functions.
    :type covariance: str or Callable

    :param variance:
        The variance of the random field.
    :type variance: float

    :param anisotropy:
        The type of anisotropy for the field. Can be one of the following:

        * `none` for an isotropic field
        * `axiparallel`
        * `geometric`
    :type anisotropy: str

    :param corrLength:
        The correlation length of the field. This can either be a scalar for
        an isotropic field or a list of length dimension for an anisotropic one
        or a row-wise dim x dim matrix for a geometric one.
    :type corrLength: float

    :param periodic:
        Whether the field should be periodic. Setting periodic boundary
        conditions sets embedding.factor = 1, i.e. behavior can't be
        controlled per boundary segment and correlation length must be
        small enough.
    :type periodic: bool

    :param embedding_factor:
        Relative size of extended domain (per dimension).
    :type embedding_factor: int

    :param embedding_type:
        Type of embedding. Can be one of "classical", "merge",
        "fold" or "cofold".
    :type embedding_factor: str

    :param sigmoid_function:
        Sigmoid function for merging, resp. smooth max for folding.
        Can be one of "smooth" or "smoothstep".
        smoothstep is better, but requires choice for recursion level.
    :type sigmoid_function: str

    :param threshold:
        Threshold for considering eigenvalues as negative
    :type threshold: float

    :param approximate:
        Whether to accept approximate results or not.
        Simply sets negative eigenvalues to zero if they occur.
    :type approximate: bool

    :param fftw_transpose:
        Whether FFTW should do transposed transforms.
    :type fftw_transpose: bool

    :param cacheInvMatvec:
        Whether matvecs with inverse covariance matrix are cached
    :type cacheInvMatvec: bool

    :param cacheInvRootMatvec:
        Whether matvecs with approximate root of inv. cov. matrix are cached
    :type cacheInvMatvec: bool

    :param cg_iterations:
        Conjugate Gradients iterations for matrix inverse multiplication
    :type cg_iterations: int

    :param cauchy_alpha:
        The Cauchy Alpha parameter for generalizedCauchy covariance
    :type cauchy_alpha: float

    :param cauchy_beta:
        The Cauchy Beta parameter for generalizedCauchy covariance
    :type cauchy_beta: float

    :param exp_gamma:
        The gamma value for gammaExponential covariance
    :type exp_gamma: float

    :param transform:
        A transformation that should be applied to the raw gaussian random
        field after evaluation. This can either be a Python callable accepting
        and returning an array of values or a string to select one of these
        pre-defined transformations:

        * `lognormal`: Applies the exponential to the field, thereby
          producing a log-normal random field.
        * `foldednormal`: Applied the absolute value to the field, thereby
          producting folded normal fields.
        * `sign`: Applies the sign function to the field, thereby producing
          binary fields that can e.g. be used to generate random subdomains.
    :type transform: str or Callable

    :param dtype:
        The floating point type to use. If the matching C++ type has not been
        compiled into the backend, an error is thrown.
    :type dtype: np.dtype

    :param seed:
        The seed for the random number generator. This can either be an integer
        to reproduce a field for the given seed or `None` which would generate
        a new seed.
    :type seed: int

    :param partitioning:
        The tuple with processors per direction. The product of all entries
        is expected to match the number of processors in the communicator.
        Alternatively, a function can be provided that accepts the number of
        processors and the cell sizes as arguments.
    :type partitioning:

    :param comm:
        The mpi4py communicator that should be used to distribute this
        random field. Defaults to MPI_COMM_WORLD. Specifying this parameter
        when using sequential builds for parafields results in an error.

    :returns:
        A random field instance.
    :rtype: RandomField
    """

    if fftw_transpose is None:
        fftw_transpose = len(cells) > 1

    cov_func = None
    if isinstance(covariance, collections.abc.Callable):
        cov_func = covariance
        covariance = "custom-iso"

    # Create the backend configuration
    backend_config = {
        "grid": {"cells": list(cells), "extensions": list(extensions)},
        "stochastic": {
            "anisotropy": anisotropy,
            "corrLength": corrLength,
            "covariance": covariance,
            "variance": variance,
            "cauchyAlpha": cauchy_alpha,
            "cauchyBeta": cauchy_beta,
            "expGamma": exp_gamma,
        },
        "embedding": {
            "approximate": approximate,
            "factor": embedding_factor,
            "periodization": embedding_type,
            "sigmoid": sigmoid_function,
            "threshold": threshold,
        },
        "fftw": {"transposed": fftw_transpose},
        "randomField": {
            "cacheInvMatvec": cacheInvMatvec,
            "cacheInvRootMatvec": cacheInvRootMatvec,
            "cg_iterations": cg_iterations,
            "periodic": periodic,
        },
    }

    frontend_config = {
        "transform": transform,
        "dtype": dtype,
        "partitioning": partitioning,
        "comm": comm,
        "covariance_function": cov_func,
    }

    # Return the Python class representing the field
    field = RandomField(backend_config, **frontend_config)

    field.generate(seed=seed)

    return field


# A mapping of numpy types to C++ type names
possible_types = {np.float64: "double", np.float32: "float"}

# Restriction of types that parafields was compiled with
available_types = {
    dt: t for dt, t in possible_types.items() if _parafields.has_precision(t)
}

# The mapping of built-in transformations
transformation_mapping = {
    "lognormal": np.exp,
    "foldednormal": np.abs,
    "sign": np.sign,
}


class RandomField:
    def __init__(
        self,
        config,
        transform=None,
        dtype=np.float64,
        partitioning=None,
        comm=None,
        covariance_function=None,
    ):
        """Create a random field from a ready backend configuration

        :param config:
            A nested dictionary containing a valid backend configuration.
        :type config: dict

        :param transform:
            A transformation that should be applied to the raw gaussian random
            field after evaluation. This can either be a Python callable accepting
            and returning an array of values or a string to select one of these
            pre-defined transformations:

            * `lognormal`: Applies the exponential to the field, thereby
            producing a log-normal random field.
            * `foldednormal`: Applied the absolute value to the field, thereby
            producting folded normal fields.
            * `sign`: Applies the sign function to the field, thereby producing
            binary fields that can e.g. be used to generate random subdomains.
        :type transform: str or Callable

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

        # Validate the given config
        self.config = validate_config(config)
        self.seed = None
        self.transform = transform
        self.covariance_function = covariance_function

        # Ensure that the given dtype is supported by parafields
        if dtype not in possible_types:
            raise NotImplementedError("Dtype not supported by parafields!")
        if dtype not in available_types:
            raise NotImplementedError(
                "parafields was not compiler for dtype, but could be!"
            )

        # Extract the partitioning (function)
        if partitioning is None:
            partitioning = default_partitioning

        if MPI is not None:
            # We use COMM_WORLD as the default communicator if running in parallel
            if comm is None:
                comm = MPI.COMM_WORLD
            # If the given partitioning is a function, call it
            if isinstance(partitioning, collections.abc.Callable):
                partitioning = partitioning(
                    MPI.COMM_WORLD.size, self.config["grid"]["cells"]
                )

        # Instantiate a C++ class for the field generator
        dim = len(self.config["grid"]["extensions"])
        FieldType = getattr(_parafields, f"RandomField{dim}D_{available_types[dtype]}")

        if comm is None:
            self._field = FieldType(dict_to_parameter_tree(self.config))
        else:
            self._field = FieldType(
                dict_to_parameter_tree(self.config), list(partitioning), comm
            )

        # Marker whether the field has been generated
        self._generated = False

        # Storage for lazy evaluation
        self._eval = None

    @property
    def dimension(self):
        return len(self.config["grid"]["cells"])

    def _add_trend_component(self, config):
        # Invalidate cached evaluations
        self._eval = None

        # Validate the given configuration
        config = validate_config(config, schema="trend.json")

        # Re-arrange the configuration to fit the backend. This
        # is because the backend uses a rather unintuitive way of
        # stacking unrelated parameters.
        if "disk0" in config:
            config["disk0"] = {
                "mean": config["disk0"]["mean_position"]
                + [config["disk0"]["mean_radius"], config["disk0"]["mean_height"]],
                "variance": config["disk0"]["variance_position"]
                + [
                    config["disk0"]["variance_radius"],
                    config["disk0"]["variance_height"],
                ],
            }
        if "block0" in config:
            config["block0"] = {
                "mean": config["block0"]["mean_position"]
                + config["block0"]["mean_extent"]
                + [config["block0"]["mean_height"]],
                "variance": config["block0"]["variance_position"]
                + config["block0"]["variance_extent"]
                + [config["block0"]["variance_height"]],
            }

        # Add the trend component in the backend
        self._field.add_trend_component(dict_to_parameter_tree(config))

        # Return self to allow chaining component additions
        return self

    def add_mean_trend_component(self, mean=1.0, variance=1.0):
        """Add a mean trend component to the field"""

        return self._add_trend_component({"mean": {"mean": mean, "variance": variance}})

    def add_slope_trend_component(self, mean=None, variance=None):
        """Add a slope trend component to the field"""

        # Apply field-dependent defaults
        if mean is None:
            mean = [1.0] * self.dimension

        if variance is None:
            variance = [1.0] * self.dimension

        # Add the component
        return self._add_trend_component(
            {"slope": {"mean": mean, "variance": variance}}
        )

    def add_disk_trend_component(
        self,
        mean_position=None,
        variance_position=None,
        mean_radius=0.05,
        variance_radius=0.01,
        mean_height=0.5,
        variance_height=0.1,
    ):
        """Add a disk trend component to the field"""

        # Apply field-dependent defaults
        if mean_position is None:
            mean_position = [0.5] * self.dimension

        if variance_position is None:
            variance_position = [0.1] * self.dimension

        # Add the component
        return self._add_trend_component(
            {
                "disk0": {
                    "mean_position": mean_position,
                    "variance_position": variance_position,
                    "mean_radius": mean_radius,
                    "variance_radius": variance_radius,
                    "mean_height": mean_height,
                    "variance_height": variance_height,
                }
            }
        )

    def add_block_trend_component(
        self,
        mean_position=None,
        variance_position=None,
        mean_extent=None,
        variance_extent=None,
        mean_height=0.5,
        variance_height=0.1,
    ):
        """Add a block trend component to the field"""

        # Apply field-dependent defaults
        if mean_position is None:
            mean_position = [0.5] * self.dimension

        if variance_position is None:
            variance_position = [0.1] * self.dimension

        if mean_extent is None:
            mean_extent = [0.5] * self.dimension

        if variance_extent is None:
            variance_extent = [0.1] * self.dimension

        # Add the component
        return self._add_trend_component(
            {
                "block0": {
                    "mean_position": mean_position,
                    "variance_position": variance_position,
                    "mean_extent": mean_extent,
                    "variance_extent": variance_extent,
                    "mean_height": mean_height,
                    "variance_height": variance_height,
                }
            }
        )

    def generate(self, seed=None):
        """Regenerate the field with the given seed

        :param seed:
            The seed to use. If seed is None, a new seed will be used
            on every call.
        :type seed: int
        """

        # Maybe calculate covariance
        if self.covariance_function is not None:
            if self.config["stochastic"]["covariance"] not in (
                "custom-aniso",
                "custom-iso",
            ):
                raise ValueError("Conflicting definition of covariance in backend!")
            self._field.compute_covariance(self.covariance_function)

        # Maybe create a new seed
        if seed is None:
            seed = time.time_ns() % (2**32)

        # Maybe invalidate any evaluations we have cached
        if seed != self.seed:
            self._eval = None
            self.seed = seed

            # Trigger field generation in the backend
            self._field.generate(seed)
            self._generated = True

    def evaluate(self):
        """Evaluate the random field

        :returns:
            A numpy array of the evaluations of the random field on
            the entire grid that the field is defined on.
        :rtype: np.ndarray
        """

        # Lazily evaluate the entire field
        if self._eval is None:
            # Maybe trigger generation
            if not self._generated:
                self.generate()

            self._eval = self._field.eval()

            # Apply transformation
            if self.transform is not None:
                # Look up transformation strings
                transform = self.transform
                if isinstance(transform, str):
                    transform = transformation_mapping[transform]

                self._eval = transform(self._eval)

        return self._eval

    def _repr_png_(self):
        """Print 2D random fields as images in Jupyter frontends"""

        # Evaluate the field
        eval_ = self.evaluate()

        # If this is not 2D, we skip visualization
        if len(eval_.shape) != 2:
            return

        # Convert to PIL array
        # Transposition is necessary because of conceptional differences between
        # numpy and pillow: https://stackoverflow.com/a/33727700
        img = Image.fromarray(np.uint8(cm.gist_earth(eval_.transpose()) * 255))

        # Ask PIL for the correct PNG repr
        return img._repr_png_()
