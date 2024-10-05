class Deseasonalizer(BaseTransformer):
    """Remove seasonal components from a time series.

    Applies `statsmodels.tsa.seasonal.seasonal_compose` and removes the `seasonal`
    component in `transform`. Adds seasonal component back again in `inverse_transform`.
    Seasonality removal can be additive or multiplicative.

    `fit` computes :term:`seasonal components <Seasonality>` and
    stores them in `seasonal_` attribute.

    `transform` aligns seasonal components stored in `seasonal_` with
    the time index of the passed :term:`series <Time series>` and then
    substracts them ("additive" model) from the passed :term:`series <Time series>`
    or divides the passed series by them ("multiplicative" model).

    Parameters
    ----------
    sp : int, default=1
        Seasonal periodicity.
    model : {"additive", "multiplicative"}, default="additive"
        Model to use for estimating seasonal component.

    Attributes
    ----------
    seasonal_ : array of length sp
        Seasonal components computed in seasonal decomposition.

    See Also
    --------
    ConditionalDeseasonalizer

    Notes
    -----
    For further explanation on seasonal components and additive vs.
    multiplicative models see
    `Forecasting: Principles and Practice <https://otexts.com/fpp3/components.html>`_.
    Seasonal decomposition is computed using `statsmodels
    <https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html>`_.

    Examples
    --------
    >>> from aeon.transformations.detrend import Deseasonalizer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = Deseasonalizer()  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        "input_data_type": "Series",
        # what is the abstract type of X: Series, or Panel
        "output_data_type": "Series",
        # what abstract type is returned: Primitives, Series, Panel
        "instancewise": True,  # is this an instance-wise transform?
        "X_inner_type": "pd.Series",
        "y_inner_type": "None",
        "fit_is_empty": False,
        "capability:inverse_transform": True,
        "transform-returns-same-time-index": True,
        "capability:multivariate": False,
        "python_dependencies": "statsmodels",
    }

    def __init__(self, sp=1, model="additive"):
        self.sp = check_sp(sp)
        allowed_models = ("additive", "multiplicative")
        if model not in allowed_models:
            raise ValueError(
                f"`model` must be one of {allowed_models}, " f"but found: {model}"
            )
        self.model = model
        self._X = None
        self.seasonal_ = None
        super().__init__()

    def _align_seasonal(self, X):
        """Align seasonal components with X's time index."""
        shift = (
            -_get_duration(
                X.index[0],
                self._X.index[0],
                coerce_to_int=True,
                unit=_get_freq(self._X.index),
            )
            % self.sp
        )
        return np.resize(np.roll(self.seasonal_, shift=shift), X.shape[0])

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series
            Data to fit transform to
        y : ignored argument for interface compatibility

        Returns
        -------
        self: a fitted instance of the estimator
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        self._X = X
        sp = self.sp

        # apply seasonal decomposition
        self.seasonal_ = seasonal_decompose(
            X,
            model=self.model,
            period=sp,
            filt=None,
            two_sided=True,
            extrapolate_trend=0,
        ).seasonal.iloc[:sp]
        return self

    def _private_transform(self, y, seasonal):
        if self.model == "additive":
            return y - seasonal
        else:
            return y / seasonal

    def _private_inverse_transform(self, y, seasonal):
        if self.model == "additive":
            return y + seasonal
        else:
            return y * seasonal

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series
            transformed version of X, detrended series
        """
        seasonal = self._align_seasonal(X)
        Xt = self._private_transform(X, seasonal)
        return Xt

    def _inverse_transform(self, X, y=None):
        """Logic used by `inverse_transform` to reverse transformation on `X`.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be inverse transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            inverse transformed version of X
        """
        seasonal = self._align_seasonal(X)
        Xt = self._private_inverse_transform(X, seasonal)
        return Xt

    def _update(self, X, y=None, update_params=False):
        """Update transformer with X and y.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X : pd.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        X_full = X.combine_first(self._X)
        self._X = X_full
        if update_params:
            self._fit(X_full, update_params=update_params)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {}

        params2 = {"sp": 2}

        return [params, params2]


class ConditionalDeseasonalizer(Deseasonalizer):
    """Remove seasonal components from time series, conditional on seasonality test.

    Fit tests for :term:`seasonality <Seasonality>` and if the passed time series
    has a seasonal component it applies seasonal decomposition provided by `statsmodels
    <https://www.statsmodels.org>`
    to compute the seasonal component.
    If the test is negative `_seasonal` is set
    to all ones (if `model` is "multiplicative")
    or to all zeros (if `model` is "additive").

    Transform aligns seasonal components stored in `seasonal_` with
    the time index of the passed series and then
    substracts them ("additive" model) from the passed series
    or divides the passed series by them ("multiplicative" model).


    Parameters
    ----------
    seasonality_test : callable or None, default=None
        Callable that tests for seasonality and returns True when data is
        seasonal and False otherwise. If None,
        90% autocorrelation seasonality test is used.
    sp : int, default=1
        Seasonal periodicity.
    model : {"additive", "multiplicative"}, default="additive"
        Model to use for estimating seasonal component.

    Attributes
    ----------
    seasonal_ : array of length sp
        Seasonal components.
    is_seasonal_ : bool
        Return value of `seasonality_test`. True when data is
        seasonal and False otherwise.

    See Also
    --------
    Deseasonalizer

    Notes
    -----
    For further explanation on seasonal components and additive vs.
    multiplicative models see
    `Forecasting: Principles and Practice <https://otexts.com/fpp3/components.html>`_.
    Seasonal decomposition is computed using `statsmodels
    <https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html>`_.

    Examples
    --------
    >>> from aeon.transformations.detrend import ConditionalDeseasonalizer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = ConditionalDeseasonalizer(sp=12)  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    def __init__(self, seasonality_test=None, sp=1, model="additive"):
        self.seasonality_test = seasonality_test
        self.is_seasonal_ = None
        super().__init__(sp=sp, model=model)

    def _check_condition(self, y):
        """Check if y meets condition."""
        if not callable(self.seasonality_test_):
            raise ValueError(
                f"`func` must be a function/callable, but found: "
                f"{type(self.seasonality_test_)}"
            )

        is_seasonal = self.seasonality_test_(y, sp=self.sp)
        if not isinstance(is_seasonal, (bool, np.bool_)):
            raise ValueError(
                f"Return type of `func` must be boolean, "
                f"but found: {type(is_seasonal)}"
            )
        return is_seasonal

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series
            Data to fit transform to
        y : ignored argument for interface compatibility

        Returns
        -------
        self: a fitted instance of the estimator
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        self._X = X
        sp = self.sp

        # set default condition
        if self.seasonality_test is None:
            self.seasonality_test_ = autocorrelation_seasonality_test
        else:
            self.seasonality_test_ = self.seasonality_test

        # check if data meets condition
        self.is_seasonal_ = self._check_condition(X)

        if self.is_seasonal_:
            # if condition is met, apply de-seasonalisation
            self.seasonal_ = seasonal_decompose(
                X,
                model=self.model,
                period=sp,
                filt=None,
                two_sided=True,
                extrapolate_trend=0,
            ).seasonal.iloc[:sp]
        else:
            # otherwise, set idempotent seasonal components
            self.seasonal_ = (
                np.zeros(self.sp) if self.model == "additive" else np.ones(self.sp)
            )

        return self

