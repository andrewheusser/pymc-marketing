#   Copyright 2022 - 2025 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Base class for conversion funnel models."""

import arviz as az
import pandas as pd
import pymc as pm
from pydantic import Field, validate_call

from pymc_marketing.base import ModelBuilder


class FunnelModelBuilder(ModelBuilder):
    """Base class for conversion funnel models."""

    model: pm.Model
    _model_type = "BaseFunnel"
    version = "0.0.1"

    @validate_call
    def __init__(
        self,
        step_column: str = Field(
            ..., description="Column name of the conversion funnel steps."
        ),
        step_order: list[str] = Field(
            ..., description="Order of the conversion funnel steps."
        ),
        variant_column: str = Field(
            ..., description="Column name of the variant variables."
        ),
        outcome_column: str = Field(
            ..., description="Column name of the outcome variable."
        ),
        model_config: dict | None = Field(None, description="Model configuration."),
        sampler_config: dict | None = Field(None, description="Sampler configuration."),
    ) -> None:
        self.step_column: str = step_column
        self.variant_column: str = variant_column
        self.step_order: list[str] = step_order
        self.outcome_column: str = outcome_column
        self.n_step: int = len(step_order)

        self.X: pd.DataFrame

        self._fit_result: az.InferenceData
        self._posterior_predictive: az.InferenceData
        super().__init__(model_config=model_config, sampler_config=sampler_config)
