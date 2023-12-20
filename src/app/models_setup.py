from enum import Enum

from pydantic import BaseModel, Field


class ModelName(str, Enum):
    decision_tree = 'decision_tree'
    logistic_regression = 'logistic_regression'


class DecisionTreeParams(BaseModel):
    criterion: str = Field(
        description='The function to measure the quality of a split',
        default='gini'
    )
    max_depth: int = Field(
        description='The maximum depth of the tree',
        default=3
    )
    min_samples_split: int = Field(
        description='The minimum number of samples required to split an internal node',
        default=5,
    )
    random_state: int = Field(
        description="Controls the randomness of the estimator",
        default=42
    )


class LogisticRegressionParams(BaseModel):
    penalty: str = Field(
        description='Specify the norm of the penalty',
        default='l2'
    )
    C: int = Field(
        description="""Inverse of regularization strength; must be a positive float. 
        Like in support vector machines, smaller values specify stronger regularization.""",
        default=1
    )
    max_iter: int = Field(
        description='Maximum number of iterations taken for the solvers to converge.',
        default=100,
    )
