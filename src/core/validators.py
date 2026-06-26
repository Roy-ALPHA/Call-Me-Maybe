from pydantic import BaseModel
from typing import Literal


class TypeDef(BaseModel):
    """Validate a single parameter or return type definition."""

    type: Literal["string", "number", "boolean", "integer"]


class FuncDef(BaseModel):
    """Validate a function definition used by the engine.

    Attributes:
        name: The function identifier.
        description: A natural-language description of the function.
        parameters: A mapping of parameter names to their types.
        returns: The return type definition.
    """

    name: str
    description: str
    parameters: dict[str, TypeDef]
    returns: TypeDef


class FuncsDef(BaseModel):
    """Validate the full list of available function definitions."""

    funcs: list[FuncDef]


class Prompt(BaseModel):
    """Validate a single user prompt entry."""

    prompt: str


class Prompts(BaseModel):
    """Validate the full list of input prompts."""

    prompts: list[Prompt]
