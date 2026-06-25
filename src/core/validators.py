from pydantic import BaseModel, model_validator
from typing import Literal

class TypeDef(BaseModel):
    type: Literal["string", "number", "boolean"] 

class FuncDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeDef]
    returns: TypeDef

class FuncsDef(BaseModel):
    funcs: list[FuncDef]

class Prompt(BaseModel):
    prompt: str

class Prompts(BaseModel):
    prompts: list[Prompt]