from pydantic import BaseModel, model_validator

class TypeDef(BaseModel):
    type: str

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