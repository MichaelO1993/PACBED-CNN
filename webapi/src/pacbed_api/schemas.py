from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel


# FIXME: might need to canonicalize zone axis
class ZoneAxis(BaseModel):
    u: int
    v: int
    w: int


class PACBEDAnalysisParams(BaseModel):
    acceleration_voltage: int  # in V
    crystal_structure: str  # FIXME: "Strontium titanate" | "Rutile"; later: CIF files or whatever
    zone_axis: ZoneAxis
    convergence_angle: float  # mrad


class DType(str, Enum):
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    float32 = "float32"
    float64 = "float64"


class RawParams(BaseModel):
    typ: Literal['raw']
    dtype: DType
    width: int
    height: int


class DM4Params(BaseModel):
    typ: Literal['dm4']


class InferenceParameters(BaseModel):
    physical_params: PACBEDAnalysisParams
    file_params: Union[RawParams, DM4Params]


class InferenceResults(BaseModel):
    thickness: float  # in Ångstrom
    mistilt: float  # mrad
    scale: float  # unitless?
    validation: str
    # TODO: include confidence of prediction?
