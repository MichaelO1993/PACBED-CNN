from enum import Enum

from pydantic import BaseModel

class ZoneAxis(BaseModel):
    u: int
    v: int
    w: int


class Point2D(BaseModel):
    y: float
    x: float


class PACBEDAnalysisParams(BaseModel):
    acceleration_voltage: float  # in V
    crystal_structure: str  # FIXME: "SrTiO3" | "Rutile"; later: CIF files or whatever
    zone_axis: ZoneAxis
    convergence_angle: float  # mrad?
    center: Point2D


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


class InferenceParameters(BaseModel):
    physical_params: PACBEDAnalysisParams
    dtype: DType
    width: int
    height: int


class InferenceResults(BaseModel):
    thickness: float  # in Angstrom?
    mistilt: float  # mrad?
    # TODO: include confidence of prediction?