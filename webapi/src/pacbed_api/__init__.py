import functools
import asyncio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from pydantic.types import Json

from . import schemas
from .predictor import Predictor

app = FastAPI()


@app.get("/")
async def root():
    """
    Returns the basic web form for uploading PACBED patterns to be analysed
    """
    return {"message": "Hello world!"}

# Select folder with CNN models and labels (for a specific system)
path_models = './PACBED-CNN-data/Trained_Models_lite/'
dataframe_path = './PACBED-CNN-data/Trained_Models_lite/df.csv'
simulation_path = './PACBED-CNN-data/'

predictor = Predictor(
    path_models,
    dataframe_path,
    simulation_path
)


async def sync_to_async(fn, pool=None, *args, **kwargs):
    """
    Run blocking function with `*args`, `**kwargs` in a thread pool.

    Parameters
    ----------
    fn : callable
        The blocking function to run in a background thread

    pool : ThreadPoolExecutor or None
        In which thread pool should the function be run? If `None`, we create a new one

    *args, **kwargs
        Passed on to `fn`
    """
    loop = asyncio.get_event_loop()
    fn = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(pool, fn)


# Json typing of `parameters` form param: https://github.com/tiangolo/fastapi/issues/2387#issuecomment-906761427
@app.post("/inference/")
async def inference(
    parameters: Json[schemas.InferenceParameters] = Form(...),
    file: UploadFile = File(...),
) -> schemas.InferenceResults:
    """
    Run CNN inference on the given PACBED pattern with the given parameters
    """
    # TODO: return different result if the inference was not available immediately
    # TODO: validate parameters - they need to fit the models we have loaded
    pp = parameters.physical_params
    assert pp.acceleration_voltage == 80000
    assert pp.zone_axis == schemas.ZoneAxis(u=0, v=0, w=1)
    assert pp.crystal_structure == "Rutile"
    assert np.allclose(pp.convergence_angle, 20)
    pattern = np.frombuffer(file.file.read(), dtype=parameters.dtype).reshape(
        (parameters.height, parameters.width)
    )
    # pattern = np.zeros((parameters.height, parameters.width), dtype=np.float32)
    result = await sync_to_async(predictor.predict, None, pattern)
    print(result)
    return schemas.InferenceResults(
        thickness=result['thickness_pred'],
        mistilt=result['mistilt_pred'],
        scale=result['scale'],
    )


@app.get("/inference/delayed/{uuid:uuid}")
def delayed_result(uuid) -> schemas.InferenceResults:
    """
    A result that was not immediately available for inference, and needed
    training in the background. The uuid will be given by the `inference`
    result in this case.
    """
    return schemas.InferenceResults(
        thickness=42.21,
        mistilt=7.21,
    )


@app.get("/pattern/")
def pattern(parameters: schemas.PACBEDAnalysisParams, inferred: schemas.InferenceResults):
    """
    Get the simulated PACDEB pattern as an image, corresponding to the given parameters
    """
    return {}
