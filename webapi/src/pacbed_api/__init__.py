import functools
import asyncio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from pydantic.types import Json
import pandas as pd
import base64

from . import schemas
from .predictor import Predictor

app = FastAPI()


@app.get("/")
async def root():
    """
    Returns the basic web form for uploading PACBED patterns to be analysed
    """
    return {"message": "Hello world!"}

# Select folder with CNN models and labels for a specific system by its ID from Register.csv
parameters_prediction = {
    'id_model': 0,  # Model Id from the specific system model register
    'conv_angle': 20.0  # Used convergence angle for recording the measured PACBED
    }

predictors = {}


def load_models():
    df_system = pd.read_csv('./data/Register.csv', sep=';', index_col='id')
    for index, row in df_system[['material']].iterrows():
        params = parameters_prediction.copy()
        params['id_system'] = index
        predictors[row['material']] = Predictor(params)


load_models()


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


# Json typing of `parameters` form param:
# https://github.com/tiangolo/fastapi/issues/2387#issuecomment-906761427
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
    predictor = predictors[pp.crystal_structure]
    assert np.allclose(pp.convergence_angle, 20)
    pattern = np.frombuffer(file.file.read(), dtype=parameters.dtype).reshape(
        (parameters.height, parameters.width)
    )
    # pattern = np.zeros((parameters.height, parameters.width), dtype=np.float32)
    result = await sync_to_async(predictor.predict, None, pattern)
    validation = await sync_to_async(predictor.validate, None, result, pattern)
    print(result)
    return schemas.InferenceResults(
        thickness=result['thickness_pred'],
        mistilt=result['mistilt_pred'],
        scale=result['scale'],
        validation=base64.encodebytes(validation.getbuffer())
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
