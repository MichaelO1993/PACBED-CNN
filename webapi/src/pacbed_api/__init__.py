from os.path import join, normpath, abspath, dirname
import base64
import functools
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic.types import Json
import pandas as pd
from ncempy.io.dm import fileDM

from . import schemas
from .predictor import Predictor

app = FastAPI()

BASE_DIR = normpath(abspath(dirname(__file__)))

app.mount("/static", StaticFiles(directory=join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=join(BASE_DIR, "templates"))


INFERENCE_WORKERS = 4
INFERENCE_INTERNAL_THREADS = 4


@app.get("/")
async def root(request: Request):
    """
    Returns the basic web form for uploading PACBED patterns to be analysed
    """
    crystal_structures = [p[0] for p in predictors.keys()]

    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "crystal_structures": crystal_structures,
        }
    )


# Select folder with CNN models and labels for a specific system by its ID from Register.csv
parameters_prediction = {
    'id_model': 0,  # Model Id from the specific system model register
    }

predictors = {}

pool = ThreadPoolExecutor(max_workers=INFERENCE_WORKERS)

def load_models():
    df_system = pd.read_csv('./data/Register.csv', sep=';', index_col='id')
    for index, row in df_system[['material', 'high tension', 'direction']].iterrows():
        params = parameters_prediction.copy()
        params['id_system'] = index
        key = (row['material'], row['high tension'], row['direction'])
        predictors[key] = Predictor(params, num_threads=INFERENCE_INTERNAL_THREADS)


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


def get_pattern_from_dm(file):
    with fileDM(file) as dmf:
        ds = dmf.getDataset(0)
        return ds['data']


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
    key = (
        pp.crystal_structure,
        pp.acceleration_voltage // 1000,
        f'({pp.zone_axis.u}, {pp.zone_axis.v}, {pp.zone_axis.w})'
    )
    predictor = predictors[key]
    assert (pp.convergence_angle <= 25) and (pp.convergence_angle >= 15)

    fp = parameters.file_params
    if fp.typ == "dm4":
        pattern = await sync_to_async(get_pattern_from_dm, pool, file.file)
    elif fp.typ == "raw":
        dtype = fp.dtype
        width = fp.width
        height = fp.height
        pattern = np.frombuffer(await sync_to_async(file.file.read, pool), dtype=dtype).reshape(
            (height, width)
        )
    # pattern = np.zeros((parameters.height, parameters.width), dtype=np.float32)
    result, pacbed_pred_out = await sync_to_async(predictor.predict, pool, pattern, pp.convergence_angle)
    validation = await sync_to_async(predictor.validate, pool, result, pacbed_pred_out,  pp.convergence_angle)
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
