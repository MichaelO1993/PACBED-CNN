from fastapi import FastAPI, UploadFile, File

from . import schemas

app = FastAPI()


@app.get("/")
async def root():
    """
    Returns the basic web form for uploading PACBED patterns to be analysed
    """
    return {"message": "Hello world!"}


@app.post("/inference/")
def inference(parameters: schemas.InferenceParameters, file: UploadFile = File(...)) -> schemas.InferenceResults:
    """
    Run CNN inference on the given PACBED pattern with the given parameters
    """
    # TODO: run CNN inference on `file` with `parameters`
    # TODO: return different result if the inference was not available immediately
    return schemas.InferenceResults(
        thickness=42.21,
        mistilt=7.21,
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