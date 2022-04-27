import numpy as np
import PIL.Image
import json
import requests
import base64
import io


def query(image_array: np.ndarray, crystal_structure: str, acceleration_voltage: int,
          convergence_angle: float, zone_u: int, zone_v: int, zone_w: int,
          host: str, port: int):
    '''
    Query the PACBED thickness and mistilt predictor service

    Parameters
    ----------

    image_array: numpy.ndarray
        Image data as NumPy array
    crystal_structure: str
        Identifier for the crystal structure. "Rutile" and "Strontium titanate"
        are supported in the current demo.
    acceleration_voltage: int
        Acceleration voltage in V. 80,000 and 300,000 are supported in the current demo.
    convergence_angle: float
        Convergence angle in mrad. 20 is supported in the current demo
    zone_u, zone_v, zone_w: int
        Zone axis. <0 0 1> is supported in the current demo.
    host: str
        Host of the web service
    port: int
        Port of the web service

    Returns
    -------
    dict
        Dictionary with the following keys:
        'thickness': float in Ångström
        'mistilt': float in mrad
        'scale': float, re-scaling of the PACBED pattern
        'validation': numpy.ndarray, RGB array with a validation plot
    '''
    resp = requests.post(f"http://{host}:{port}/inference/", files={
        "file": ("pacbed.raw", bytes(image_array), "application/octet-stream"),
        "parameters": (None, json.dumps({
            "file_params": {
                "typ": "raw",
                "dtype": str(image_array.dtype),
                "width": image_array.shape[1],
                "height": image_array.shape[0],
            },
            "physical_params": {
                "acceleration_voltage": acceleration_voltage,
                "zone_axis": {"u": zone_u, "v": zone_v, "w": zone_w},
                "crystal_structure": crystal_structure,
                "convergence_angle": convergence_angle,
            }
        }), "application/json"),
    }, )
    resp = resp.json()
    png_bytes = base64.b64decode(resp['validation'])
    bytes_io = io.BytesIO(png_bytes)

    pil_image = PIL.Image.open(bytes_io, formats=['PNG'])

    rgb = np.asarray(pil_image)
    resp['validation'] = rgb
    return resp


def arrayfromID(DM, id: int):
    '''
    Helper function for GMS Python
    to return an image as array.
    '''
    image = DM.FindImageByID(id)
    return image.GetNumArray()


def imagefromresponse(DM, resp, namespace: str = 'pacbed'):
    '''
    Helper function for GMS python to return the web service result to DMScript.

    Python code can be called from DMScript, but can't return values directly.
    Furthermore, GMS Python doesn't allow to create RGB images yet.

    This helper function creates separate images from the RGB channels of the
    validation result, assigns names in the specified namespace, and attaches
    the response data as image tags to the "red" channel. By default, the images
    have names 'pacbed:viz_r', 'pacbed:viz_g' and 'pacbed:viz_b' for the r, g, b
    channels. DMScript code can then "pick up" the response under these names
    and delete the temporary images.

    Parameters
    ----------

    DM: module
        The "DM" module which is mapped into the GMS Python namepace.
        Supplied as a parameter to avoid issues in environments where
        it is not available.
    resp: dict
        The result of calling :meth:`query`.
    namespace: str
        String to prepend to the image names, by default 'pacbed'.
    '''
    rgb = resp['validation']
    r_ = DM.CreateImage(rgb[:, :, 0].copy())
    g_ = DM.CreateImage(rgb[:, :, 1].copy())
    b_ = DM.CreateImage(rgb[:, :, 2].copy())

    r_.SetName(f'{namespace}:viz_r')
    g_.SetName(f'{namespace}:viz_g')
    b_.SetName(f'{namespace}:viz_b')

    tags = r_.GetTagGroup()
    for key in ('thickness', 'mistilt', 'scale'):
        tags.SetTagAsFloat(key, resp[key])
    r_.ShowImage()
    g_.ShowImage()
    b_.ShowImage()
