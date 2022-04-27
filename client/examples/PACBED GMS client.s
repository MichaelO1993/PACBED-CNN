// Minimal example for GMS integration
// Send the front image to the PACBED thickness prediction service
// and show result

Image front := GetFrontImage()
number id = front.ImageGetID()

// Build Python script as a string to query the service.
// arrayfromID and imagefromresponse are helper functions to facilitate
// data exchange within GMS.
string pyScript = "from pacbedclient import imagefromresponse, query, arrayfromID; "
pyScript += "imagefromresponse(DM, query(image_array=arrayfromID(DM, " + id + "), "
pyScript += "crystal_structure='Rutile', acceleration_voltage=80000, "
pyScript += "convergence_angle=20, zone_u=0, zone_v=0, zone_w=1, "
pyScript += "host='localhost', port=8230))"

ExecutePythonScriptString( pyScript, 1 )

// Pick up the images that imagefromresponse created.
// This is a work-around to return data from Python code
// to DMScript
image r_img := GetNamedImage("pacbed:viz_r")
image g_img := GetNamedImage("pacbed:viz_g")
image b_img := GetNamedImage("pacbed:viz_b")

// Create RGB image from the r, g, b channels and show
image validation := rgb(r_img, g_img, b_img)
validation.ShowImage()

// The prediction result is stored as image tags
TagGroup sourcetags = imagegettaggroup(r_img)
TagGroup targettags = imagegettaggroup(validation)

taggroupcopytagsfrom(targettags,sourcetags)

// Delete the temporary images that were used
// for data transfer
DeleteImage(r_img)
DeleteImage(g_img)
DeleteImage(b_img)
