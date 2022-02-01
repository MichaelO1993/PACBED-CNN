Image front := GetFrontImage()
number id = front.ImageGetID()

string pyScript = "from pacbedclient import imagefromresponse, query, arrayfromID; "
pyScript += "imagefromresponse(DM, query(arrayfromID(DM, " + id + ")))"



ExecutePythonScriptString( pyScript, 1 )

number nImg = CountImages()
for ( number i = 0 ; i < nImg ; i ++ )
{

 image img := FindImageByIndex(i)
 result(GetLabel( img ) + " " + GetName(img))
}

image r_img := GetNamedImage("pacbed:viz_r")
image g_img := GetNamedImage("pacbed:viz_g")
image b_img := GetNamedImage("pacbed:viz_b")

image validation := rgb(r_img, g_img, b_img)
validation.ShowImage()

TagGroup sourcetags = imagegettaggroup(r_img)
TagGroup targettags = imagegettaggroup(validation)

taggroupcopytagsfrom(targettags,sourcetags)

DeleteImage(r_img)
DeleteImage(g_img)
DeleteImage(b_img)
